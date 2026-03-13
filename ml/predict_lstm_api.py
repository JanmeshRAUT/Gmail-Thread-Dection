"""
Character-Level LSTM — PyTorch Prediction Module
==================================================
Loads the trained .pt checkpoint and performs inference on URLs.
Must match the architecture in train_lstm.py exactly.
"""

import sys, os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import torch
import torch.nn as nn

# ─── Constants (must match train_lstm.py) ─────────────────────────────────────
MAX_URL_LEN   = 100
EMBEDDING_DIM = 32
LSTM_HIDDEN   = 64         # UPDATED: Regularized for better generalization

# ─── Risk thresholds ──────────────────────────────────────────────────────────
THRESH_CRITICAL = 0.85
THRESH_HIGH     = 0.65
THRESH_MEDIUM   = 0.45
DECISION_THR    = 0.45   # probability above this → phishing

# ─── PyTorch model definition (mirrors train_lstm.py) ─────────────────────────
class PhishingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            dropout=0.4,                    # UPDATED: Increased for regularization
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.5)     # UPDATED: Increased dropout layers
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)     # UPDATED: Reduced from 64 to 32
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb    = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out.max(dim=1).values
        pooled = self.dropout1(pooled)
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)
        hidden = self.dropout2(hidden)      # UPDATED: Added dropout after ReLU
        return self.fc2(hidden).squeeze(1)


class PhishingDetector:
    """Load trained PyTorch LSTM and run phishing predictions."""

    def __init__(self, model_path: str = None, char2idx_path: str = None):
        ml_dir        = os.path.dirname(os.path.abspath(__file__))
        model_path    = model_path    or os.path.join(ml_dir, "models", "lstm_model.pt")
        char2idx_path = char2idx_path or os.path.join(ml_dir, "models", "char2idx.pkl")

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Loading GPU-trained LSTM model on {self.device} ...")

        checkpoint      = torch.load(model_path, map_location=self.device)
        vocab_size      = checkpoint["vocab_size"]
        embed_dim       = checkpoint.get("embed_dim",  EMBEDDING_DIM)
        hidden_dim      = checkpoint.get("hidden_dim", LSTM_HIDDEN)

        self.model = PhishingLSTM(vocab_size, embed_dim, hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.char2idx = joblib.load(char2idx_path)
        self.max_len  = MAX_URL_LEN
        print(f"[✓] GPU-trained model ready!")
        print(f"    Device: {self.device} | Vocab: {vocab_size} | Hidden dim: {hidden_dim}")
        print(f"    Regularization: Dropout(0.5) + L2 penalty | Training data: 300k URLs")

    # ── Tokenisation ───────────────────────────────────────────────────────────
    def _tokenise(self, url: str) -> list:
        seq = [self.char2idx.get(c, 1) for c in str(url)[:self.max_len]]
        seq += [0] * (self.max_len - len(seq))
        return seq[:self.max_len]

    # ── Risk mapping ───────────────────────────────────────────────────────────
    @staticmethod
    def _risk(prob: float) -> str:
        if prob >= THRESH_CRITICAL: return "CRITICAL"
        if prob >= THRESH_HIGH:     return "HIGH"
        if prob >= THRESH_MEDIUM:   return "MEDIUM"
        return "LOW"

    # ── Single prediction ──────────────────────────────────────────────────────
    def predict(self, url: str, threshold: float = DECISION_THR) -> dict:
        try:
            seq  = torch.tensor([self._tokenise(url)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logit = self.model(seq)
                prob  = float(torch.sigmoid(logit).item())

            is_phishing = prob >= threshold
            return {
                "url":              url,
                "is_phishing":      is_phishing,
                "probability":      prob,
                "prediction_class": int(is_phishing),
                "label":            "phishing" if is_phishing else "benign",
                "risk_level":       self._risk(prob),
                "confidence":       float(max(prob, 1 - prob)),
            }
        except Exception as exc:
            print(f"[!] predict error for {url}: {exc}")
            return {
                "url": url, "error": str(exc),
                "is_phishing": None, "probability": None, "risk_level": "UNKNOWN",
            }

    # ── Batch prediction ───────────────────────────────────────────────────────
    def predict_batch(self, urls: list, threshold: float = DECISION_THR) -> list:
        if not urls:
            return []
        try:
            X = torch.tensor(
                [self._tokenise(u) for u in urls], dtype=torch.long
            ).to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(X)).cpu().numpy()

            return [
                {
                    "url":              url,
                    "is_phishing":      float(p) >= threshold,
                    "probability":      float(p),
                    "prediction_class": int(float(p) >= threshold),
                    "label":            "phishing" if float(p) >= threshold else "benign",
                    "risk_level":       self._risk(float(p)),
                    "confidence":       float(max(float(p), 1 - float(p))),
                }
                for url, p in zip(urls, probs)
            ]
        except Exception as exc:
            return [self.predict(u, threshold) for u in urls]

    # ── Explainability ─────────────────────────────────────────────────────────
    def explain_prediction(self, url: str) -> dict:
        result  = self.predict(url)
        url_str = str(url)
        patterns = []
        if "@" in url_str:
            patterns.append("@ symbol detected — credential redirection risk")
        if url_str.count("-") > 3:
            patterns.append(f"Excessive hyphens ({url_str.count('-')}) — domain spoofing")
        if len(url_str) > 75:
            patterns.append(f"Long URL ({len(url_str)} chars) — obfuscation risk")
        if url_str.count("%") > 2:
            patterns.append("Heavy URL encoding — possible obfuscation")
        if any(k in url_str.lower() for k in ["login","verify","secure","confirm","update","account"]):
            patterns.append("Phishing keyword found in URL path")
        return {
            "url": url,
            "prediction": result,
            "suspicious_patterns": patterns,
            "summary": (
                f"URL classified as {result['label'].upper()} with {result['risk_level']} risk "
                f"(confidence {result['confidence']:.1%})"
            ),
        }


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    d = PhishingDetector()
    tests = [
        "https://www.google.com",
        "https://github.com/openai/openai-python",
        "http://verify-your-account-click-here.xyz",
        "https://paypal-secure-confirm.malicious.ru/reset",
        "https://apple-id-verify.site/login?user=john@email.com&redirect=%2F",
    ]
    print("\n" + "="*65)
    print("PyTorch LSTM — SELF-TEST")
    print("="*65)
    for url in tests:
        r = d.predict(url)
        flag = "🚨" if r["is_phishing"] else "✅"
        print(f"{flag} [{r['risk_level']:8s}] p={r['probability']:.3f}  {url}")
