"""
Flask API Server — LSTM Phishing Detection
===========================================
Serves phishing-detection predictions from the character-level LSTM model.

Endpoints:
  GET  /health           → health check
  POST /analyze          → analyse full email (extracts URLs + heuristics)
  POST /predict          → predict a single URL

Security fixes applied:
  - CORS restricted to localhost:3000 only
  - Structured error responses (no raw tracebacks)
"""

import sys
import os

# UTF-8 wrapper handled by imported modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_lstm_api import PhishingDetector

app = Flask(__name__)

# ── FIX: Restrict CORS to localhost only (was wide-open before) ────────────────
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# ── Load model once at startup ─────────────────────────────────────────────────
print("[*] Loading GPU-trained character-level LSTM model...")
try:
    ml_dir   = os.path.dirname(os.path.abspath(__file__))
    detector = PhishingDetector(
        model_path    = os.path.join(ml_dir, "models", "lstm_model.pt"),     # UPDATED: PyTorch format
        char2idx_path = os.path.join(ml_dir, "models", "char2idx.pkl"),
    )
    print("[+] Model ready - GPU trained with regularization")
except Exception as exc:
    print(f"[-] Failed to load model: {exc}")
    detector = None
    sys.exit(1)


# ─── Helpers ───────────────────────────────────────────────────────────────────
URL_RE = re.compile(r'https?://[^\s<>")\]]*')

PHISHING_KEYWORDS = [
    "verify", "confirm", "secure", "update", "account", "login",
    "password", "click here", "urgent", "limited time", "act now",
    "you have won", "congratulations", "free gift", "suspended",
    "bank details", "credit card", "lottery", "paypal", "amazon",
]

def extract_urls(text: str) -> list[str]:
    """Extract all HTTP(S) URLs from raw text."""
    return URL_RE.findall(text)

def text_heuristics(content: str) -> dict:
    """
    FIX: Analyse plain-text body even when no URLs are present.
    Returns a base risk score and detected keywords.
    """
    lower = content.lower()
    found = [kw for kw in PHISHING_KEYWORDS if kw in lower]
    base_score = min(len(found) * 12, 80)   # 0–80
    return {"keywords": found, "base_score": base_score}


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": detector is not None}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyse full email content for phishing threats."""
    try:
        data        = request.get_json(force=True) or {}
        content     = data.get("content", "")
        # attachments = data.get("attachments", [])   # reserved for future use

        if not content:
            return jsonify({"error": "Missing email content"}), 400

        # 1. Extract URLs
        urls = extract_urls(content)
        print(f"[EMAIL] {len(urls)} URLs found")

        # 2. Text heuristics (catches phishing with no links)
        heuristics = text_heuristics(content)

        # 3. Batch-predict all URLs at once
        threats, phishing_urls, risk_scores = [], [], []

        if urls:
            results = detector.predict_batch(urls, threshold=0.45)
            for r in results:
                risk_scores.append(r["probability"] or 0)
                if r["is_phishing"]:
                    phishing_urls.append({
                        "url":        r["url"],
                        "risk_level": r["risk_level"],
                        "confidence": r["confidence"],
                    })
                    threats.append(
                        f"[ALERT] Phishing URL: {r['url']} "
                        f"(Risk: {r['risk_level']}, confidence {r['confidence']:.0%})"
                    )
                    print(f"  [PHISHING] {r['url']} → {r['risk_level']}")
                else:
                    print(f"  [SAFE]     {r['url']}")

        # 4. Determine overall risk
        max_url_prob = max(risk_scores) if risk_scores else 0.0

        if phishing_urls:
            has_critical = any(u["risk_level"] == "CRITICAL" for u in phishing_urls)
            overall_risk = "CRITICAL" if has_critical else "HIGH"
            summary = f"Detected {len(phishing_urls)} phishing URL(s) — immediate action required"

        elif not urls:
            # FIX: No-URL emails now evaluated by text heuristics
            kw_count = len(heuristics["keywords"])
            if kw_count >= 4:
                overall_risk = "HIGH"
                summary = (
                    f"No URLs but {kw_count} phishing keyword(s) detected "
                    f"({', '.join(heuristics['keywords'][:3])}…)"
                )
                threats.append(f"Suspicious language: {', '.join(heuristics['keywords'])}")
            elif kw_count >= 2:
                overall_risk = "MEDIUM"
                summary = f"Suspicious language patterns detected ({kw_count} keywords)"
                threats.append(f"Suspicious language: {', '.join(heuristics['keywords'])}")
            else:
                overall_risk = "LOW"
                summary = "No URLs and no suspicious language detected"

        else:
            # All URLs safe — blend url probs with text heuristics
            blended = max(max_url_prob * 100, heuristics["base_score"])
            if blended >= 60:
                overall_risk = "MEDIUM"
                summary = f"URLs appear safe but some suspicious patterns detected"
            else:
                overall_risk = "LOW"
                summary = f"All {len(urls)} URL(s) appear legitimate"

        # 5. Spam score
        spam_score  = int(max(max_url_prob * 100, heuristics["base_score"]))
        is_spam     = overall_risk in ("HIGH", "CRITICAL")
        spam_reason = (
            f"Detected {len(phishing_urls)} phishing URL(s)"
            if phishing_urls
            else (
                f"Suspicious keywords: {', '.join(heuristics['keywords'])}"
                if heuristics["keywords"]
                else "No phishing indicators found"
            )
        )

        recommendation = (
            "DO NOT click any links or reply — mark as phishing immediately"
            if overall_risk in ("CRITICAL", "HIGH")
            else "Use caution — verify sender identity before acting"
            if overall_risk == "MEDIUM"
            else "Email appears safe to read"
        )

        response = {
            "riskLevel":      overall_risk,
            "summary":        summary,
            "threats":        threats if threats else ["No direct threats detected"],
            "recommendation": recommendation,
            "isSpam":         is_spam,
            "spamScore":      spam_score,
            "spamReason":     spam_reason,
            "urls_analyzed":  len(urls),
            "phishing_urls":  phishing_urls,
            "model":          "Character-Level LSTM Phishing Detector",
        }

        print(f"[+] Analysis complete: {overall_risk} risk, spam={is_spam}, score={spam_score}")
        return jsonify(response), 200

    except Exception as exc:
        print(f"[-] Error in /analyze: {exc}")
        return jsonify({"error": "Internal analysis error"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict phishing risk for a single URL."""
    try:
        data = request.get_json(force=True) or {}
        url  = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "Missing URL"}), 400

        result = detector.predict(url, threshold=0.45)
        print(f"[+] Predict {url} → {result['label']} ({result['risk_level']})")
        return jsonify(result), 200

    except Exception as exc:
        print(f"[-] Error in /predict: {exc}")
        return jsonify({"error": "Internal prediction error"}), 500


if __name__ == "__main__":
    print("[>>] LSTM Phishing Detection API — Character-Level Model")
    print("[>>] Endpoints: POST /analyze  |  POST /predict  |  GET /health")
    app.run(host="0.0.0.0", port=5000, debug=False)
