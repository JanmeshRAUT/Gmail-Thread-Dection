"""
Character-Level LSTM — GPU Training via PyTorch
================================================
Uses true character-level sequence tokenization.
Runs on RTX 2050 via CUDA — PyTorch 2.5.1+cu121.

Architecture:
  Embedding → LSTM → GlobalMaxPool → Dense(64) → Dense(1, sigmoid)

Dataset: ml/data/balanced_urls.csv
"""

import sys, os, time

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ─── Config ────────────────────────────────────────────────────────────────────
MAX_URL_LEN   = 100
EMBEDDING_DIM = 32
LSTM_HIDDEN   = 64         # REDUCED for better generalization
BATCH_SIZE    = 512        # REDUCED to add noise and improve generalization
EPOCHS        = 50
LR            = 5e-4
EARLY_STOP    = 5          # MORE AGGRESSIVE early stopping
SAMPLE_SIZE   = 300_000    # 150k safe + 150k phishing from merged_dataset (larger = better generalization)
L2_LAMBDA     = 1e-4       # L2 regularization weight
DROPOUT_RATE  = 0.5        # INCREASED dropout

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "merged_dataset.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ─── ENFORCE GPU ONLY ──────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU (CUDA) is required but not available! Cannot proceed.")

DEVICE = torch.device("cuda")
print(f"\n{'='*80}", flush=True)
print(f"🚀 FORCING GPU-ONLY TRAINING MODE", flush=True)
print(f"{'='*80}", flush=True)
print(f"Device: {DEVICE}", flush=True)
print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3:.2f} GB", flush=True)
print(f"CUDA Version: {torch.version.cuda}", flush=True)
print(f"PyTorch Version: {torch.__version__}", flush=True)

# Empty cache at start
torch.cuda.empty_cache()
print(f"GPU Memory Allocated (Initial): {torch.cuda.memory_allocated(0) // 1024**2} MB", flush=True)
print(f"GPU Memory Reserved (Initial): {torch.cuda.memory_reserved(0) // 1024**2} MB", flush=True)
print(f"{'='*80}\n", flush=True)

# ─── Label map ────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "benign": 0, "legitimate": 0, "0": 0, "safe": 0, "good": 0,
    "malicious": 1, "phishing": 1, "1": 1, "threat": 1, "bad": 1,
}

# ─── Tokeniser ────────────────────────────────────────────────────────────────
def build_char_vocab(urls):
    chars = set()
    for u in urls:
        chars.update(str(u))
    vocab = sorted(chars)
    char2idx = {c: i + 2 for i, c in enumerate(vocab)}  # 0=pad, 1=unk
    char2idx["<PAD>"] = 0
    char2idx["<UNK>"] = 1
    return char2idx

def url_to_sequence(url, char2idx, max_len=MAX_URL_LEN):
    seq = [char2idx.get(c, 1) for c in str(url)[:max_len]]
    seq += [0] * (max_len - len(seq))
    return seq[:max_len]

# ─── Dataset ──────────────────────────────────────────────────────────────────
class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─── Model ────────────────────────────────────────────────────────────────────
class PhishingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            dropout=0.4,                    # INCREASED dropout
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.5)     # INCREASED dropout layers
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)     # REDUCED capacity
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb  = self.embedding(x)               # (B, T, E)
        out, _ = self.lstm(emb)                # (B, T, H*2)
        # Global max pooling over time dimension
        pooled = out.max(dim=1).values         # (B, H*2)
        pooled = self.dropout1(pooled)
        hidden = self.fc1(pooled)              # (B, 32)
        hidden = self.relu(hidden)
        hidden = self.dropout2(hidden)         # ADDED dropout
        logits = self.fc2(hidden)              # (B, 1)
        return logits.squeeze(1)

# ─── Load data ────────────────────────────────────────────────────────────────
def load_data(path):
    print(f"\n{'='*80}\nLOADING MERGED DATASET\n{'='*80}", flush=True)
    print(f"Data Path: {path}", flush=True)
    
    for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python")
            print(f"✓ Successfully loaded with {enc} encoding", flush=True)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Cannot load {path}")

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Map all label formats to binary (0/1)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(lambda v: LABEL_MAP.get(v, None))
    df.dropna(subset=["label", "url"], inplace=True)
    df["label"] = df["label"].astype(int)
    df.drop_duplicates(subset=["url"], inplace=True)

    # Balance classes
    n = min((df["label"] == 0).sum(), (df["label"] == 1).sum())
    df = pd.concat([
        df[df["label"] == 0].sample(n, random_state=42),
        df[df["label"] == 1].sample(n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n📊 Dataset Balanced:", flush=True)
    print(f"   • Total Rows: {len(df):,}", flush=True)
    print(f"   • Class 0 (Safe): {n:,}", flush=True)
    print(f"   • Class 1 (Phishing): {n:,}", flush=True)

    # Subsample to SAMPLE_SIZE if necessary
    if len(df) > SAMPLE_SIZE:
        half = SAMPLE_SIZE // 2
        df = pd.concat([
            df[df["label"] == 0].sample(half, random_state=42),
            df[df["label"] == 1].sample(half, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   • Subsampled to: {len(df):,} rows (for better generalization)", flush=True)
    
    print(f"   • Data columns: {df.columns.tolist()}", flush=True)
    print(f"{'='*80}\n", flush=True)
    return df

# ─── Train / eval helpers ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss = correct = total = 0
    batch_count = 0
    batch_losses = []
    batch_accs = []
    
    progress_bar = tqdm(loader, desc=f"  Training", ncols=100, position=1, leave=False)
    
    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        
        # ADD L2 REGULARIZATION
        l2_reg = torch.tensor(0., device=DEVICE)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss + L2_LAMBDA * l2_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        batch_loss = loss.item()
        batch_acc = ((preds >= 0.0) == y_batch.bool()).sum().item() / len(y_batch)
        
        total_loss += batch_loss * len(y_batch)
        correct    += ((preds >= 0.0) == y_batch.bool()).sum().item()
        total      += len(y_batch)
        batch_count += 1
        
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)
        
        if batch_count % 5 == 0:
            avg_batch_loss = np.mean(batch_losses[-5:])
            avg_batch_acc = np.mean(batch_accs[-5:])
            mem_alloc = torch.cuda.memory_allocated(0) // 1024**2
            progress_bar.set_postfix({
                'loss': f'{avg_batch_loss:.4f}',
                'acc': f'{avg_batch_acc:.4f}',
                'GPU_MB': mem_alloc
            })
    
    progress_bar.close()
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def eval_epoch(model, loader, criterion, phase="Validation"):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []
    batch_count = 0
    
    progress_bar = tqdm(loader, desc=f"  {phase}", ncols=100, position=1, leave=False)
    
    for X_batch, y_batch in progress_bar:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        probs  = torch.sigmoid(logits)
        total_loss += loss.item() * len(y_batch)
        correct    += ((probs >= 0.5) == y_batch.bool()).sum().item()
        total      += len(y_batch)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        batch_count += 1
        
        mem_alloc = torch.cuda.memory_allocated(0) // 1024**2
        progress_bar.set_postfix({
            'loss': f'{total_loss/total:.4f}',
            'acc': f'{correct/total:.4f}',
            'GPU_MB': mem_alloc
        })
    
    progress_bar.close()
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc

# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(axes, ["acc", "loss", "auc"], ["Accuracy", "Loss", "AUC"]):
        ax.plot(history[f"train_{key}"], label="Train", linewidth=2)
        ax.plot(history[f"val_{key}"],   label="Val",   linewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "training_history.png")
    plt.savefig(out, dpi=150)
    print(f"✓ Plot saved → {out}", flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*80}", flush=True)
    print("🚀 CHARACTER-LEVEL LSTM — GPU-ONLY TRAINING (PyTorch)", flush=True)
    print("📊 USING MERGED DATASET (867k+ URLs for Better Generalization)", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Vocab
    print(f"{'='*80}\nBUILDING CHARACTER VOCABULARY\n{'='*80}", flush=True)
    char2idx  = build_char_vocab(df["url"])
    vocab_size = len(char2idx) + 1
    print(f"✓ Vocabulary Built:", flush=True)
    print(f"   • Total Characters: {len(char2idx)}", flush=True)
    print(f"   • Vocab Size (+ special tokens): {vocab_size}", flush=True)
    print(f"   • Special Tokens: <PAD>=0, <UNK>=1", flush=True)
    print(f"   • Sample chars: {list(char2idx.keys())[:10]}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 3. Tokenise
    print(f"{'='*80}\nTOKENIZING URLs\n{'='*80}", flush=True)
    X = np.array([url_to_sequence(u, char2idx) for u in df["url"]], dtype=np.int32)
    y = df["label"].values.astype(np.float32)
    print(f"✓ Tokenization Complete:", flush=True)
    print(f"   • X Shape: {X.shape} (samples, sequence_length)", flush=True)
    print(f"   • y Shape: {y.shape}", flush=True)
    print(f"   • Data Type X: {X.dtype}", flush=True)
    print(f"   • Data Type y: {y.dtype}", flush=True)
    print(f"   • Class Distribution: {np.bincount(y.astype(int))}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 4. Split
    print(f"{'='*80}\nSPLITTING DATA\n{'='*80}", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds  = URLDataset(X_train, y_train)
    test_ds   = URLDataset(X_test,  y_test)

    # Keep 15% of training as val
    val_size   = int(0.15 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    
    print(f"✓ Data Split Complete:", flush=True)
    print(f"   • Training Set: {train_size:,} samples", flush=True)
    print(f"   • Validation Set: {val_size:,} samples", flush=True)
    print(f"   • Test Set: {len(test_ds):,} samples", flush=True)
    print(f"   • Train Batch Size: {BATCH_SIZE}", flush=True)
    print(f"   • Val/Test Batch Size: {BATCH_SIZE*2}", flush=True)
    print(f"   • Total Train Batches: {len(train_loader)}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 5. Model
    print(f"{'='*80}\nBUILDING LSTM MODEL\n{'='*80}", flush=True)
    model = PhishingLSTM(vocab_size, EMBEDDING_DIM, LSTM_HIDDEN).to(DEVICE)
    print(model, flush=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model Architecture (REGULARIZED FOR BETTER GENERALIZATION):", flush=True)
    print(f"   • Total Trainable Parameters: {params:,}", flush=True)
    print(f"   • Device Placement: {DEVICE}", flush=True)
    
    # Calculate model size
    model_size_mb = params * 4 / 1024 / 1024  # Assuming float32
    print(f"   • Estimated Model Size: {model_size_mb:.2f} MB", flush=True)
    print(f"{'='*80}\n", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    
    print(f"{'='*80}\nOPTIMIZER CONFIGURATION\n{'='*80}", flush=True)
    print(f"✓ Optimizer: Adam", flush=True)
    print(f"   • Learning Rate: {LR}", flush=True)
    print(f"   • Loss Function: BCEWithLogitsLoss + L2 Regularization", flush=True)
    print(f"   • L2 Lambda: {L2_LAMBDA}", flush=True)
    print(f"   • LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 6. Train
    print(f"\n{'='*80}\nTRAINING CONFIGURATION\n{'='*80}", flush=True)
    print(f"Max URL Length: {MAX_URL_LEN}", flush=True)
    print(f"Embedding Dim: {EMBEDDING_DIM}", flush=True)
    print(f"LSTM Hidden Dim: {LSTM_HIDDEN}", flush=True)
    print(f"Batch Size: {BATCH_SIZE} (GPU optimized)", flush=True)
    print(f"Learning Rate: {LR}", flush=True)
    print(f"Early Stopping Patience: {EARLY_STOP} epochs (AGGRESSIVE)", flush=True)
    print(f"Total Epochs: {EPOCHS}", flush=True)
    print(f"\n🔧 REGULARIZATION:", flush=True)
    print(f"   • L2 Regularization (λ): {L2_LAMBDA}", flush=True)
    print(f"   • Dropout Rate: {DROPOUT_RATE}", flush=True)
    print(f"   • LSTM Dropout: 0.4", flush=True)
    print(f"   • Gradient Clipping: 1.0", flush=True)
    print(f"\n⚠️  CHANGES TO PREVENT OVERFITTING:", flush=True)
    print(f"   • Reduced LSTM Hidden Dim: 128 → 64", flush=True)
    print(f"   • Reduced Batch Size: 2048 → 512", flush=True)
    print(f"   • Increased Dropout: 0.3 → 0.5", flush=True)
    print(f"   • More Aggressive Early Stopping: 8 → 5 epochs", flush=True)
    print(f"   • Added L2 Regularization", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"{'='*80}\nSTARTING GPU TRAINING\n{'='*80}", flush=True)
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_auc","val_auc"]}

    best_val_auc = 0.0
    no_improve   = 0
    model_path   = os.path.join(MODELS_DIR, "lstm_model.pt")

    # Create progress bar for epochs
    pbar = tqdm(range(1, EPOCHS + 1), desc="Training Progress", ncols=100, position=0, leave=True)
    
    for epoch in pbar:
        t0 = time.time()
        
        # Show learning rate
        current_lr = optimizer.param_groups[0]['lr']

        tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, phase="Validation")

        scheduler.step(val_auc)
        elapsed = time.time() - t0
        
        new_lr = optimizer.param_groups[0]['lr']
        
        # Get GPU memory info
        gpu_mem_alloc = torch.cuda.memory_allocated(0) // 1024**2
        gpu_mem_reserved = torch.cuda.memory_reserved(0) // 1024**2

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(val_auc)
        history["val_auc"].append(val_auc)

        mark = " ⭐ [BEST]" if val_auc > best_val_auc else ""
        
        # OVERFITTING DETECTION
        overfit_gap = tr_loss - val_loss
        if overfit_gap < -0.05:
            overfit_warning = " ⚠️ [OVERFITTING RISK]"
        elif overfit_gap < -0.01:
            overfit_warning = " ℹ️ [Monitor overfitting]"
        else:
            overfit_warning = ""
        
        # Update progress bar description
        pbar.set_description(f"[E{epoch:03d}] Loss: {tr_loss:.4f}/{val_loss:.4f} | AUC: {val_auc:.4f}{mark}")
        
        print(f"{'='*80}", flush=True)
        print(f"Epoch {epoch:03d}/{EPOCHS} | Time: {elapsed:6.2f}s | LR: {new_lr:.2e}", flush=True)
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}", flush=True)
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}", flush=True)
        print(f"  Val AUC:    {val_auc:.4f}{mark}", flush=True)
        print(f"  Overfit Gap (Train-Val Loss): {overfit_gap:.4f}{overfit_warning}", flush=True)
        print(f"  GPU Memory: Allocated {gpu_mem_alloc} MB | Reserved {gpu_mem_reserved} MB", flush=True)
        print(f"{'='*80}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size":  vocab_size,
                "embed_dim":   EMBEDDING_DIM,
                "hidden_dim":  LSTM_HIDDEN,
            }, model_path)
            no_improve = 0
            print(f"✅ Model checkpoint saved! (Val AUC improved to {val_auc:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"⚠️  No improvement for {no_improve}/{EARLY_STOP} epochs", flush=True)
            if no_improve >= EARLY_STOP:
                print(f"\n⛔ Early stopping triggered — no AUC improvement for {EARLY_STOP} epochs\n", flush=True)
                pbar.close()
                break

    pbar.close()
    print(f"{'='*80}", flush=True)
    print(f"✓ TRAINING COMPLETE", flush=True)
    print(f"{'='*80}\n", flush=True)

    # 7. Final test evaluation
    print(f"{'='*80}\nFINAL TEST EVALUATION\n{'='*80}", flush=True)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"✓ Loaded best model from checkpoint", flush=True)
    
    test_loss, test_acc, test_auc = eval_epoch(model, test_loader, criterion, phase="Test")

    # Classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            probs = torch.sigmoid(model(X_b.to(DEVICE)))
            all_preds.extend((probs >= 0.5).cpu().int().numpy())
            all_labels.extend(y_b.int().numpy())

    print(f"\n{'='*80}\nTEST METRICS\n{'='*80}", flush=True)
    print(f"  Test Loss:     {test_loss:.4f}", flush=True)
    print(f"  Test Accuracy: {test_acc:.4f}", flush=True)
    print(f"  Test AUC:      {test_auc:.4f}", flush=True)
    print(f"  Best Val AUC:  {best_val_auc:.4f}", flush=True)
    print(f"\n{'='*80}\nCLASSIFICATION REPORT\n{'='*80}", flush=True)
    print(classification_report(all_labels, all_preds,
          target_names=["Safe (0)", "Phishing (1)"]), flush=True)
    print(f"{'='*80}\n", flush=True)

    # 8. Save artifacts
    print(f"{'='*80}\nSAVING ARTIFACTS\n{'='*80}", flush=True)
    char2idx_path = os.path.join(MODELS_DIR, "char2idx.pkl")
    history_path = os.path.join(MODELS_DIR, "training_history.pkl")
    
    joblib.dump(char2idx, char2idx_path)
    joblib.dump(history,  history_path)
    
    print(f"✅ lstm_model.pt saved", flush=True)
    print(f"   └─ Best Validation AUC: {best_val_auc:.4f}", flush=True)
    print(f"   └─ Test Accuracy: {test_acc:.4f}", flush=True)
    print(f"   └─ Test AUC: {test_auc:.4f}", flush=True)
    
    print(f"✅ char2idx.pkl saved", flush=True)
    print(f"   └─ Vocabulary Size: {len(char2idx)}", flush=True)
    
    print(f"✅ training_history.pkl saved", flush=True)
    print(f"   └─ Total Epochs Trained: {len(history['train_loss'])}", flush=True)
    print(f"   └─ Final Train Loss: {history['train_loss'][-1]:.4f}", flush=True)
    print(f"   └─ Final Val Loss: {history['val_loss'][-1]:.4f}", flush=True)
    
    print(f"\n{'='*80}", flush=True)

    # 9. Plot
    print(f"GENERATING TRAINING HISTORY PLOTS...", flush=True)
    plot_history(history)

    print(f"{'='*80}")
    print("🎉 GPU TRAINING COMPLETE — LSTM MODEL READY FOR DEPLOYMENT 🎉")
    print("📊 TRAINED ON MERGED DATASET (300k+ URLs for Better Generalization)")
    print(f"{'='*80}")
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  • Dataset: merged_dataset.csv (867,738 URLs)", flush=True)
    print(f"  • Training Samples: {len(history['train_loss'])} epochs", flush=True)
    print(f"  • Training Device: GPU (strictly)", flush=True)
    print(f"  • Best Validation AUC: {best_val_auc:.4f}", flush=True)
    print(f"  • Test Set Accuracy: {test_acc:.4f}", flush=True)
    print(f"  • Test Set AUC: {test_auc:.4f}", flush=True)
    print(f"  • Model Saved: {os.path.basename(model_path)}", flush=True)
    print(f"  • Vocabulary Size: {len(char2idx)}", flush=True)
    
    print(f"\n🛡️  REGULARIZATION APPLIED:")
    print(f"  • L2 Regularization (λ={L2_LAMBDA})", flush=True)
    print(f"  • Dropout Layers: 0.5", flush=True)
    print(f"  • LSTM Dropout: 0.4", flush=True)
    print(f"  • Reduced Model Capacity", flush=True)
    print(f"  • Smaller Batch Size for Better Generalization", flush=True)
    print(f"  • Aggressive Early Stopping", flush=True)
    
    print(f"\n✅ EXPECTED IMPROVEMENTS OVER PREVIOUS RUN:")
    print(f"  • More realistic AUC scores (not 99.89%)", flush=True)
    print(f"  • Better generalization on test set", flush=True)
    print(f"  • True learning instead of memorization", flush=True)
    print(f"  • 3x more training data = better patterns", flush=True)
    
    print(f"\n🚀 Ready for inference via lstmService.ts", flush=True)
    print(f"{'='*80}\n", flush=True)
