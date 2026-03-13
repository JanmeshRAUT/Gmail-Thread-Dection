## How to Upload Your Model to GitHub Releases

### Step 1: Create a Release on GitHub

1. Go to your repository: **https://github.com/JanmeshRAUT/Gmail-Thread-Dection**

2. Click **"Releases"** (or go to `/releases`)

3. Click **"Create a new release"** (or "Draft a new release")

### Step 2: Fill in Release Details

**Release Name:** `v1.0-model`

**Title:** `LSTM Model Release v1.0 - GPU-Trained Phishing Detector`

**Description:**
```
🔬 **GPU-Trained Character-Level LSTM for Phishing Detection**

📊 **Model Statistics:**
- Vocabulary: 243 unique characters
- Training data: 300,000 balanced URLs
- Validation AUC: 99.67%
- Test Accuracy: 98.07%
- Test AUC: 99.72%

📦 **Package Contents:**
- `lstm_model.pt` - PyTorch trained model (0.24 MB)
- `char2idx.pkl` - Character vocabulary mapping
- `training_history.pkl` - Training metrics
- `api_server.py` - Flask API for predictions
- `predict_lstm_api.py` - Model loading logic
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation

🚀 **Quick Start:**
```bash
# 1. Unzip the package
unzip lstm_model_release_v1.0.zip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API server
python api_server.py

# 4. Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.google.com"}'
```

✅ **Performance Metrics:**
- Character-level tokenization (max 100 chars)
- Bidirectional LSTM (64 hidden units)
- L2 regularization (penalty: 1e-4)
- Dropout: 0.5 + 0.4 on LSTM
- Early stopping at epoch 36

📝 **Training Framework:**
- PyTorch 2.5.1+cu121
- CUDA 12.1
- NVIDIA RTX 2050 GPU

🔗 **Documentation:** See `README.md` in the package
```

### Step 3: Upload the Release File

1. Scroll to **"Attach binaries"** section

2. Click **"Choose files"** or drag-and-drop

3. Select: `lstm_model_release_v1.0.zip` (0.23 MB)

### Step 4: Publish

1. Click **"Publish release"**

2. GitHub will host the file and generate a download link

### Step 5: Share the Links

**Your model will be available at:**
```
Direct download:
https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/download/v1.0-model/lstm_model_release_v1.0.zip

Release page:
https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/tag/v1.0-model
```

### Users Can Now Download With:
```bash
# Option 1: Direct download
wget https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/download/v1.0-model/lstm_model_release_v1.0.zip

# Option 2: Use curl
curl -L -O https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/download/v1.0-model/lstm_model_release_v1.0.zip

# Option 3: From GitHub web interface
# Go to releases → v1.0-model → Download ZIP
```

### Alternative: Share as Documentation Feature

Add to your GitHub README.md:
```markdown
## 📊 Pre-trained Model

Download the GPU-trained LSTM model:
- **[v1.0 Release](https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/tag/v1.0-model)** (0.23 MB)
- Validation AUC: **99.67%**
- Test Accuracy: **98.07%**

See [MODEL_RELEASE.md](./MODEL_RELEASE.md) for details
```

### Advanced: Using Git Release Tools (Optional)

If you have GitHub CLI installed:
```bash
gh release create v1.0-model \
  --title "LSTM Model Release v1.0" \
  --notes "GPU-trained phishing detector model" \
  ./lstm_model_release_v1.0.zip
```

---

**Your model is now ready to share! 🎉**

Package location: `e:\Gmail\Gmail-Thread-Dection\lstm_model_release_v1.0.zip`
