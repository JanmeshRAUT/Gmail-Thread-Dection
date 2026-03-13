# GPU-Trained LSTM Phishing Detection Model

## Model Details

- **File:** `lstm_model.pt`
- **Size:** 0.24 MB
- **Training Data:** 300,000 balanced URLs
- **Architecture:** Character-level Bidirectional LSTM
- **Performance:** 99.67% Validation AUC, 98.07% Test Accuracy
- **Device:** NVIDIA GPU (CUDA 12.1 required for training, CPU fallback for inference)

## Model Configuration

```
Embedding Dimension: 32
LSTM Hidden Dimension: 64
Vocabulary Size: 243 unique characters
Max URL Length: 100 characters
Batch Size: 512
Dropout Rate: 0.5 (with L2 regularization)
```

## Supporting Files

### Model Assets
- `lstm_model.pt` - PyTorch trained model
- `char2idx.pkl` - Character-to-index vocabulary mapping
- `training_history.pkl` - Training metrics and history

### Configuration
- `ml/api_server.py` - Flask API server for inference
- `ml/predict_lstm_api.py` - Model loading and prediction logic
- `ml/requirements.txt` - Python dependencies

## Installation & Usage

### 1. Download Model
```bash
wget https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/download/v1.0-model/lstm_model.pt
wget https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases/download/v1.0-model/char2idx.pkl
```

### 2. Place Files
```
Gmail-Thread-Dection/
├── ml/
│   ├── models/
│   │   ├── lstm_model.pt       ← Download here
│   │   ├── char2idx.pkl        ← Download here
│   │   └── training_history.pkl
│   ├── api_server.py
│   ├── predict_lstm_api.py
│   └── requirements.txt
```

### 3. Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### 4. Start API Server
```bash
python ml/api_server.py
# Server runs on http://localhost:5000
```

### 5. Test Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.google.com"}'
```

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "ok", "model_loaded": true}
```

### Predict Single URL
```
POST /predict
Body: {"url": "https://example.com"}
Response: {
  "url": "https://example.com",
  "prediction_class": 0,
  "confidence": 0.9952,
  "is_phishing": false,
  "label": "benign",
  "probability": 0.0048,
  "risk_level": "LOW"
}
```

### Analyze Email Content
```
POST /analyze
Body: {"content": "email body...", "attachments": [...]}
Response: {
  "riskLevel": "LOW|MEDIUM|HIGH|CRITICAL",
  "summary": "...",
  "threats": [...],
  "urls_analyzed": 5,
  "phishing_urls": [...]
}
```

## Risk Levels

- **LOW** (< 0.45): Safe domain
- **MEDIUM** (0.45 - 0.65): Suspicious patterns detected
- **HIGH** (0.65 - 0.85): High phishing probability
- **CRITICAL** (≥ 0.85): Likely phishing URL

## Training Details

### Dataset
- Source: Merged from 4 datasets
- Total URLs: 867,000+ (subsampled to 300,000 for training)
- Distribution: Balanced 50/50 (benign/phishing)
- Train/Val/Test: 68%/12%/20%

### Regularization Applied
- Dropout: 0.5
- L2 Weight Penalty: 1e-4
- Gradient Clipping: norm=1.0
- Early Stopping: 5 epochs patience

### Training Results
```
Total Epochs: 36 (stopped early)
Final Training Loss: 0.0614
Final Validation Loss: 0.0663
Validation Accuracy: 98.13%
Validation AUC: 0.9967
Test Accuracy: 98.07%
Test AUC: 0.9972
```

## System Integration

### Option 1: Standalone Flask API
```bash
python ml/api_server.py
# RESTful predictions on port 5000
```

### Option 2: Node.js Frontend Integration
```bash
npm run dev
# Integrated with Express backend at port 3000
# Routes requests to Flask API
```

### Option 3: Direct Python Usage
```python
from ml.predict_lstm_api import PhishingDetector
detector = PhishingDetector()
result = detector.predict("https://suspicious-url.com")
print(result['risk_level'], result['confidence'])
```

## Requirements

### For Inference (CPU)
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Flask (for API server)

### For Training (GPU - NVIDIA required)
- CUDA 12.1
- PyTorch 2.5.1+cu121
- scikit-learn
- tqdm
- pandas

## License

Model trained for phishing detection research. 
Use responsibly and ethically.

## Citation

If you use this model in research, please cite:
```bibtex
@model{lstm_phishing_detector,
  title={GPU-Trained Character-Level LSTM for Phishing URL Detection},
  author={Janmesh Raut},
  year={2026},
  url={https://github.com/JanmeshRAUT/Gmail-Thread-Dection}
}
```

## Support

- GitHub Issues: [Report bugs](https://github.com/JanmeshRAUT/Gmail-Thread-Dection/issues)
- Model Performance: [View metrics](https://github.com/JanmeshRAUT/Gmail-Thread-Dection/releases)

---
Last Updated: March 13, 2026
