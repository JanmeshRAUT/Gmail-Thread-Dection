import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def standardize_labels(df):
    """Standardize mixed labels (benign/malicious/0/1) to binary"""
    if 'label' not in df.columns:
        raise ValueError("'label' column not found")
    
    # Create a copy
    df = df.copy()
    
    # Convert to string and lowercase for comparison
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    # Map labels
    label_mapping = {
        'benign': 0,
        'legitimate': 0,
        'legitimate ': 0,
        '0': 0,
        'malicious': 1,
        'phishing': 1,
        'phishing ': 1,
        'threat': 1,
        '1': 1,
    }
    
    # Apply mapping
    df['label'] = df['label'].map(lambda x: label_mapping.get(x, None))
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    return df

def extract_features(url):
    """Extract 13 features from URL"""
    try:
        url_str = str(url)
        features = {
            'url_length': len(url_str),
            'num_dots': url_str.count('.'),
            'num_dashes': url_str.count('-'),
            'num_underscores': url_str.count('_'),
            'num_slashes': url_str.count('/'),
            'num_at': url_str.count('@'),
            'num_question': url_str.count('?'),
            'num_equals': url_str.count('='),
            'num_ampersand': url_str.count('&'),
            'num_hash': url_str.count('#'),
            'num_percent': url_str.count('%'),
            'num_digits': sum(1 for c in url_str if c.isdigit()),
            'num_special': sum(1 for c in url_str if not c.isalnum() and c != '/'),
        }
        return features
    except:
        return {k: 0 for k in ['url_length', 'num_dots', 'num_dashes', 'num_underscores', 
                                'num_slashes', 'num_at', 'num_question', 'num_equals', 
                                'num_ampersand', 'num_hash', 'num_percent', 'num_digits', 'num_special']}

def load_merged_data():
    """Load merged dataset"""
    print("\n" + "="*80)
    print("LOADING MERGED DATASET")
    print("="*80)
    
    filepath = 'ml/data/merged_dataset.csv'
    
    # Load with error handling
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Successfully loaded with {encoding} encoding")
            break
        except Exception as e:
            continue
    
    if df is None:
        raise ValueError(f"Could not load {filepath}")
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    return df

def preprocess_data(df):
    """Preprocess and extract features"""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    
    # Standardize labels
    print("Standardizing labels...")
    df = standardize_labels(df)
    
    print(f"✓ Labels standardized")
    print(f"  📈 Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"     Class {label}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Drop duplicates if any
    df = df.drop_duplicates(subset=['url'])
    
    # Extract features
    print(f"\n🔧 Extracting features from {len(df):,} URLs...")
    feature_list = []
    for url in tqdm(df['url'], desc="Extracting features"):
        feature_list.append(extract_features(url))
    
    features_df = pd.DataFrame(feature_list)
    
    # Combine
    df = pd.concat([df[['url', 'label']].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    print(f"✓ Features extracted: {features_df.shape[1]} features")
    return df

def prepare_lstm_data(df, max_length=20, sample_size=None):
    """Prepare data for LSTM training"""
    print("\n" + "="*80)
    print("PREPARING LSTM DATA")
    print("="*80)
    
    # Sample if dataset too large
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} rows from {len(df):,}...")
        df = df.sample(n=sample_size, random_state=42, stratify=df['label'])
    
    X = df.drop(['url', 'label'], axis=1).values
    y = df['label'].values
    
    print(f"Feature shape before normalization: {X.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_lstm = np.repeat(X[:, np.newaxis, :], max_length, axis=1)
    
    print(f"LSTM input shape: {X_lstm.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Class distribution in training: {np.bincount(y_train)}")
    print(f"✓ Class distribution in test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, scaler

def build_robust_lstm_model(input_shape):
    """Build improved LSTM model with better architecture"""
    print("\n" + "="*80)
    print("BUILDING LSTM MODEL")
    print("="*80)
    
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print("✓ Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=64):
    """Train the LSTM model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('ml/models/lstm_model_robust.keras', save_best_only=True, monitor='val_accuracy', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    print(f"Training with {epochs} epochs, batch size {batch_size}...\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy:.4f}")
    print(f"✓ Test AUC: {test_auc:.4f}")
    
    return test_loss, test_accuracy, test_auc

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Training Complete!', ha='center', va='center', 
                   fontsize=20, fontweight='bold', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('ml/graphs/training_history_robust.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved to ml/graphs/training_history_robust.png")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROBUST PHISHING URL DETECTION - LSTM MODEL TRAINING")
    print("="*80)
    
    # Load and preprocess
    df = load_merged_data()
    df = preprocess_data(df)
    
    # Prepare LSTM data (using all data or sample if needed)
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, max_length=20, sample_size=None)
    
    # Build model
    model = build_robust_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Train
    history = train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=64)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    print("\n" + "="*80)
    print("SAVING ARTIFACTS")
    print("="*80)
    
    joblib.dump(scaler, 'ml/models/scaler_robust.pkl')
    joblib.dump(df.drop(['url', 'label'], axis=1).columns.tolist(), 'ml/models/feature_names_robust.pkl')
    joblib.dump(history.history, 'ml/models/training_history_robust.pkl')
    
    print("✓ Saved: lstm_model_robust.keras")
    print("✓ Saved: scaler_robust.pkl")
    print("✓ Saved: feature_names_robust.pkl")
    print("✓ Saved: training_history_robust.pkl")
    
    # Plot
    plot_training_history(history)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE - ROBUST MODEL READY")
    print("="*80 + "\n")
