import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import re

def extract_features(url):
    """Extract features from URL"""
    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_dashes': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_at': url.count('@'),
        'num_question': url.count('?'),
        'num_equals': url.count('='),
        'num_ampersand': url.count('&'),
        'num_hash': url.count('#'),
        'num_percent': url.count('%'),
        'num_digits': sum(1 for c in url if c.isdigit()),
        'num_special': sum(1 for c in url if not c.isalnum() and c != '/'),
    }
    return features

def load_and_preprocess_data(filepath):
    """Load and preprocess dataset"""
    print(f"Loading data from {filepath}")
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
            print(f"Successfully loaded with {encoding} encoding")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if df is None:
        raise ValueError(f"Could not load CSV file with any encoding: {encodings}")
    
    # Rename columns if needed
    if 'URL' in df.columns:
        df = df.rename(columns={'URL': 'url'})
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['url'])
    df = df.dropna()
    
    print(f"Dataset size: {len(df)}")
    
    # Extract features
    print("Extracting features...")
    feature_list = []
    for url in tqdm(df['url'], desc="Processing URLs"):
        if isinstance(url, str):
            feature_list.append(extract_features(url))
        else:
            feature_list.append({k: 0 for k in extract_features("http://example.com").keys()})
    
    features_df = pd.DataFrame(feature_list)
    
    # Combine with labels
    df = pd.concat([df[['url', 'label']].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    return df, le

def prepare_lstm_data(df, max_length=20):
    """Prepare data for LSTM training"""
    X = df.drop(['url', 'label'], axis=1).values
    y = df['label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for LSTM (samples, timesteps, features)
    # Repeat features to create sequence
    X_lstm = np.repeat(X[:, np.newaxis, :], max_length, axis=1)
    
    return X_lstm, y, scaler

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    """Train the LSTM model"""
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('ml/models/lstm_model.keras', save_best_only=True, monitor='val_accuracy')
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ml/graphs/training_metrics.png', dpi=300)
    print("Training history plot saved to ml/graphs/training_metrics.png")

if __name__ == "__main__":
    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data('ml/data/balanced_urls.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare training data
    X_lstm, y, scaler = prepare_lstm_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32)
    
    # Save model and preprocessing objects
    joblib.dump(label_encoder, 'ml/models/label_encoder.pkl')
    joblib.dump(scaler, 'ml/models/scaler.pkl')
    joblib.dump(df.drop(['url', 'label'], axis=1).columns.tolist(), 'ml/models/feature_names.pkl')
    
    # Plot and save training history
    plot_training_history(history)
    joblib.dump(history.history, 'ml/models/training_history.pkl')
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\nModel training completed successfully!")
    print("Saved files:")
    print("  - ml/models/lstm_model.keras")
    print("  - ml/models/label_encoder.pkl")
    print("  - ml/models/scaler.pkl")
    print("  - ml/models/feature_names.pkl")
    print("  - ml/models/training_history.pkl")
    print("  - ml/graphs/training_metrics.png")
