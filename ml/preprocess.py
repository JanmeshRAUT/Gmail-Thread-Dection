import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from tqdm import tqdm

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

def preprocess_data(filepath):
    """Load and preprocess dataset"""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Rename columns
    if 'URL' in df.columns:
        df = df.rename(columns={'URL': 'url'})
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['url'])
    
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
    df = pd.concat([df[['url', 'label']], features_df], axis=1)
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    return df, le

def prepare_training_data(df, test_size=0.2):
    """Prepare data for training"""
    X = df.drop(['url', 'label'], axis=1).values
    y = df['label'].values
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load and preprocess data
    df, label_encoder = preprocess_data('ml/data/balanced_urls.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
