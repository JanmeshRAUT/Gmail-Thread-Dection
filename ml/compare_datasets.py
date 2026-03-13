import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filepath, name):
    """Load and analyze a single dataset"""
    print(f"\n{'='*90}")
    print(f"ANALYZING: {name}")
    print(f"{'='*90}")
    
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python', nrows=1000 if 'PhiUSIIL' in filepath else None)
                print(f"✓ Loaded with {encoding} encoding")
                break
            except:
                continue
        
        if df is None:
            print(f"✗ Failed to load {filepath}")
            return None
        
        return df
    
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def analyze_structure(df, name):
    """Analyze dataset structure"""
    print(f"\n📊 DATASET STRUCTURE:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"\n  Columns: {df.columns.tolist()}")
    print(f"\n  Data Types:\n{df.dtypes}")
    
    # Memory
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\n  💾 Memory: {memory_mb:.2f} MB")
    
    return df

def analyze_label_quality(df):
    """Analyze label quality and distribution"""
    print(f"\n📈 LABEL ANALYSIS:")
    
    possible_label_cols = [col for col in df.columns if col.lower() in ['label', 'class', 'category', 'target']]
    
    if not possible_label_cols:
        print("  ⚠️  No clear label column found")
        return None
    
    label_col = possible_label_cols[0]
    print(f"  Label column: '{label_col}'")
    
    # Value distribution
    print(f"\n  Value Distribution:")
    value_counts = df[label_col].value_counts()
    for val, count in value_counts.items():
        print(f"    {val}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Class balance
    if len(value_counts) == 2:
        ratio = value_counts.iloc[0] / value_counts.iloc[1]
        balance = min(ratio, 1/ratio)
        print(f"\n  ⚖️  Class Balance: {balance:.2%}")
        if balance > 0.4:
            print(f"     ✓ GOOD - Well balanced")
        else:
            print(f"     ⚠️  IMBALANCED")
    
    return label_col

def analyze_feature_quality(df):
    """Analyze feature quality"""
    print(f"\n🔧 FEATURE ANALYSIS:")
    
    # Numeric columns (potential features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  Numeric features: {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        print(f"\n  Top features: {list(numeric_cols[:10])}")
        
        # Missing values
        missing = df[numeric_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"\n  Missing Values in Features:")
            print(missing[missing > 0])
        else:
            print(f"\n  ✓ No missing values in features")
        
        # Statistical summary
        print(f"\n  Statistical Summary:")
        print(df[numeric_cols].describe().to_string())
    
    return numeric_cols

def analyze_url_column(df):
    """Analyze URL column if exists"""
    print(f"\n🔗 URL ANALYSIS:")
    
    url_col = None
    for col in df.columns:
        if col.lower() in ['url', 'uri', 'website', 'domain']:
            url_col = col
            break
    
    if url_col:
        print(f"  URL column: '{url_col}'")
        print(f"  Sample URLs:")
        for i, url in enumerate(df[url_col].head(3)):
            print(f"    {i+1}. {url}")
        
        url_lengths = df[url_col].astype(str).str.len()
        print(f"\n  URL Length Statistics:")
        print(f"    Min: {url_lengths.min()}")
        print(f"    Max: {url_lengths.max()}")
        print(f"    Avg: {url_lengths.mean():.2f}")
    else:
        print("  ✗ No URL column found")
    
    return url_col

def score_dataset(df, name, label_col, numeric_cols):
    """Score dataset suitability"""
    print(f"\n⭐ DATASET QUALITY SCORE:")
    
    scores = {}
    
    # Size score
    size_score = min(100, (len(df) / 100000) * 100)
    scores['Size'] = size_score
    print(f"  Size ({len(df):,} rows): {size_score:.1f}/100")
    
    # Feature richness
    feature_score = min(100, (len(numeric_cols) / 20) * 100)
    scores['Feature Richness'] = feature_score
    print(f"  Features ({len(numeric_cols)} cols): {feature_score:.1f}/100")
    
    # Class balance
    if label_col:
        value_counts = df[label_col].value_counts()
        if len(value_counts) == 2:
            ratio = value_counts.iloc[0] / value_counts.iloc[1]
            balance = min(ratio, 1/ratio)
            balance_score = balance * 100
        else:
            balance_score = 50
    else:
        balance_score = 0
    
    scores['Class Balance'] = balance_score
    print(f"  Class Balance: {balance_score:.1f}/100")
    
    # Data quality (completeness)
    if numeric_cols is not None and len(numeric_cols) > 0:
        completeness = 100 - (df[numeric_cols].isnull().sum().sum() / (len(df) * len(numeric_cols)) * 100)
        scores['Completeness'] = completeness
    else:
        completeness = 50
        scores['Completeness'] = completeness
    
    print(f"  Data Completeness: {completeness:.1f}/100")
    
    # Overall score
    overall = np.mean(list(scores.values()))
    scores['OVERALL'] = overall
    print(f"\n  🎯 OVERALL SCORE: {overall:.1f}/100")
    
    return scores

if __name__ == "__main__":
    print("\n" + "="*90)
    print("COMPREHENSIVE DATASET COMPARISON FOR PHISHING DETECTION MODEL")
    print("="*90)
    
    # Load datasets
    datasets = {
        'balanced_urls.csv': 'ml/data/balanced_urls.csv',
        'Global_Cybersecurity_Threats_2015-2024.csv': 'ml/data/Global_Cybersecurity_Threats_2015-2024.csv',
        'PhiUSIIL_Phishing_URL_Dataset.csv': 'ml/data/PhiUSIIL_Phishing_URL_Dataset.csv'
    }
    
    results = {}
    
    for name, filepath in datasets.items():
        df = load_dataset(filepath, name)
        
        if df is not None:
            df = analyze_structure(df, name)
            label_col = analyze_label_quality(df)
            numeric_cols = analyze_feature_quality(df)
            url_col = analyze_url_column(df)
            scores = score_dataset(df, name, label_col, numeric_cols)
            
            results[name] = {
                'df': df,
                'label_col': label_col,
                'numeric_cols': numeric_cols,
                'url_col': url_col,
                'scores': scores
            }
    
    # Comparison summary
    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY")
    print(f"{'='*90}\n")
    
    print(f"{'Dataset':<45} {'Score':<10} {'Features':<10} {'Rows':<15}")
    print("-" * 90)
    
    for name, data in results.items():
        score = data['scores']['OVERALL']
        features = len(data['numeric_cols']) if data['numeric_cols'] is not None else 0
        rows = len(data['df'])
        print(f"{name:<45} {score:>8.1f}/100  {features:>8}  {rows:>13,}")
    
    # Recommendation
    print(f"\n{'='*90}")
    print("RECOMMENDATION")
    print(f"{'='*90}\n")
    
    if not results:
        print("✗ Could not load any datasets")
    else:
        best_dataset = max(results.items(), key=lambda x: x[1]['scores']['OVERALL'])
        print(f"✓ RECOMMENDED: {best_dataset[0]}")
        print(f"   Score: {best_dataset[1]['scores']['OVERALL']:.1f}/100")
        
        if best_dataset[1]['scores']['Feature Richness'] > 80:
            print(f"\n   Reason: High-quality features ({len(best_dataset[1]['numeric_cols'])} features)")
            print(f"           Perfect for deep learning models like LSTM")
        elif best_dataset[1]['scores']['Size'] > 80:
            print(f"\n   Reason: Large dataset ({len(best_dataset[1]['df']):,} samples)")
            print(f"           Better generalization capability")
        
        print(f"\n   Next Steps:")
        print(f"   1. Use this primary dataset")
        print(f"   2. Clean and preprocess")
        print(f"   3. Train robust LSTM model")
        print(f"   4. (Optional) Augment with other datasets if needed")
    
    print(f"\n{'='*90}\n")
