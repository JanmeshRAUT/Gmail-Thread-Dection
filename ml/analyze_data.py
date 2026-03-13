import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def analyze_dataset(filepath, name):
    """Analyze individual dataset"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {name}")
    print(f"{'='*80}")
    
    try:
        # Load with error handling
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
                print(f"✓ Loaded with {encoding} encoding")
                break
            except:
                continue
        
        if df is None:
            print(f"✗ Failed to load {filepath}")
            return None
        
        # Basic info
        print(f"\n📊 SHAPE: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"\n📋 COLUMNS: {df.columns.tolist()}")
        
        # Data types
        print(f"\n📝 DATA TYPES:")
        print(df.dtypes)
        
        # Missing values
        print(f"\n⚠️  MISSING VALUES:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values")
        else:
            print(missing[missing > 0])
        
        # Duplicates
        print(f"\n🔄 DUPLICATES: {df.duplicated().sum():,} rows")
        
        # Label distribution
        if 'label' in df.columns or 'Label' in df.columns:
            label_col = 'label' if 'label' in df.columns else 'Label'
            print(f"\n📈 LABEL DISTRIBUTION:")
            print(df[label_col].value_counts())
            print(f"\n   Proportions:")
            print(df[label_col].value_counts(normalize=True).round(4) * 100)
        
        # URL info if available
        if 'url' in df.columns or 'URL' in df.columns:
            url_col = 'url' if 'url' in df.columns else 'URL'
            print(f"\n🔗 URL STATISTICS:")
            print(f"   Min length: {df[url_col].astype(str).str.len().min()}")
            print(f"   Max length: {df[url_col].astype(str).str.len().max()}")
            print(f"   Avg length: {df[url_col].astype(str).str.len().mean():.2f}")
        
        # First few rows
        print(f"\n📦 SAMPLE DATA:")
        print(df.head(3))
        
        return df
    
    except Exception as e:
        print(f"✗ Error analyzing {name}: {str(e)}")
        return None

def merge_datasets(df1, df2, df3, name1, name2, name3):
    """Merge all three datasets"""
    print(f"\n{'='*80}")
    print(f"MERGING ALL DATASETS")
    print(f"{'='*80}")
    
    dfs = []
    
    # Prepare each dataset
    for df, name in [(df1, name1), (df2, name2), (df3, name3)]:
        if df is None:
            continue
        
        # Create a copy
        df_copy = df.copy()
        
        # Standardize columns to 'url' and 'label'
        if 'URL' in df_copy.columns:
            df_copy = df_copy.rename(columns={'URL': 'url'})
        if 'Label' in df_copy.columns:
            df_copy = df_copy.rename(columns={'Label': 'label'})
        
        # Keep only url and label columns if they exist
        available_cols = [col for col in ['url', 'label'] if col in df_copy.columns]
        if available_cols:
            df_copy = df_copy[available_cols]
            
            # Remove rows with missing url
            df_copy = df_copy.dropna(subset=['url'] if 'url' in available_cols else [])
            
            # Remove duplicates within dataset
            before = len(df_copy)
            df_copy = df_copy.drop_duplicates(subset=['url'] if 'url' in available_cols else [])
            after = len(df_copy)
            
            print(f"\n{name}:")
            print(f"  Before dedup: {before:,}")
            print(f"  After dedup: {after:,}")
            print(f"  Removed: {before-after:,}")
            
            dfs.append(df_copy)
    
    # Merge all
    print(f"\nMerging {len(dfs)} datasets...")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Combined size: {len(merged):,}")
    
    # Remove global duplicates
    print(f"Removing global duplicates...")
    before = len(merged)
    merged = merged.drop_duplicates(subset=['url'] if 'url' in merged.columns else [])
    after = len(merged)
    print(f"Before: {before:,} | After: {after:,} | Removed: {before-after:,}")
    
    return merged

def analyze_merged(merged_df):
    """Analyze merged dataset"""
    print(f"\n{'='*80}")
    print(f"MERGED DATASET ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📊 SHAPE: {merged_df.shape[0]:,} rows × {merged_df.shape[1]:,} columns")
    print(f"\n📋 COLUMNS: {merged_df.columns.tolist()}")
    
    # Label distribution
    if 'label' in merged_df.columns:
        print(f"\n📈 LABEL DISTRIBUTION:")
        print(merged_df['label'].value_counts())
        print(f"\n   Proportions:")
        print((merged_df['label'].value_counts(normalize=True) * 100).round(2))
    
    # Memory usage
    print(f"\n💾 MEMORY USAGE: {merged_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Save merged dataset
    output_path = 'ml/data/merged_dataset.csv'
    print(f"\n💾 Saving merged dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Saved successfully")
    
    return merged_df

def create_visualization(merged_df):
    """Create visualizations"""
    print(f"\nCreating visualizations...")
    
    if 'label' in merged_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Label distribution
        label_counts = merged_df['label'].value_counts()
        axes[0].bar(label_counts.index, label_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].set_xlabel('Label')
        
        # Proportions
        props = merged_df['label'].value_counts(normalize=True) * 100
        axes[1].pie(props.values, labels=[f'Class {i}' for i in props.index], autopct='%1.1f%%', 
                   colors=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Label Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ml/graphs/data_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to ml/graphs/data_analysis.png")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA ANALYSIS - PHISHING URL DETECTION")
    print("="*80)
    
    # Analyze individual datasets
    df1 = analyze_dataset('ml/data/balanced_urls.csv', 'balanced_urls.csv')
    df2 = analyze_dataset('ml/data/Global_Cybersecurity_Threats_2015-2024.csv', 'Global_Cybersecurity_Threats_2015-2024.csv')
    df3 = analyze_dataset('ml/data/PhiUSIIL_Phishing_URL_Dataset.csv', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    
    # Merge datasets
    merged_df = merge_datasets(df1, df2, df3, 'balanced_urls.csv', 'Global_Cybersecurity_Threats_2015-2024.csv', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    
    # Analyze merged
    merged_df = analyze_merged(merged_df)
    
    # Visualize
    create_visualization(merged_df)
    
    print(f"\n{'='*80}")
    print("✓ DATA ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
