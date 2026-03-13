import pandas as pd
import os

datasets = {
    'balanced_urls.csv': 'Currently Used (BALANCED)',
    'PhiUSIIL_Phishing_URL_Dataset.csv': 'Phishing URL Dataset',
    'Global_Cybersecurity_Threats_2015-2024.csv': 'Cybersecurity Threats (2015-2024)',
    'merged_dataset.csv': 'Merged Dataset'
}

print("\n" + "="*80)
print("AVAILABLE DATASETS ANALYSIS")
print("="*80 + "\n")

for filename, description in datasets.items():
    filepath = os.path.join('data', filename)
    if os.path.exists(filepath):
        try:
            # Get file size
            file_size = os.path.getsize(filepath) / (1024**2)
            
            # Load dataset info
            df = pd.read_csv(filepath)
            
            print(f"📊 {filename}")
            print(f"   Description: {description}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Rows: {df.shape[0]:,}")
            print(f"   Columns: {df.shape[1]}")
            print(f"   Column Names: {[c.strip() for c in df.columns]}")
            
            # Check for label column
            if 'label' in df.columns:
                labels = df['label'].value_counts()
                print(f"   Label Distribution: {dict(labels)}")
            
            # Show sample row
            if 'url' in df.columns:
                print(f"   Sample URL: {df['url'].iloc[0][:60]}...")
            
            print()
        except Exception as e:
            print(f"   ❌ Error reading: {e}\n")

print("="*80 + "\n")
