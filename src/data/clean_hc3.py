# src/data/clean_hc3.py

import pandas as pd
import re
from pathlib import Path

def clean_text(text, max_length=2000):
    """Clean text while preserving AI detection signals"""
    
    if pd.isna(text) or text.strip() == "":
        return None  # Mark for removal
    
    text = str(text)
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Replace URLs with token (preserves that URLs exist, useful signal!)
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    
    # 3. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 4. Truncate very long texts (prevent OOM during training)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

def clean_dataset(input_csv, output_csv, max_length=2000, min_length=20):
    """
    Clean HC3 dataset:
    - Remove duplicates
    - Remove empty/very short texts  
    - Truncate very long texts
    - Minimal cleaning (HTML, URLs)
    """
    
    print(f"\n{'='*60}")
    print(f"Cleaning: {input_csv}")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(input_csv)
    original_size = len(df)
    print(f"Original size: {original_size}")
    
    # 1. Clean text
    print("\n1. Cleaning text content...")
    df['text'] = df['text'].apply(lambda x: clean_text(x, max_length))
    
    # 2. Remove None/empty texts
    print("2. Removing empty texts...")
    df = df[df['text'].notna()]
    df = df[df['text'].str.len() >= min_length]
    print(f"   Removed {original_size - len(df)} empty/short texts")
    
    # 3. Remove duplicates (IMPORTANT!)
    print("3. Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    duplicates_removed = before_dedup - len(df)
    print(f"   Removed {duplicates_removed} duplicate texts")
    
    # 4. Balance classes after cleaning
    print("\n4. Checking class balance...")
    label_counts = df['label'].value_counts()
    print(f"   Class distribution:")
    for label, count in label_counts.items():
        label_name = "AI" if label == 1 else "HUMAN"
        print(f"     {label_name}: {count} ({count/len(df)*100:.1f}%)")
    
    # Optional: Balance classes if heavily skewed
    if label_counts.max() / label_counts.min() > 2.0:
        print("\n   ⚠️  Classes imbalanced! Balancing...")
        min_samples = label_counts.min()
        df = df.groupby('label').sample(n=min_samples, random_state=42)
        print(f"   Balanced to {min_samples} samples per class")
    
    # 5. Final stats
    print(f"\n{'='*60}")
    print("Final Statistics:")
    print(f"{'='*60}")
    print(f"  Final size: {len(df)}")
    print(f"  Removed total: {original_size - len(df)} ({(original_size - len(df))/original_size*100:.1f}%)")
    
    df['text_length'] = df['text'].str.len()
    print(f"\n  Text length after cleaning:")
    print(f"    Min: {df['text_length'].min()}")
    print(f"    Max: {df['text_length'].max()}")
    print(f"    Mean: {df['text_length'].mean():.0f}")
    print(f"    Median: {df['text_length'].median():.0f}")
    
    # 6. Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop helper column before saving
    df = df.drop('text_length', axis=1)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved to: {output_csv}\n")
    
    return df

if __name__ == "__main__":
    # Clean both train and test
    print("\n" + "="*60)
    print("HC3 Dataset Cleaning Pipeline")
    print("="*60)
    
    train_df = clean_dataset(
        'data/processed/train.csv',
        'data/processed/train_clean.csv',
        max_length=2000,  # Truncate to 2000 chars (prevents OOM)
        min_length=20      # Remove texts < 20 chars
    )
    
    test_df = clean_dataset(
        'data/processed/test.csv',
        'data/processed/test_clean.csv',
        max_length=2000,
        min_length=20
    )
    
    print("\n" + "="*60)
    print("✅ All cleaning complete!")
    print("="*60)
    print("\nNext step:")
    print("  python src/data/prepare_prompts.py")