# scripts/check_data_quality.py

import pandas as pd
import numpy as np

def analyze_dataset(csv_path):
    """Quick quality check"""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {csv_path}")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Total examples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Text length stats
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\nText length (characters):")
    print(f"  Min: {df['text_length'].min()}")
    print(f"  Max: {df['text_length'].max()}")
    print(f"  Mean: {df['text_length'].mean():.0f}")
    print(f"  Median: {df['text_length'].median():.0f}")
    
    print(f"\nWord count:")
    print(f"  Min: {df['word_count'].min()}")
    print(f"  Max: {df['word_count'].max()}")
    print(f"  Mean: {df['word_count'].mean():.0f}")
    
    # Check for issues
    print(f"\n{'='*60}")
    print("Potential Issues:")
    print(f"{'='*60}")
    
    # Empty texts
    empty = df['text'].isna().sum() + (df['text'].str.strip() == '').sum()
    print(f"  Empty texts: {empty}")
    
    # Very short texts
    very_short = (df['text_length'] < 10).sum()
    print(f"  Very short (<10 chars): {very_short}")
    
    # Very long texts
    very_long = (df['text_length'] > 5000).sum()
    print(f"  Very long (>5000 chars): {very_long}")
    
    # Duplicates
    duplicates = df.duplicated(subset=['text']).sum()
    print(f"  Duplicate texts: {duplicates}")
    
    # HTML tags (rough check)
    html_tags = df['text'].str.contains('<[^>]+>', regex=True).sum()
    print(f"  Contains HTML tags: {html_tags}")
    
    # URLs
    urls = df['text'].str.contains('http[s]?://', regex=True).sum()
    print(f"  Contains URLs: {urls}")
    
    return df

if __name__ == "__main__":
    train_df = analyze_dataset('data/processed/train.csv')
    test_df = analyze_dataset('data/processed/test.csv')