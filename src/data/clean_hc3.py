# src/data/clean_hc3.py

import pandas as pd
import re
import csv
import os
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

def clean_dataset(input_csv, output_csv, max_length=2000, min_length=20, balance_classes=True, verbose=True):
    """
    Clean HC3 dataset:
    - Remove duplicates
    - Remove empty/very short texts
    - Truncate very long texts
    - Minimal cleaning (HTML, URLs)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Cleaning: {input_csv}")
        print(f"{'='*60}\n")

    # Load with robust CSV reader
    df = pd.read_csv(
        input_csv,
        engine="python",
        dtype={"text": "string", "label": "Int64"},
        on_bad_lines="skip"
    )
    original_size = len(df)
    if verbose:
        print(f"Original size: {original_size}")

    # Ensure required columns exist
    if "text" not in df or "label" not in df:
        raise ValueError(f"Missing required columns in input: {list(df.columns)}")

    # Keep only text and label columns
    df = df[["text", "label"]].copy()

    # Clean text content
    if verbose:
        print("\n1. Cleaning text content...")
    df['text'] = df['text'].apply(lambda x: clean_text(x, max_length))

    # Coerce labels to numeric and handle missing values
    if verbose:
        print("2. Cleaning labels...")
    df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')

    # Remove None/empty texts and invalid labels
    if verbose:
        print("3. Removing empty texts and invalid labels...")
    df = df[df['text'].notna() & df['label'].notna() & (df['text'].str.len() >= min_length)]
    df['label'] = df['label'].astype(int)
    if verbose:
        print(f"   Removed {original_size - len(df)} empty/short/invalid rows")

    # Remove duplicates
    if verbose:
        print("4. Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    if verbose:
        print(f"   Removed {before_dedup - len(df)} duplicate texts")

    # Balance classes after cleaning
    if verbose:
        print("\n5. Checking class balance...")
    label_counts = df['label'].value_counts()
    if verbose:
        print(f"   Class distribution:")
        for label, count in label_counts.items():
            label_name = "AI" if label == 1 else "HUMAN"
            print(f"     {label_name}: {count} ({count/len(df)*100:.1f}%)")

    # Optional: Balance classes if heavily skewed
    if balance_classes and label_counts.max() / label_counts.min() > 2.0:
        if verbose:
            print("\n   ⚠️  Classes imbalanced! Balancing...")
        min_samples = label_counts.min()
        df = df.groupby('label').sample(n=min_samples, random_state=42)
        if verbose:
            print(f"   Balanced to {min_samples} samples per class")

    # Final stats
    if verbose:
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
        df = df.drop('text_length', axis=1)

    # Save with consistent formatting
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n"
    )

    if verbose:
        print(f"\n✅ Saved to: {output_csv}\n")

    return df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HC3 Dataset Cleaning Pipeline")
    print("="*60)

    # Define all files to process
    files_to_process = [
        ("train.csv", "train_clean.csv", True),      # balance train set
        ("test.csv", "test_clean.csv", False),       # don't balance test set
        ("val.csv", "val_model.csv", False),
        ("val_split.csv", "val_split_model.csv", False),
        ("train_split.csv", "train_split_model.csv", False),
    ]

    for input_file, output_file, balance in files_to_process:
        input_path = os.path.join("data/processed", input_file)
        output_path = os.path.join("data/processed", output_file)

        if not os.path.exists(input_path):
            print(f"\n⚠️  Skipped {input_file}: File not found")
            continue

        try:
            clean_dataset(
                input_path,
                output_path,
                max_length=2000,
                min_length=20,
                balance_classes=balance
            )
        except Exception as e:
            print(f"\n⚠️  Error processing {input_file}: {e}")

    print("\n" + "="*60)
    print("✅ All cleaning complete!")
    print("="*60)
    print("\nNext step:")
    print("  python src/data/prepare_prompts.py")