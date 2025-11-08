import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def make_prompt(feature, text):
    """Create feature-specific prompts"""
    templates = {
        "stylistic": """[INST] You are a stylistic auditor. Focus on repetition patterns, sentence rhythm, and writing burstiness.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
        
        "syntactic": """[INST] You are a syntax auditor. Focus on punctuation patterns, clause structure, and grammatical regularity.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
        
        "semantic": """[INST] You are a semantics auditor. Focus on specificity, concreteness, and information density.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
        
        "repetition": """[INST] You are a repetition detector. Focus on formulaic structure and repeated phrases.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """
    }
    return templates[feature].format(text=text)

def create_splits(df, val_size=0.15):
    """
    Split training data into train/val
    Keep test separate
    """
    # Split train into train + val
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,  # 15% for validation
        stratify=df['label'],  # Keep class balance
        random_state=42
    )

    print(f"Original: {len(df)} examples")
    print(f"Train: {len(train_df)} examples")
    print(f"Val: {len(val_df)} examples")

    return train_df, val_df

def prepare_feature_dataset(df, output_path, feature):
    """Convert dataframe to feature-specific JSONL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            prompt = make_prompt(feature, row['text'])
            target = "AI" if row['label'] == 1 else "HUMAN"
            
            example = {
                'prompt': prompt,
                'completion': target,
                'label': row['label']
            }
            f.write(json.dumps(example) + '\n')
    
    print(f"✓ Created {output_path} with {len(df)} examples")
    return output_path

if __name__ == "__main__":
    # Load original splits
    train_df_full = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Create train/val split from training data
    train_df, val_df = create_splits(train_df_full, val_size=0.15)
    
    # Save splits
    train_df.to_csv('data/processed/train_split.csv', index=False)
    val_df.to_csv('data/processed/val_split.csv', index=False)
    
    print("\n" + "="*60)
    print("Creating feature-specific datasets...")
    print("="*60 + "\n")
    
    features = ["stylistic", "syntactic", "semantic", "repetition"]
    
    for feature in features:
        print(f"\nFeature: {feature}")
        print("-" * 40)
        
        # Train
        prepare_feature_dataset(
            train_df,
            f'data/prompts/{feature}_train.jsonl',
            feature
        )
        
        # Validation
        prepare_feature_dataset(
            val_df,
            f'data/prompts/{feature}_val.jsonl',
            feature
        )
        
        # Test
        prepare_feature_dataset(
            test_df,
            f'data/prompts/{feature}_test.jsonl',
            feature
        )
    
    print("\n" + "="*60)
    print("✅ All datasets created!")
    print("="*60)
    
    # Summary
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} examples")
    print(f"  Val:   {len(val_df)} examples")
    print(f"  Test:  {len(test_df)} examples")