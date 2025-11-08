# src/data/load_hc3.py

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_hc3():
    """
    Load HC3 dataset from HuggingFace
    Returns train/test splits
    """
    # Downloads automatically from HuggingFace Hub
    dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    
    # HC3 structure: 
    # - 'question': the prompt
    # - 'human_answers': list of human responses
    # - 'chatgpt_answers': list of ChatGPT responses
    
    return dataset

def process_hc3_to_binary(dataset):
    """
    Convert HC3 to binary classification format
    """
    data = []

    for idx, item in dataset.iterrows():
        # Human examples (label = 0)
        for human_answer in item['human_answers']:
            data.append({
                'text': human_answer,
                'label': 0,  # Human
                'writer': 'human', 
                'source': item['source']
            })
        
        # AI examples (label = 1)  
        for ai_answer in item['chatgpt_answers']:
            data.append({
                'text': ai_answer,
                'label': 1,  # AI
                'writer': 'chatgpt', 
                'source': item['source']

            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test loading
    dataset = load_hc3()
    df = dataset['train'].to_pandas()

    # Save the full raw dataset
    df.to_csv('/home/ubuntu/environment/ai-slop-detector/data/hc3.csv', index=False)

    # Cut into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Process and save
    train_df = process_hc3_to_binary(train)
    test_df = process_hc3_to_binary(test)

    train_df.to_csv('/home/ubuntu/environment/ai-slop-detector/data/processed/train.csv', index=False)
    test_df.to_csv('/home/ubuntu/environment/ai-slop-detector/data/processed/test.csv', index=False)
    
    print(f"Processed {len(train_df)} training examples")
    print(f"Processed {len(test_df)} test examples")