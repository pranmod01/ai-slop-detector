import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

def preprocess_function(examples, tokenizer, max_length=1536):
    """Tokenize prompts and completions"""
    # Combine prompt + completion for training
    full_texts = [
        p + c for p, c in zip(examples['prompt'], examples['completion'])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding=False  # We'll pad in collator
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def main(args):
    print(f"\n{'='*60}")
    print(f"Training Agent: {args.feature}")
    print(f"{'='*60}\n")
    
    # 1. Load tokenizer and model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. Prepare model for LoRA training
    print("Setting up LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Load dataset
    print(f"Loading {args.feature} dataset...")
    dataset = load_dataset(
        'json',
        data_files={
            'train': f'data/prompts/{args.feature}_train.jsonl',
            'test': f'data/prompts/{args.feature}_test.jsonl'
        }
    )
    
    # 4. Tokenize
    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,  # Use bfloat16 for Trainium
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Disable wandb for hackathon
        dataloader_num_workers=4,
    )
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
    )
    
    # 8. Train
    print("\nStarting training...")
    trainer.train()
    
    # 9. Save adapter
    print(f"\nSaving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training args for reference
    with open(f"{args.output_dir}/training_config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n✓ Agent '{args.feature}' training complete!")
    print(f"  Saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--feature", required=True, 
                       choices=["stylistic", "syntactic", "semantic", "repetition"])
    parser.add_argument("--output_dir", required=True)
    
    # Training args
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1536)
    
    # LoRA args
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()
    main(args)