# src/training/train_lora.py

import os
import argparse
import torch
import time
import json
import subprocess
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from huggingface_hub import login
from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM
from torch_xla.core.xla_model import is_master_ordinal

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_secret(secret_name, region_name):
    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        paginator = client.get_paginator('list_secrets')
        for page in paginator.paginate():
            for secret in page['SecretList']:
                if secret['Name'].startswith(secret_name):
                    response = client.get_secret_value(SecretId=secret['ARN'])
                    if 'SecretString' in response:
                        return response['SecretString']
        return None
    except Exception:
        return None

def create_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": sample['prompt']},
            {"role": "assistant", "content": sample['completion']}
        ]
    }

def training_function(script_args, training_args):
    log(f"Training Agent: {script_args.feature}")

    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': f'data/prompts/{script_args.feature}_train.jsonl',
            'validation': f'data/prompts/{script_args.feature}_val.jsonl',
        }
    )
    train_dataset = dataset['train'].map(create_conversation, remove_columns=dataset['train'].column_names)
    eval_dataset = dataset['validation'].map(create_conversation, remove_columns=dataset['validation'].column_names)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(script_args.model_id, training_args.trn_config, torch_dtype=dtype)

    # Configure LoRA
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "gate_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFT Config
    sft_config = NeuronSFTConfig(
        max_seq_length=script_args.max_seq_length,
        packing=True,
        **training_args.to_dict(),
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": True},
    )

    # Trainer
    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    log("Starting training...")
    trainer.train()
    log("Training complete!")

    # Save model and tokenizer
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    with open(f"{training_args.output_dir}/training_config.json", 'w') as f:
        json.dump(vars(script_args), f, indent=2)

@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer_id: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    feature: str = field(default="stylistic")
    max_seq_length: int = field(default=1024)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    secret_name: str = field(default="huggingface/token")
    secret_region: str = field(default="us-west-2")

if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    hf_token = os.environ.get("HF_TOKEN") or get_secret(script_args.secret_name, script_args.secret_region)
    if hf_token:
        login(token=hf_token)

    set_seed(training_args.seed)
    training_function(script_args, training_args)