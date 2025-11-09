# Future version
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_real_agents(base_model, adapters_path):
    agents = {}
    for feature in ["stylistic", "syntactic", "semantic", "repetition"]:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, f"{adapters_path}/{feature}")
        model.eval()
        agents[feature] = model
    return agents
