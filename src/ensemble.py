# This loads ALL 4 trained agents and combines their predictions

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

class SlopDetectorEnsemble:
    """
    Loads all 4 trained agents and combines predictions
    """
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print("🔄 Loading multi-agent ensemble...")
        
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model ONCE
        print("  Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load each agent adapter
        self.features = ["stylistic", "syntactic", "semantic", "repetition"]
        self.agents = {}
        
        for feature in self.features:
            print(f"  Loading {feature} agent...")
            try:
                self.agents[feature] = PeftModel.from_pretrained(
                    self.base_model,
                    f"models/agent_{feature}",
                    adapter_name=feature  # Name the adapter
                )
            except Exception as e:
                print(f"    ⚠️  Failed to load {feature}: {e}")
        
        print(f"✅ Loaded {len(self.agents)}/4 agents\n")
        
        # Ensemble weights
        self.weights = {
            'stylistic': 0.25,
            'syntactic': 0.20,
            'semantic': 0.30,
            'repetition': 0.25
        }
        
        # Get AI/HUMAN token IDs
        self.ai_token_id = self.tokenizer.encode("AI", add_special_tokens=False)[0]
        self.human_token_id = self.tokenizer.encode("HUMAN", add_special_tokens=False)[0]
    
    def _make_prompt(self, feature, text):
        """Create feature-specific prompt"""
        # Same templates as in prepare_prompts.py
        templates = {
            "stylistic": f"""[INST] You are a stylistic auditor. Focus on repetition patterns, sentence rhythm, and writing burstiness.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
            
            "syntactic": f"""[INST] You are a syntax auditor. Focus on punctuation patterns, clause structure, and grammatical regularity.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
            
            "semantic": f"""[INST] You are a semantics auditor. Focus on specificity, concreteness, and information density.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """,
            
            "repetition": f"""[INST] You are a repetition detector. Focus on formulaic structure and repeated phrases.
Is this text AI-generated or human-written? Answer with ONE WORD: "AI" or "HUMAN".

Text:
{text}

Answer:[/INST] """
        }
        return templates[feature]
    
    def _get_score(self, agent, prompt):
        """Get AI probability from one agent"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1536
        )
        inputs = {k: v.to(agent.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = agent(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get logits for AI/HUMAN tokens
            ai_logit = logits[self.ai_token_id].item()
            human_logit = logits[self.human_token_id].item()
            
            # Convert to probability
            probs = torch.softmax(torch.tensor([human_logit, ai_logit]), dim=0)
            ai_prob = probs[1].item()
        
        return ai_prob * 100  # 0-100 scale
    
    def detect(self, text):
        """
        Run all agents and return ensemble prediction
        
        Returns:
            {
                'final_score': float,
                'verdict': str,
                'confidence': str,
                'agent_scores': dict,
                'disagreement': float
            }
        """
        scores = {}
        
        # Run each agent
        for feature in self.features:
            if feature in self.agents:
                prompt = self._make_prompt(feature, text)
                score = self._get_score(self.agents[feature], prompt)
                scores[feature] = round(score, 2)
            else:
                scores[feature] = 50.0  # Neutral if agent missing
        
        # Weighted ensemble
        final_score = sum(scores[f] * self.weights[f] for f in scores)
        
        # Calculate disagreement (standard deviation)
        disagreement = np.std(list(scores.values()))
        
        return {
            'final_score': round(final_score, 2),
            'verdict': 'AI SLOP' if final_score > 70 else 'LIKELY HUMAN',
            'confidence': 'HIGH' if abs(final_score - 50) > 30 else 'MEDIUM' if abs(final_score - 50) > 15 else 'LOW',
            'agent_scores': scores,
            'disagreement': round(disagreement, 2),
            'explanation': self._generate_explanation(scores, final_score)
        }
    
    def _generate_explanation(self, scores, final_score):
        """Generate human-readable explanation"""
        high_scorers = [f for f, s in scores.items() if s > 70]
        low_scorers = [f for f, s in scores.items() if s < 30]
        
        if final_score > 70:
            return f"Likely AI-generated. High confidence from: {', '.join(high_scorers)}"
        elif final_score < 30:
            return f"Likely human-written. Low scores from: {', '.join(high_scorers)}"
        else:
            return "Uncertain. Mixed signals from agents."


if __name__ == "__main__":
    # Quick test
    print("Testing ensemble...\n")
    
    detector = SlopDetectorEnsemble()
    
    test_texts = [
        "This product is amazing! Perfect for anyone looking for quality.",
        "ngl the battery is mid af. died after like 6hrs lol. camera decent tho"
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        print(f"{'='*60}")
        result = detector.detect(text)
        print(f"Verdict: {result['verdict']}")
        print(f"Score: {result['final_score']}/100")
        print(f"Confidence: {result['confidence']}")
        print(f"\nAgent Breakdown:")
        for agent, score in result['agent_scores'].items():
            print(f"  {agent:12s}: {score:5.1f}%")
        print(f"\nDisagreement: {result['disagreement']:.2f}")