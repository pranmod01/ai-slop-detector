import random
import pandas as pd

class MockAgent:
    def __init__(self, feature):
        self.feature = feature

    def predict(self, texts):
        # Fake prediction probabilities between 0 and 1
        return [random.random() for _ in texts]


def load_mock_agents():
    features = ["stylistic", "syntactic", "semantic", "repetition"]
    return {f: MockAgent(f) for f in features}


def run_inference(texts):
    agents = load_mock_agents()
    preds = {}

    # Each agent outputs a probability (simulated)
    for f, model in agents.items():
        preds[f] = model.predict(texts)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(preds)
    df["ensemble_score"] = df.mean(axis=1)
    df["label"] = (df["ensemble_score"] > 0.5).astype(int)
    return df
