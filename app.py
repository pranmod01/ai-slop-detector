# app.py
import json
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

# =========================
# Mock Model Implementation
# =========================

FEATURES = ["stylistic", "syntactic", "semantic", "repetition"]

class MockTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"[CHAT] {messages[-1]['content']}"

class MockModel:
    """
    Lightweight stand-in for a LoRA adapted causal LM.
    Deterministic per (feature, text) via hashing so results are stable.
    """
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def predict(self, text: str) -> Dict[str, Any]:
        seed = int(hashlib.md5(f"{self.feature_name}::{text}".encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rng = random.Random(seed)

        confidence = round(rng.uniform(0.35, 0.95), 2)
        thresholds = {
            "stylistic": 0.55,
            "syntactic": 0.60,
            "semantic": 0.65,
            "repetition": 0.50,
        }
        th = thresholds.get(self.feature_name, 0.6)

        label = int(confidence >= th)
        rationales = [
            "Surface cues suggest AI-like tone.",
            "Structure resembles common LLM cadence.",
            "Content leans generic; limited specificity.",
            "Strong repetition patterns detected.",
            "Natural variation consistent with human text.",
        ]
        rationale = rationales[seed % len(rationales)]

        return {
            "label": label,
            "confidence": confidence,
            "rationale": rationale,
            "threshold": th,
        }

@dataclass
class LoadedMock:
    tokenizer: MockTokenizer
    model: MockModel

def load_mock_model(feature: str) -> LoadedMock:
    return LoadedMock(tokenizer=MockTokenizer(), model=MockModel(feature))

# =========================
# Inference helpers
# =========================

def build_prompt(text: str, feature: str) -> str:
    return (
        f"Detect {feature} signals of AI-generated writing.\n\n"
        f"Text:\n{text}\n\n"
        f"Respond with a JSON object: "
        f'{{"label": 0 or 1, "confidence": 0..1, "rationale": "short reason"}}'
    )

def generate_json_response(tokenizer: MockTokenizer, model: MockModel, prompt: str) -> Tuple[Dict[str, Any], str]:
    try:
        text = prompt.split("Text:\n", 1)[1].rsplit("\n\n", 1)[0]
    except Exception:
        text = prompt
    result = model.predict(text)
    as_str = json.dumps(
        {"label": result["label"], "confidence": result["confidence"], "rationale": result["rationale"]}
    )
    return result, as_str

def run_single_inference(text: str, feature: str) -> Dict[str, Any]:
    lm = load_mock_model(feature)
    prompt = build_prompt(text, feature)
    result, _ = generate_json_response(lm.tokenizer, lm.model, prompt)
    return result

def run_batch_inference(df: pd.DataFrame, text_col: str, feature: str) -> pd.DataFrame:
    lm = load_mock_model(feature)
    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        txt = str(row[text_col])
        prompt = build_prompt(txt, feature)
        result, _ = generate_json_response(lm.tokenizer, lm.model, prompt)
        out_rows.append(
            {
                "text": txt,
                "label": result["label"],
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "threshold": result["threshold"],
            }
        )
    return pd.DataFrame(out_rows)

# ---------- NEW: ensemble helpers ----------

def run_single_ensemble(text: str, weights: Dict[str, float]) -> Dict[str, Any]:
    per_agent = {}
    for f in FEATURES:
        per_agent[f] = run_single_inference(text, f)

    # weighted average of confidences in [0,1]
    wsum = sum(weights.values()) or 1.0
    ensemble_score = sum(weights[f] * per_agent[f]["confidence"] for f in FEATURES) / wsum

    # simple threshold 0.5 for ensemble (you can tune later)
    label = int(ensemble_score >= 0.5)

    return {
        "per_agent": per_agent,
        "ensemble_score": round(ensemble_score, 4),
        "label": label,
    }

def run_batch_ensemble(df: pd.DataFrame, text_col: str, weights: Dict[str, float]) -> pd.DataFrame:
    # get per-agent predictions
    per_agent_frames = []
    for f in FEATURES:
        preds = run_batch_inference(df, text_col=text_col, feature=f)
        preds = preds.rename(
            columns={
                "label": f"label_{f}",
                "confidence": f"conf_{f}",
                "rationale": f"rationale_{f}",
                "threshold": f"th_{f}",
            }
        )
        per_agent_frames.append(preds)

    merged = pd.concat([df.reset_index(drop=True)] + per_agent_frames, axis=1)

    wsum = sum(weights.values()) or 1.0
    merged["ensemble_score"] = (
        sum(weights[f] * merged[f"conf_{f}"] for f in FEATURES) / wsum
    )
    merged["ensemble_label"] = (merged["ensemble_score"] >= 0.5).astype(int)
    return merged

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="AI Slop Detector (Mock Demo + Ensemble)", layout="wide")

st.title("🤖 AI Slop Detector — Demo (Mock Models + Ensemble)")
st.caption("This keeps your original single-agent flow and adds an **Ensemble (all four)** option.")

with st.sidebar:
    st.header("Settings")

    target = st.selectbox(
        "Analyze",
        options=FEATURES + ["Ensemble (all four)"],
        index=0,
        help="Choose a single agent or the ensemble."
    )
    mode = st.radio("Mode", options=["Single Text", "Batch (CSV)"], horizontal=False)

    if target == "Ensemble (all four)":
        st.markdown("**Ensemble Weights** (will be normalized)")
        w_sty = st.slider("stylistic", 0.0, 3.0, 1.0, 0.1)
        w_syn = st.slider("syntactic", 0.0, 3.0, 1.0, 0.1)
        w_sem = st.slider("semantic", 0.0, 3.0, 1.0, 0.1)
        w_rep = st.slider("repetition", 0.0, 3.0, 1.0, 0.1)
        weights = {"stylistic": w_sty, "syntactic": w_syn, "semantic": w_sem, "repetition": w_rep}
    else:
        weights = {f: 1.0 for f in FEATURES}  # unused for single-agent

    st.markdown("---")
    st.markdown("**Notes**")
    st.write(
        "• Outputs are deterministic per text & feature (hashing)\n"
        "• Swap the mock loader with real HF/PEFT models later"
    )

# ---------- Single Text ----------
if mode == "Single Text":
    st.subheader("Single Text Inference")
    sample = (
        "Airplane windows are aligned with fuselage spars for structural reasons, "
        "so they rarely line up with seats after airline retrofits."
    )
    text = st.text_area("Enter text", value=sample, height=180)

    run_btn = st.button("Run Detection", type="primary")

    if run_btn and text.strip():
        if target == "Ensemble (all four)":
            with st.spinner("Running ensemble (mock) ..."):
                out = run_single_ensemble(text.strip(), weights)

            # per-agent table
            rows = []
            for f in FEATURES:
                r = out["per_agent"][f]
                rows.append(
                    {
                        "agent": f,
                        "confidence": r["confidence"],
                        "label": r["label"],
                        "threshold": r["threshold"],
                        "rationale": r["rationale"],
                    }
                )
            st.markdown("### 🧩 Per-Agent Predictions")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ensemble summary
            st.markdown("### 🧠 Ensemble")
            left, right = st.columns([1, 2])
            with left:
                st.metric(
                    "Final Prediction",
                    "AI" if out["label"] == 1 else "Human",
                    delta=f"{out['ensemble_score']:.2f} score",
                )
            with right:
                st.info(
                    "Ensemble = weighted mean of per-agent confidences. "
                    "Threshold = 0.50 for demo (tune later)."
                )

        else:
            with st.spinner("Running mock inference..."):
                result = run_single_inference(text.strip(), target)
            left, right = st.columns([1, 2])
            with left:
                st.metric("Predicted Label", "AI" if result["label"] == 1 else "Human")
                st.metric("Confidence", f"{result['confidence']:.2f}")
                st.metric("Threshold", f"{result['threshold']:.2f}")
            with right:
                st.markdown("**Rationale**")
                st.info(result["rationale"])

# ---------- Batch CSV ----------
elif mode == "Batch (CSV)":
    st.subheader("Batch Inference from CSV")
    st.write("Upload a CSV with a **`text`** column. (Other columns are preserved.)")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
            st.stop()

        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Detection", type="primary"):
            if target == "Ensemble (all four)":
                with st.spinner("Running ensemble batch (mock) ..."):
                    merged = run_batch_ensemble(df, text_col="text", weights=weights)
                st.success("Done!")

                st.markdown("### 🧠 Ensemble Results")
                st.dataframe(
                    merged[
                        ["text", "ensemble_score", "ensemble_label"]
                        + [f"conf_{f}" for f in FEATURES]
                        + [f"label_{f}" for f in FEATURES]
                    ].head(100),
                    use_container_width=True,
                )

                ai_rate = (merged["ensemble_label"] == 1).mean()
                st.markdown(f"**AI-flag rate (ensemble)**: {ai_rate:.1%}")

                st.download_button(
                    "Download Results CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_ensemble_mock.csv",
                    mime="text/csv",
                )
            else:
                with st.spinner("Running mock batch inference..."):
                    preds = run_batch_inference(df, text_col="text", feature=target)
                    merged = pd.concat([df.reset_index(drop=True), preds], axis=1)
                st.success("Done!")
                st.write("Results:")
                st.dataframe(merged.head(100), use_container_width=True)

                ai_rate = (merged["label"] == 1).mean()
                st.markdown(f"**AI-flag rate**: {ai_rate:.1%}")

                st.download_button(
                    "Download Results CSV",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{target}_mock.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Swap to real LoRA models later by replacing the Mock loader with your HF/PEFT loader.")
