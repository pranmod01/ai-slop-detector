# src/data/fix_csvs.py
import os, csv, pandas as pd

IN_DIR  = "data/processed"
OUT_DIR = "data/processed"

def load_any(path):
    # very forgiving reader
    return pd.read_csv(
        path,
        engine="python",            # tolerant parser
        dtype={"text": "string", "label": "Int64"},
        on_bad_lines="skip"         # skip malformed rows
    )

def clean(df):
    # keep only text,label; coerce & drop bad
    if "text" not in df or "label" not in df:
        raise ValueError(f"Missing required columns in input: {list(df.columns)}")
    out = df[["text", "label"]].copy()
    out["text"]  = out["text"].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(int)  # to plain int
    return out

def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,  # minimal quoting to avoid over-quoting issues
        lineterminator="\n"
    )

for name in ["train.csv", "val.csv", "val_split.csv", "test.csv", "train_split.csv"]:
    p = os.path.join(IN_DIR, name)
    if not os.path.exists(p): 
        continue
    try:
        df = load_any(p)
        slim = clean(df)
        out = os.path.join(OUT_DIR, name.replace(".csv", "_model.csv"))
        write_csv(slim, out)
        print(f"✓ Wrote {out} (rows={len(slim)})")
    except Exception as e:
        print(f"⚠️ Skipped {p}: {e}")
