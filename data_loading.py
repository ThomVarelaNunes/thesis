"""
data_loading.py
---------------
Loads and merges all model predictions with the human interoceptivity ratings.
Import this at the top of each notebook:

    from data_loading import load_data
    df = load_data()

Each row is one word with columns:
    human                    – mean human interoceptive rating (0–5)
    qwen_response            – Qwen discrete response (0–5)
    qwen_expected            – Qwen probability-weighted expected score
    llama_instruct_response  – Llama-Instruct discrete response
    llama_instruct_expected  – Llama-Instruct expected score
    llama_base_response      – Llama-Base discrete response (cleaned)
    llama_base_expected      – Llama-Base expected score
    <prefix>_error           – model expected score minus human rating

Paths are resolved relative to this file, so the notebooks run from a clone of
the repository without editing anything. Set the THESIS_DATA_DIR environment
variable if the data files live somewhere else.
"""

import json
import os
from pathlib import Path

import pandas as pd

# Data files are expected next to this script (override with THESIS_DATA_DIR).
DATA_DIR = Path(os.environ.get("THESIS_DATA_DIR", Path(__file__).resolve().parent))

FILES = {
    "human":          "words_full_dataset_with_interoceptive.jsonl",
    "qwen":           "prediction_qwen_temp0_full_data_set.jsonl",
    "llama_instruct": "pred_llama_instruct_temp0_full_dataset.jsonl",
    "llama_base":     "prediction_llama_base_combined_all_retries_cleaned.jsonl",
}


def _check(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name} in {path.parent}. "
            "Run the notebooks from the repository folder, or set THESIS_DATA_DIR "
            "to the folder holding the prediction files."
        )
    return path


def _load_human(path):
    with open(_check(path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows).rename(columns={"INTEROCEPTIVE": "human"})


def _load_model(path, prefix):
    with open(_check(path), encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    # Clean response column — some rows have non-numeric strings (e.g. "0=no")
    df["response"] = pd.to_numeric(df["response"], errors="coerce")

    return df[["WORD", "response", "expected_score"]].rename(columns={
        "response":       f"{prefix}_response",
        "expected_score": f"{prefix}_expected",
    })


def load_data(models=("qwen", "llama_instruct", "llama_base"), verbose=True):
    """Merge the human ratings with the predictions of the requested models.

    Pass models=("qwen", "llama_instruct") to skip the base model.
    """
    df = _load_human(DATA_DIR / FILES["human"])

    for prefix in models:
        df = df.merge(_load_model(DATA_DIR / FILES[prefix], prefix), on="WORD")

    # Error columns (model − human) for each model
    for prefix in models:
        df[f"{prefix}_error"] = df[f"{prefix}_expected"] - df["human"]

    if verbose:
        print(f"Loaded {len(df):,} words — columns: {df.columns.tolist()}")
    return df


if __name__ == "__main__":
    print(load_data().head())
