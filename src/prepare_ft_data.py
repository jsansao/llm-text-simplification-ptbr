import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_pairs, PSS2_ORI_NAT, _read_tsv

DATA_DIR = Path("data") / "ft"
RANDOM_STATE = 42
N_HOLDOUT = 200
TRAIN_RATIO = 0.9


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load full dataset (preserve original indices from the TSV)
    full = _read_tsv(PSS2_ORI_NAT)
    full.columns = [c.strip().lower() for c in full.columns]
    full = full.rename(columns={
        "sentence_text_from": "original",
        "sentence_text_to": "simplified",
    })
    full = full[full["changed"].str.strip().str.upper() == "S"].copy()
    print(f"Dataset completo (apenas modificados): {len(full)} pares")

    # 2. Stratified hold-out by tercile of token count (same method as load_pairs)
    full["_n_tokens"] = full["original"].str.split().str.len()
    full["_stratum"] = pd.qcut(full["_n_tokens"], q=3, labels=[0, 1, 2])
    holdout = full.groupby("_stratum", group_keys=False).apply(
        lambda g: g.sample(n=N_HOLDOUT // 3, random_state=RANDOM_STATE)
    )
    remaining = N_HOLDOUT - len(holdout)
    if remaining > 0:
        extra = full.drop(holdout.index).sample(n=remaining, random_state=RANDOM_STATE)
        holdout = pd.concat([holdout, extra])
    holdout = holdout.drop(columns=["_n_tokens", "_stratum"])

    # 3. Remove hold-out by actual index (not reset index)
    train_val = full.drop(holdout.index).drop(columns=["_n_tokens", "_stratum"])
    print(f"Restante para treino/val: {len(train_val)} pares")
    print(f"Interseção holdout ∩ restante: {len(set(holdout.index) & set(train_val.index))}")

    # 4. 90/10 split
    train_val = train_val.reset_index(drop=True)
    train = train_val.sample(frac=TRAIN_RATIO, random_state=RANDOM_STATE)
    val = train_val.drop(train.index)
    print(f"Treino: {len(train)} | Validação: {len(val)}")
    assert len(train) + len(val) + len(holdout) == len(full)

    # 5. Save PTT5 format (.json list)
    def to_ptt5(df):
        return [{"input": f"simplify: {row['original']}", "output": row["simplified"]}
                for _, row in df.iterrows()]

    for name, df in [("ptt5_train", train), ("ptt5_val", val), ("ptt5_holdout", holdout)]:
        path = DATA_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_ptt5(df), f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(df)} exemplos")

    # 6. Save Qwen format (.jsonl)
    def to_qwen(df):
        for _, row in df.iterrows():
            yield {
                "input": row["original"],
                "output": row["simplified"],
                "messages": [
                    {"role": "user", "content": row["original"]},
                    {"role": "assistant", "content": row["simplified"]},
                ],
            }

    for name, df in [("qwen_train", train), ("qwen_val", val), ("qwen_holdout", holdout)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in to_qwen(df):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(df)} exemplos")

    print("OK: dados preparados sem vazamento.")


if __name__ == "__main__":
    main()
