import csv
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_URL = "https://github.com/sidleal/porsimplessent.git"
DATA_DIR = Path("data") / "porsimplessent"

PSS2_ORI_NAT = DATA_DIR / "pss" / "pss2_align_length_ori_nat.tsv"
PSS2_ORI_STR = DATA_DIR / "pss" / "pss2_align_length_ori_str.tsv"
PSS3_ORI_NAT = DATA_DIR / "pss" / "pss3_align_no_splits_ori_nat.tsv"
TRIPLETS = DATA_DIR / "pss" / "triplets_length.tsv"


def download_dataset():
    if DATA_DIR.exists():
        print(f"[dataset] Dataset já existe em {DATA_DIR}")
        return
    print(f"[dataset] Clonando dataset de {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", REPO_URL, str(DATA_DIR)],
        check=True, capture_output=True,
    )
    print("[dataset] Download concluído.")


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE)


def load_pairs(
    path: Path = PSS2_ORI_NAT,
    only_changed: bool = True,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    df = _read_tsv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "sentence_text_from": "original",
        "sentence_text_to": "simplified",
    })
    if only_changed:
        df = df[df["changed"].str.strip().str.upper() == "S"].copy()
    if max_samples and len(df) > max_samples:
        if random_state is not None:
            df["_n_tokens"] = df["original"].str.split().str.len()
            df["_stratum"] = pd.qcut(df["_n_tokens"], q=3, labels=[0, 1, 2])
            sampled = df.groupby("_stratum", group_keys=False).apply(
                lambda g: g.sample(n=max_samples // 3, random_state=random_state)
            )
            remaining = max_samples - len(sampled)
            if remaining > 0:
                extra = df.drop(sampled.index).sample(n=remaining, random_state=random_state)
                sampled = pd.concat([sampled, extra])
            df = sampled.drop(columns=["_n_tokens", "_stratum"], errors="ignore")
        else:
            df = df.head(max_samples).copy()
    return df.reset_index(drop=True)


def load_triplets(
    path: Path = TRIPLETS,
    only_changed: bool = True,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    df = _read_tsv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "original_text": "original",
        "natural_text": "natural",
        "strong_text": "strong",
    })
    if only_changed:
        changed = (
            df["changed_ori_nat"].str.strip().str.upper() == "S"
        ) | (
            df["changed_nat_str"].str.strip().str.upper() == "S"
        )
        df = df[changed].copy()
    if max_samples:
        df = df.head(max_samples).copy()
    return df.reset_index(drop=True)


def dataset_stats():
    download_dataset()
    pairs = load_pairs(only_changed=False)
    triplets = load_triplets(only_changed=False)
    changed_pairs = load_pairs(only_changed=True)
    print(f"Pares ORI->NAT (total):      {len(pairs)}")
    print(f"Pares ORI->NAT (simplified): {len(changed_pairs)}")
    print(f"Triplets (total):            {len(triplets)}")
    print(f"Média de tokens (original):  {pairs['original'].str.split().str.len().mean():.1f}")
    print(f"Média de tokens (simplif.):  {pairs['simplified'].str.split().str.len().mean():.1f}")
