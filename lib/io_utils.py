from pathlib import Path
import pandas as pd

MANIFEST_COLUMNS = ["youtube_id", "start", "end", "label_mapped"]
LEDGER_COLUMNS = ["source", "youtube_id", "start", "end", "label", "status", "out_path", "sr", "cliplen", "note"]

def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def read_manifest(path):
    return pd.read_csv(path, sep="\t")

def write_manifest(df, path):
    path = Path(path)
    _ensure_parent(path)
    df.to_csv(path, sep="\t", index=False)

def append_ledger(row: dict, ledger_path):
    ledger_path = Path(ledger_path)
    _ensure_parent(ledger_path)
    if not ledger_path.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(ledger_path, index=False)
    pd.DataFrame([row])[LEDGER_COLUMNS].to_csv(ledger_path, mode="a", header=False, index=False)
