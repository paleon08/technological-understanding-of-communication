# lib/scripts/embed_from_folder.py
from pathlib import Path
import argparse
from tuc.loading_base_model import ingest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_root", required=True, help="input folder containing audio files")
    ap.add_argument("--out", dest="out_root", required=True, help="output folder for embeddings")
    ap.add_argument("--backend", default=None, choices=[None, "external", "wav2vec2"],
                    help="audio backend (None=external default)")
    ap.add_argument("--model-id", default=None, help="HF audio model id (optional)")
    args = ap.parse_args()

    ingest.embed_from_folder(args.in_root, args.out_root,
                             backend=args.backend, model_id=args.model_id)

if __name__ == "__main__":
    main()
