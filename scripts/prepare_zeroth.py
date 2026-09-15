"""Zeroth-Korean(51시간, CC BY 4.0)을 내려받아 FLAC 파일 + 매니페스트로 변환한다.

내 목소리 수십 분만으로는 DS2를 처음부터 학습할 수 없다. 먼저 공개 데이터로
'한국어를 알아듣는' 기본 모델을 만든 뒤, 내 목소리로 파인튜닝한다.

사용법: uv run scripts/prepare_zeroth.py --out data/zeroth
"""

import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from ds2.data import write_manifest

REPO = "kresnik/zeroth_korean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/zeroth")
    ap.add_argument("--limit", type=int, default=0, help="빠른 실험용: split당 최대 발화 수 (0=전부)")
    args = ap.parse_args()
    out = Path(args.out)

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    for split in ["train", "test"]:
        rows = []
        split_files = sorted(f for f in files if Path(f).name.startswith(split) or f"/{split}/" in f)
        for fname in split_files:
            if args.limit and len(rows) >= args.limit:
                break
            table = pq.read_table(hf_hub_download(REPO, fname, repo_type="dataset"), columns=["id", "audio", "text"])
            for rec in tqdm(table.to_pylist(), desc=f"{split}:{Path(fname).name}"):
                if args.limit and len(rows) >= args.limit:
                    break
                path = out / split / f"{rec['id']}.flac"
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = rec["audio"]["bytes"]
                path.write_bytes(raw)
                info = sf.info(io.BytesIO(raw))
                rows.append({"audio": str(path), "text": rec["text"], "duration": info.duration})
        write_manifest(out / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} utterances, {sum(r['duration'] for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    main()
