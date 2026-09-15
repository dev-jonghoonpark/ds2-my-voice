"""학습된 체크포인트로 데모 페이지 데이터(docs/data.js, docs/audio/)를 만든다.

docs/ 폴더는 그대로 GitHub Pages로 배포할 수 있는 정적 사이트다.
index.html / app.js / style.css 는 손으로 쓴 파일이고, 이 스크립트는 데이터와 오디오만 다시 만든다.

사용법:
  uv run scripts/build_demo.py
  cd docs && python -m http.server 8000   # 로컬 미리보기
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from compare import transcribe_all
from ds2.data import SpeechDataset, collate, read_manifest
from ds2.decode import cer, wer
from ds2.features import SAMPLE_RATE, load_audio
from ds2.tokenizer import KoreanJamoTokenizer


def epoch_curve(log_path: str) -> list[dict]:
    """학습 로그에서 epoch별 valid CER을 뽑는다 (tqdm 때문에 같은 줄이 여러 번 찍힐 수 있어 epoch으로 중복 제거)."""
    text = Path(log_path).read_text(encoding="utf-8").replace("\r", "\n")
    curve = {int(m[1]): float(m[2]) for m in re.finditer(r"^\[epoch (\d+)\][^\n]*valid CER ([\d.]+)", text, re.M)}
    return [{"epoch": e, "cer": c} for e, c in sorted(curve.items())]


def manifest_stats(path: str) -> dict:
    rows = read_manifest(path)
    return {"count": len(rows), "minutes": round(sum(r["duration"] for r in rows) / 60, 1)}


def evaluate_set(name, manifest, ckpts, tokenizer, device, n_samples, max_sample_dur, audio_dir):
    ds = SpeechDataset([manifest], tokenizer, max_duration=float("inf"))
    loader = DataLoader(ds, batch_size=16, collate_fn=collate)
    refs = [tokenizer.normalize(r["text"]) for r in ds.rows]
    hyps = {key: transcribe_all(path, loader, tokenizer, device) for key, path in ckpts.items()}

    samples = []
    for i, row in enumerate(ds.rows):
        if len(samples) >= n_samples:
            break
        if row["duration"] > max_sample_dur:
            continue
        out = audio_dir / f"{name}_{len(samples):02d}.wav"
        sf.write(out, load_audio(row["audio"]), SAMPLE_RATE)
        samples.append({"audio": f"audio/{out.name}", "duration": round(row["duration"], 2),
                        "ref": refs[i], **{key: hyps[key][i] for key in hyps}})

    return {
        "count": len(refs),
        "metrics": {key: {"cer": round(cer(refs, h), 4), "wer": round(wer(refs, h), 4)} for key, h in hyps.items()},
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/base/best.pt")
    ap.add_argument("--mine", default="checkpoints/my_voice/best.pt")
    ap.add_argument("--base-log", default="checkpoints/base/train.log")
    ap.add_argument("--mine-log", default="checkpoints/my_voice.log")
    ap.add_argument("--zeroth-train", default="data/zeroth/train.jsonl")
    ap.add_argument("--zeroth-test", default="data/zeroth/test.jsonl")
    ap.add_argument("--my-train", default="data/my_voice/train.jsonl")
    ap.add_argument("--my-test", default="data/my_voice/test.jsonl")
    ap.add_argument("--my-samples", type=int, default=100, help="내 목소리 테스트 샘플 수 (기본: 전부)")
    ap.add_argument("--zeroth-samples", type=int, default=8)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoreanJamoTokenizer()
    out = Path(args.out)
    audio_dir = out / "audio"
    shutil.rmtree(audio_dir, ignore_errors=True)
    audio_dir.mkdir(parents=True)

    ckpts = {"base": args.base, "mine": args.mine}
    base_ckpt = torch.load(args.base, map_location="cpu")
    n_params = sum(v.numel() for k, v in base_ckpt["model"].items() if "running" not in k and "num_batches" not in k)

    data = {
        "model": {**base_ckpt["config"], "params_m": round(n_params / 1e6, 1)},
        "train": {"zeroth": manifest_stats(args.zeroth_train), "my_voice": manifest_stats(args.my_train)},
        "curves": {"base": epoch_curve(args.base_log), "mine": epoch_curve(args.mine_log)},
        "sets": {
            "my_voice": evaluate_set("my_voice", args.my_test, ckpts, tokenizer, device,
                                     args.my_samples, float("inf"), audio_dir),
            "zeroth": evaluate_set("zeroth", args.zeroth_test, ckpts, tokenizer, device,
                                   args.zeroth_samples, 10.0, audio_dir),
        },
    }
    (out / "data.js").write_text("window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n",
                                 encoding="utf-8")
    (out / ".nojekyll").touch()

    for key, s in data["sets"].items():
        m = s["metrics"]
        print(f"{key:<9} base CER {m['base']['cer']:.3f} -> mine CER {m['mine']['cer']:.3f}  (샘플 {len(s['samples'])}개)")
    size = sum(f.stat().st_size for f in audio_dir.iterdir()) / 1e6
    print(f"{out}/data.js, {out}/audio/ ({size:.1f} MB) 생성 완료")


if __name__ == "__main__":
    main()
