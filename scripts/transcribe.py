"""파일 또는 마이크 입력을 받아쓴다.

  uv run scripts/transcribe.py --ckpt checkpoints/my_voice/best.pt --file some.wav
  uv run scripts/transcribe.py --ckpt checkpoints/my_voice/best.pt --mic
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ds2.decode import greedy_decode
from ds2.features import load_audio, spectrogram
from ds2.model import DeepSpeech2
from ds2.tokenizer import KoreanJamoTokenizer
from record import record_until_enter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", nargs="+")
    g.add_argument("--mic", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoreanJamoTokenizer()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DeepSpeech2(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    def run(audio):
        spec = spectrogram(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs, out_lengths = model(spec, torch.tensor([spec.shape[-1]], device=device))
        return tokenizer.decode(greedy_decode(log_probs, out_lengths)[0])

    if args.file:
        for f in args.file:
            print(f"{f}: {run(load_audio(f))}")
        return
    while True:
        if input("Enter=녹음 시작, q=종료 > ").strip().lower() == "q":
            break
        print("→", run(record_until_enter()))


if __name__ == "__main__":
    main()
