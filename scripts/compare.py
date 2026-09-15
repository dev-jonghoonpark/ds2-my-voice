"""여러 체크포인트를 같은 테스트셋에서 비교한다. 이 데모의 '결과물'.

사용법:
  uv run scripts/compare.py --test data/my_voice/test.jsonl \
      --ckpt base=checkpoints/base/best.pt mine=checkpoints/my_voice/best.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ds2.data import SpeechDataset, collate
from ds2.decode import cer, greedy_decode, wer
from ds2.model import DeepSpeech2
from ds2.tokenizer import KoreanJamoTokenizer


def transcribe_all(ckpt_path, loader, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DeepSpeech2(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    hyps = []
    with torch.no_grad():
        for spec, lengths, _, _ in loader:
            log_probs, out_lengths = model(spec.to(device), lengths.to(device))
            hyps += [tokenizer.decode(h) for h in greedy_decode(log_probs, out_lengths)]
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="이름=경로")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoreanJamoTokenizer()
    ds = SpeechDataset(args.test, tokenizer, max_duration=1e9)
    loader = DataLoader(ds, batch_size=16, collate_fn=collate)
    refs = [tokenizer.normalize(r["text"]) for r in ds.rows]

    results = {}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        results[name] = transcribe_all(path, loader, tokenizer, device)

    print(f"\n테스트 발화 {len(refs)}개")
    print(f"{'모델':<12}{'CER':>8}{'WER':>8}")
    for name, hyps in results.items():
        print(f"{name:<12}{cer(refs, hyps):>8.3f}{wer(refs, hyps):>8.3f}")

    print("\n예시")
    for i in range(min(args.show, len(refs))):
        print(f"\n  정답      : {refs[i]}")
        for name, hyps in results.items():
            print(f"  {name:<10}: {hyps[i]}")


if __name__ == "__main__":
    main()
