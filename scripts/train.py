"""DeepSpeech2 학습 / 파인튜닝.

사전학습 (공개 데이터):
  uv run scripts/train.py --train data/zeroth/train.jsonl --valid data/zeroth/test.jsonl \
      --epochs 30 --lr 3e-4 --out checkpoints/base

파인튜닝 (내 목소리):
  uv run scripts/train.py --init checkpoints/base/best.pt \
      --train data/my_voice/train.jsonl --valid data/my_voice/test.jsonl \
      --epochs 40 --lr 5e-5 --freeze-conv --batch-size 16 --out checkpoints/my_voice
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ds2.data import SortaGradSampler, SpeechDataset, collate
from ds2.decode import cer, greedy_decode
from ds2.model import DeepSpeech2
from ds2.tokenizer import BLANK, KoreanJamoTokenizer


def evaluate(model, loader, tokenizer, device, show=0):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for spec, lengths, targets, target_lengths in loader:
            log_probs, out_lengths = model(spec.to(device), lengths.to(device))
            hyps += [tokenizer.decode(h) for h in greedy_decode(log_probs.float(), out_lengths)]
            refs += [tokenizer.decode(t.tolist()) for t in targets.split(target_lengths.tolist())]
    for r, h in list(zip(refs, hyps))[:show]:
        print(f"  정답: {r}\n  예측: {h}")
    return cer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", nargs="+", required=True)
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--init", help="파인튜닝 시작 체크포인트")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rnn-hidden", type=int, default=512)
    ap.add_argument("--rnn-layers", type=int, default=5)
    ap.add_argument("--freeze-conv", action="store_true", help="conv 층 고정 (저수준 음향 특징은 재사용)")
    ap.add_argument("--freeze-rnn", type=int, default=0, help="아래쪽 RNN 층 N개 고정")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-duration", type=float, default=16.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoreanJamoTokenizer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        model = DeepSpeech2(**ckpt["config"])
        model.load_state_dict(ckpt["model"])
        print(f"초기화: {args.init} (당시 valid CER {ckpt.get('cer', float('nan')):.3f})")
    else:
        model = DeepSpeech2(len(tokenizer), args.rnn_hidden, args.rnn_layers)
    model.to(device)

    if args.freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False
    for rnn in model.rnns[: args.freeze_rnn]:
        for p in rnn.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"(학습 대상 {sum(p.numel() for p in trainable) / 1e6:.1f}M), device={device}")

    train_ds = SpeechDataset(args.train, tokenizer, augment=not args.no_augment, max_duration=args.max_duration)
    valid_ds = SpeechDataset(args.valid, tokenizer, max_duration=float("inf"))  # 평가는 전부 사용
    sampler = SortaGradSampler(train_ds, args.batch_size)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers)
    print(f"train {len(train_ds)}개 / valid {len(valid_ds)}개")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, min(1000, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1)))))
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best = evaluate(model, valid_loader, tokenizer, device, show=2) if args.init else float("inf")
    if args.init:
        print(f"[epoch 0] 파인튜닝 전 valid CER {best:.3f}")
        # 파인튜닝이 오히려 나빠지기만 해도 best.pt는 '시작 모델'로 남도록
        torch.save({"model": model.state_dict(), "config": model.config, "cer": best, "epoch": 0}, out / "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        # 고정한 층의 BatchNorm 통계도 고정해야 적은 데이터에 흔들리지 않는다
        if args.freeze_conv:
            model.conv.eval()
        for rnn in model.rnns[: args.freeze_rnn]:
            rnn.eval()

        total, n = 0.0, 0
        for spec, lengths, targets, target_lengths in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            spec, lengths = spec.to(device), lengths.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                log_probs, out_lengths = model(spec, lengths)
            # CTC는 수치적으로 민감하므로 fp32로 계산. CTCLoss 입력은 (T, B, C)
            loss = ctc(log_probs.float().transpose(0, 1), targets, out_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 400)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, n = total + loss.item(), n + 1

        valid_cer = evaluate(model, valid_loader, tokenizer, device, show=2)
        print(f"[epoch {epoch}] train loss {total / max(n, 1):.3f}  valid CER {valid_cer:.3f}")
        ckpt = {"model": model.state_dict(), "config": model.config, "cer": valid_cer, "epoch": epoch}
        torch.save(ckpt, out / "last.pt")
        if valid_cer < best:
            best = valid_cer
            torch.save(ckpt, out / "best.pt")
            print(f"  ✓ best 갱신 -> {out / 'best.pt'}")
    print(f"완료. best valid CER {best:.3f}")


if __name__ == "__main__":
    main()
