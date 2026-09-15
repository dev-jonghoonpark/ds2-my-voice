"""CTC 디코딩과 평가 지표."""

import torch

from .tokenizer import BLANK


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """프레임별 argmax -> 연속 중복 제거 -> blank 제거.

    논문은 n-gram 언어모델을 결합한 beam search를 쓰지만, 여기선 음향모델 자체의 성능을 보기 위해 greedy만 쓴다.
    """
    best = log_probs.argmax(-1).cpu()
    results = []
    for seq, n in zip(best, lengths.cpu()):
        seq = torch.unique_consecutive(seq[:n]).tolist()
        results.append([t for t in seq if t != BLANK])
    return results


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """문자 오류율 (공백 제외). 한국어는 띄어쓰기가 흔들리므로 WER보다 CER이 적절하다."""
    errors = sum(edit_distance(r.replace(" ", ""), h.replace(" ", "")) for r, h in zip(refs, hyps))
    total = sum(len(r.replace(" ", "")) for r in refs)
    return errors / max(total, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return errors / max(total, 1)
