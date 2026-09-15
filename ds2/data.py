"""매니페스트(JSONL) 기반 데이터셋.

한 줄 = {"audio": "경로.wav", "text": "전사문", "duration": 초}
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from .features import load_audio, spec_augment, spectrogram


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path: str | Path, rows: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class SpeechDataset(Dataset):
    def __init__(self, manifests: list[str], tokenizer, augment=False, max_duration=16.0):
        self.rows = [r for m in manifests for r in read_manifest(m)
                     if r.get("duration", 0) <= max_duration]
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        spec = spectrogram(load_audio(row["audio"]))
        if self.augment:
            spec = spec_augment(spec)
        return spec, torch.tensor(self.tokenizer.encode(row["text"]), dtype=torch.long)


def collate(batch):
    specs, targets = zip(*batch)
    lengths = torch.tensor([s.shape[1] for s in specs])
    target_lengths = torch.tensor([len(t) for t in targets])
    padded = torch.zeros(len(specs), specs[0].shape[0], int(lengths.max()))
    for i, s in enumerate(specs):
        padded[i, :, : s.shape[1]] = s
    return padded, lengths, torch.cat(targets), target_lengths


class SortaGradSampler(Sampler):
    """DS2 논문의 SortaGrad: 첫 epoch은 짧은 발화부터(커리큘럼), 이후엔 길이가 비슷한 배치를 섞는다."""

    def __init__(self, dataset: SpeechDataset, batch_size: int):
        self.order = sorted(range(len(dataset)), key=lambda i: dataset.rows[i].get("duration", 0))
        self.batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        batches = [self.order[i:i + self.batch_size] for i in range(0, len(self.order), self.batch_size)]
        if self.epoch > 0:
            random.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size
