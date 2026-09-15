"""녹음을 train/test로 나눈다. test 문장은 학습에 절대 쓰지 않아야 개선 효과를 정직하게 측정할 수 있다.

사용법: uv run scripts/split_my_voice.py --dir data/my_voice --test-ratio 0.15
"""

import argparse
import random
from pathlib import Path

from ds2.data import read_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/my_voice")
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.dir)
    rows = read_manifest(d / "metadata.jsonl")
    random.Random(args.seed).shuffle(rows)
    n_test = max(1, int(len(rows) * args.test_ratio))
    test, train = rows[:n_test], rows[n_test:]
    write_manifest(d / "train.jsonl", train)
    write_manifest(d / "test.jsonl", test)
    for name, rs in [("train", train), ("test", test)]:
        print(f"{name}: {len(rs)}개, {sum(r['duration'] for r in rs) / 60:.1f}분")


if __name__ == "__main__":
    main()
