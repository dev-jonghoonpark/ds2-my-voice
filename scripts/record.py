"""문장을 하나씩 보여주고 내 목소리를 녹음한다.

사용법: uv run scripts/record.py --prompts prompts/ko_prompts.txt --out data/my_voice
  Enter: 녹음 시작 → Enter: 녹음 종료
  p: 방금 녹음 들어보기 / r: 방금 문장 다시 녹음 / s: 건너뛰기 / q: 종료 (이어서 녹음 가능)
"""

import argparse
import json
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def record_until_enter() -> np.ndarray:
    import sounddevice as sd  # PortAudio 필요: sudo apt install libportaudio2

    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=lambda d, *_: q.put(d.copy())):
        input("  ● 녹음 중... 끝나면 Enter")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)


def play(audio: np.ndarray):
    import sounddevice as sd

    print("  ▶ 재생 중...")
    sd.play(audio, SR)
    sd.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/ko_prompts.txt")
    ap.add_argument("--out", default="data/my_voice")
    ap.add_argument("--device", default=None, help="입력 장치 번호/이름 (python -m sounddevice 로 확인)")
    args = ap.parse_args()
    if args.device is not None:
        import sounddevice as sd
        # (입력, 출력) 쌍. 출력은 None으로 두어 들어보기는 시스템 기본 스피커로 재생한다.
        sd.default.device = (int(args.device) if args.device.isdigit() else args.device, None)

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(l)["id"] for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#")]
    print(f"문장 {len(prompts)}개 중 {len(done)}개 녹음됨. 조용한 곳에서 평소 말투로 읽어주세요.\n")

    i = 0
    while i < len(prompts):
        uid = f"utt{i:04d}"
        if uid in done:
            i += 1
            continue
        print(f"[{i + 1}/{len(prompts)}] {prompts[i]}")
        cmd = input("  Enter=녹음, s=건너뛰기, q=종료 > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        audio = record_until_enter()
        dur = len(audio) / SR
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        print(f"  길이 {dur:.1f}s, 최대 음량 {peak:.2f}" + ("  ⚠ 너무 작음" if peak < 0.05 else "") +
              ("  ⚠ 클리핑" if peak > 0.99 else ""))
        if dur < 0.5:
            continue
        while (ans := input("  저장? (Enter=저장, p=들어보기, r=다시) > ").strip().lower()) == "p":
            play(audio)
        if ans == "r":
            continue
        wav = out / "wav" / f"{uid}.wav"
        sf.write(wav, audio, SR)
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": uid, "audio": str(wav), "text": prompts[i], "duration": dur},
                               ensure_ascii=False) + "\n")
        i += 1

    n = sum(1 for _ in open(meta_path, encoding="utf-8")) if meta_path.exists() else 0
    print(f"\n총 {n}개 녹음. 다음: uv run scripts/split_my_voice.py")


if __name__ == "__main__":
    main()
