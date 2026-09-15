"""오디오 로딩과 특징 추출 (DeepSpeech2 논문과 같은 선형 스펙트로그램)."""

import numpy as np
import soundfile as sf
import soxr
import torch

SAMPLE_RATE = 16000
WIN_LENGTH = 320   # 20ms
HOP_LENGTH = 160   # 10ms
N_FFT = 320        # -> 161개 주파수 bin
N_FREQ = N_FFT // 2 + 1


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
    return audio


def spectrogram(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    """log(1 + |STFT|) 후 발화 단위로 평균/분산 정규화. 반환 shape: (freq, time)."""
    x = torch.as_tensor(audio, dtype=torch.float32)
    stft = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=torch.hamming_window(WIN_LENGTH), return_complex=True,
    )
    spec = torch.log1p(stft.abs())
    return (spec - spec.mean()) / (spec.std() + 1e-5)


def spec_augment(spec: torch.Tensor, freq_masks=2, freq_width=27, time_masks=2, time_width=40) -> torch.Tensor:
    """SpecAugment (2019). 원래 DS2에는 없지만, 내 목소리처럼 데이터가 적을 때 과적합을 크게 줄여준다."""
    spec = spec.clone()
    n_freq, n_time = spec.shape
    for _ in range(freq_masks):
        w = np.random.randint(0, freq_width + 1)
        f0 = np.random.randint(0, max(1, n_freq - w))
        spec[f0:f0 + w, :] = 0
    for _ in range(time_masks):
        w = np.random.randint(0, min(time_width, n_time // 5) + 1)
        t0 = np.random.randint(0, max(1, n_time - w))
        spec[:, t0:t0 + w] = 0
    return spec
