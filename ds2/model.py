"""DeepSpeech2 모델 (Amodei et al., 2015, arXiv:1512.02595).

  spectrogram ─► Conv2d x2 (+BN, Hardtanh) ─► BiGRU x N (+sequence-wise BN) ─► Linear ─► CTC

논문 원본은 최대 11GB급 모델이지만, RTX 3070(8GB)에서 돌 수 있도록 크기를 줄였다.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .features import N_FREQ


class SequenceWise(nn.Module):
    """(B, T, H) 텐서에 BatchNorm1d를 타임스텝별로 적용 (논문의 sequence-wise BN)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        b, t, h = x.shape
        return self.module(x.reshape(b * t, h)).reshape(b, t, h)


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super().__init__()
        self.bn = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x, lengths):
        if self.bn is not None:
            x = self.bn(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        # 양방향 출력을 더한다 (concat 대신 sum: 파라미터 절약, 논문 구현들도 흔히 사용)
        h = out.size(-1) // 2
        return out[..., :h] + out[..., h:]


class DeepSpeech2(nn.Module):
    def __init__(self, n_classes: int, rnn_hidden=512, rnn_layers=5, conv_channels=32):
        super().__init__()
        self.config = dict(n_classes=n_classes, rnn_hidden=rnn_hidden,
                           rnn_layers=rnn_layers, conv_channels=conv_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(conv_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(conv_channels),
            nn.Hardtanh(0, 20, inplace=True),
        )
        freq = (N_FREQ + 2 * 20 - 41) // 2 + 1  # conv1: 161 -> 81
        freq = (freq + 2 * 10 - 21) // 2 + 1    # conv2:  81 -> 41
        self.rnn_input = conv_channels * freq

        self.rnns = nn.ModuleList(
            BatchRNN(self.rnn_input if i == 0 else rnn_hidden, rnn_hidden, batch_norm=i > 0)
            for i in range(rnn_layers)
        )
        self.fc = nn.Sequential(
            SequenceWise(nn.BatchNorm1d(rnn_hidden)),
            nn.Linear(rnn_hidden, n_classes, bias=False),
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        # 시간축 stride는 첫 conv에서만 2 (kernel 11, padding 5)
        return (lengths + 2 * 5 - 11) // 2 + 1

    def forward(self, spec: torch.Tensor, lengths: torch.Tensor):
        """spec: (B, F, T) -> log_probs: (B, T', C), out_lengths: (B,)"""
        x = self.conv(spec.unsqueeze(1))            # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out_lengths = self.output_lengths(lengths)
        for rnn in self.rnns:
            x = rnn(x, out_lengths)
        return self.fc(x).log_softmax(-1), out_lengths
