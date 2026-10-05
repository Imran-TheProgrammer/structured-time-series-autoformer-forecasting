"""Autoformer for Task 2 (AI651 Assignment 1, Leaderboard Challenge).

Architecture: Wu, Xu, Wang and Long (2021), "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting", sections 3.1-3.2. The layer layout
(embedding without positions, seasonal layer norm, decoder start, trend accumulation)
follows the authors' public code, https://github.com/thuml/Autoformer. The code below is
written from scratch.

Reused from my Task 1 notebook: SeriesDecomposition.forward and aggregate_delays (my own
implementations) and delay_scores (supplied by the course).

Differences from the paper, each one a choice tested in task2.py:
  * the starting trend of the forecast can fade away instead of staying at the window mean
    (three options, see Autoformer.forward);
  * delays are chosen per example, as in Task 1, and weighted with a learned temperature;
  * one head, because the delay scores are shared by all heads and channels anyway.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomposition(nn.Module):
    """Centered moving average: x = remainder + trend. Ends repeat the nearest value."""

    def __init__(self, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):                       # x [B,L,C] -> (remainder, trend), each [B,L,C]
        radius = self.kernel // 2
        padded = F.pad(x.transpose(1, 2), (radius, radius), mode="replicate")  # [B,C,L+2*radius]
        trend = F.avg_pool1d(padded, self.kernel, stride=1).transpose(1, 2)    # [B,L,C]
        return x - trend, trend


def delay_scores(queries, keys):
    # queries, keys: [B,heads,features,L] -> scores [B,L], one per delay tau = 0 .. L-1.
    # R(tau) = sum_t q[t] * k[(t - tau) mod L], centered in time, averaged over heads and
    # features. An FFT computes every delay's score together in O(L log L).
    queries = queries - queries.mean(-1, keepdim=True)
    keys = keys - keys.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(queries, dim=-1) * torch.fft.rfft(keys, dim=-1).conj()
    return torch.fft.irfft(spectrum, n=queries.shape[-1], dim=-1).mean((1, 2))


def aggregate_delays(values, delays, weights):
    # values: [B,heads,features,L]; delays, weights: [B,K] -> [B,heads,features,L]
    # z[t] = sum_j weights[j] * values[(t - delays[j]) mod L], per example.
    length = values.shape[-1]
    positions = torch.arange(length, device=values.device)
    mixed = torch.zeros_like(values)
    for j in range(delays.shape[-1]):
        source = (positions - delays[:, j, None]) % length              # [B,L]: t reads t - delay
        shifted = values.gather(-1, source[:, None, None, :].expand_as(values))
        mixed = mixed + weights[:, j, None, None, None] * shifted
    return mixed


class AutoCorrelation(nn.Module):
    """Score every delay, keep the best k = floor(factor * ln L), mix the shifted values."""

    def __init__(self, d, factor=1):
        super().__init__()
        self.factor = factor
        self.query, self.key, self.value = (nn.Linear(d, d) for _ in range(3))
        self.out = nn.Linear(d, d)
        self.raw_temperature = nn.Parameter(torch.tensor(0.5413248546))   # softplus -> 1.0

    @property
    def temperature(self):
        return F.softplus(self.raw_temperature) + 1e-6

    def forward(self, x, memory):               # x [B,L,d] reads from memory [B,S,d] -> [B,L,d]
        length = x.shape[1]
        q, k, v = self.query(x), self.key(memory), self.value(memory)
        if memory.shape[1] < length:            # make the memory as long as x, as the paper does
            pad = (0, 0, 0, length - memory.shape[1])
            k, v = F.pad(k, pad), F.pad(v, pad)
        else:
            k, v = k[:, :length], v[:, :length]
        q, k, v = (z.transpose(1, 2).unsqueeze(1) for z in (q, k, v))      # [B,1,d,L]
        scores = delay_scores(q, k) / length                               # [B,L]
        top_k = max(1, int(self.factor * math.log(length)))
        selected, delays = scores.topk(top_k, dim=-1)                      # [B,K]
        weights = (selected / self.temperature).softmax(-1)                # [B,K]
        mixed = aggregate_delays(v, delays, weights)                       # [B,1,d,L]
        return self.out(mixed.squeeze(1).transpose(1, 2))


class SeasonalNorm(nn.Module):
    """Layer norm, then remove each channel's mean over time (the paper's seasonal norm)."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                       # [B,L,d]
        x = self.norm(x)
        return x - x.mean(1, keepdim=True)


class Embedding(nn.Module):
    """Value embedding plus an embedding of the known-ahead features. No positions.

    The features enter through one linear layer ("linear") or through a small network ("mlp")
    of `marks_layers` linear layers (2 or 3) whose middle is `marks_hidden` times the model width.
    """

    def __init__(self, marks, d, dropout, marks_net="linear", marks_hidden=1, marks_layers=2):
        super().__init__()
        if marks_net not in {"linear", "mlp"}:
            raise ValueError("marks_net must be 'linear' or 'mlp'")
        self.value = nn.Conv1d(1, d, 3, padding=1, padding_mode="circular", bias=False)
        if not marks:
            self.mark = None
        elif marks_net == "linear":
            self.mark = nn.Linear(marks, d, bias=False)
        else:
            if marks_layers not in {2, 3}:
                raise ValueError("marks_layers must be 2 or 3")
            hidden = marks_hidden * d
            middle = [nn.Linear(hidden, hidden), nn.GELU()] if marks_layers == 3 else []
            self.mark = nn.Sequential(nn.Linear(marks, hidden), nn.GELU(), *middle,
                                      nn.Linear(hidden, d, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mark):                 # x [B,L,1], mark [B,L,M] -> [B,L,d]
        embedded = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.mark is not None:
            embedded = embedded + self.mark(mark)
        return self.drop(embedded)


def feed_forward(d, dropout):
    return nn.Sequential(nn.Linear(d, 2 * d, bias=False), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(2 * d, d, bias=False))


class EncoderLayer(nn.Module):
    def __init__(self, d, kernel, factor, dropout):
        super().__init__()
        self.mix = AutoCorrelation(d, factor)
        self.feed_forward = feed_forward(d, dropout)
        self.strip1, self.strip2 = SeriesDecomposition(kernel), SeriesDecomposition(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # [B,S,d]; the removed trends are dropped
        x, _ = self.strip1(x + self.drop(self.mix(x, x)))
        x, _ = self.strip2(x + self.drop(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, kernel, factor, dropout):
        super().__init__()
        self.self_mix, self.cross_mix = AutoCorrelation(d, factor), AutoCorrelation(d, factor)
        self.feed_forward = feed_forward(d, dropout)
        self.strip1, self.strip2, self.strip3 = (SeriesDecomposition(kernel) for _ in range(3))
        self.trend = nn.Conv1d(d, 1, 3, padding=1, padding_mode="circular", bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, encoded):              # [B,L,d], [B,S,d] -> remainder [B,L,d], trend [B,L,1]
        x, trend1 = self.strip1(x + self.drop(self.self_mix(x, x)))
        x, trend2 = self.strip2(x + self.drop(self.cross_mix(x, encoded)))
        x, trend3 = self.strip3(x + self.drop(self.feed_forward(x)))
        removed = trend1 + trend2 + trend3
        return x, self.trend(removed.transpose(1, 2)).transpose(1, 2)


class Autoformer(nn.Module):
    def __init__(self, marks, seq_len=168, label_len=84, pred_len=168, d=32, e_layers=1,
                 d_layers=1, kernel=25, factor=1, dropout=0.1, trend_start="fade",
                 marks_net="linear", marks_hidden=1, marks_layers=2):
        super().__init__()
        if trend_start not in {"mean", "fade", "anchor"}:
            raise ValueError("trend_start must be 'mean', 'fade' or 'anchor'")
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.trend_start = trend_start
        self.split = SeriesDecomposition(kernel)
        self.encoder_embed = Embedding(marks, d, dropout, marks_net, marks_hidden, marks_layers)
        self.decoder_embed = Embedding(marks, d, dropout, marks_net, marks_hidden, marks_layers)
        self.encoder = nn.ModuleList(EncoderLayer(d, kernel, factor, dropout)
                                     for _ in range(e_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d, kernel, factor, dropout)
                                     for _ in range(d_layers))
        self.encoder_norm, self.decoder_norm = SeasonalNorm(d), SeasonalNorm(d)
        self.projection = nn.Linear(d, 1)
        if trend_start != "mean":               # sigmoid -> 0.966, the lag-1 autocorrelation
            self.raw_fade = nn.Parameter(torch.tensor(math.log(0.966 / (1 - 0.966))))

    @property
    def fade(self):
        return torch.sigmoid(self.raw_fade)

    def forward(self, x, mark_enc, mark_dec):   # [B,S,1], [B,S,M], [B,label+pred,M] -> [B,pred]
        seasonal, trend = self.split(x)
        zeros = torch.zeros(x.shape[0], self.pred_len, 1, device=x.device)
        seasonal = torch.cat([seasonal[:, -self.label_len:], zeros], 1)      # known remainder, then zeros

        encoded = self.encoder_embed(x, mark_enc)
        for layer in self.encoder:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)

        decoded = self.decoder_embed(seasonal, mark_dec)
        removed_trend = 0
        for layer in self.decoder:
            decoded, removed = layer(decoded, encoded)
            removed_trend = removed_trend + removed
        own = removed_trend + self.projection(self.decoder_norm(decoded))
        own = own[:, -self.pred_len:, 0]        # [B,pred]: what the decoder itself forecasts

        # The starting trend is only ever added to the decoder's output (no layer reads it),
        # so it is chosen here. All values are scaled, so 0 is the long-run mean.
        if self.trend_start == "mean":          # paper: the level of the input window persists
            return x.mean(1) + own
        steps = torch.arange(1, self.pred_len + 1, device=x.device)
        if self.trend_start == "fade":          # the last smoothed level fades to the long-run mean
            level = trend[:, -1]
        else:                                   # "anchor": start at the last observed value, i.e. fade
            level = x[:, -1] - own[:, :1]       # the gap between it and the decoder's first step
        return level * self.fade ** steps + own


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
