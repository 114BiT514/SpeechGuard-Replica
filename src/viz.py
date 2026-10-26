"""[M6 任务2] 报告/前端可视化素材：频谱三联图、敏感词时间轴、指标曲线。

matplotlib Agg 后端，中文字体 SimHei；所有函数返回保存路径，可直接进报告与前端。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SR = 16000
_NFFT = 1024


def _ax_spec(ax, wav, title, fs=SR, first=False):
    *_, im = ax.specgram(np.asarray(wav, dtype=np.float64), NFFT=_NFFT, Fs=fs,
                         noverlap=_NFFT // 2, cmap="magma", vmin=-90, vmax=-10)
    ax.set_title(title, fontsize=10)
    if first:
        ax.set_ylabel("频率 (Hz)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return im


def spectrogram_row(wavs: list[tuple[np.ndarray, str]], path: str,
                    fs: int = SR, height: float = 2.6) -> str:
    """一行多联频谱图（原始/保护/攻击/恢复），共用色标。"""
    n = len(wavs)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 1.2, height), dpi=150)
    axes = np.atleast_1d(axes)
    im = None
    for k, (w, title) in enumerate(wavs):
        im = _ax_spec(axes[k], w, title, fs=fs, first=(k == 0))
        axes[k].set_xlabel("时间 (s)", fontsize=8.5)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("强度 (dB)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def sensitive_timeline(ann, dur: float, path: str) -> str:
    """敏感项时间轴（发布页人工增删的可视化）。"""
    items = [it for it in ann.items if not it.deleted]
    fig, ax = plt.subplots(figsize=(9, 1.6), dpi=150)
    ycats = {}
    for it in items:
        ycats.setdefault(it.category, len(ycats))
        ax.barh(ycats[it.category], it.t1 - it.t0, left=it.t0, height=0.6,
                color="#b45536", alpha=0.85)
        ax.text(it.t0, ycats[it.category] + 0.45, it.text, fontsize=7, va="bottom")
    ax.set_yticks(range(len(ycats)))
    ax.set_yticklabels(list(ycats), fontsize=8)
    ax.set_xlim(0, max(dur, 1))
    ax.set_xlabel("时间 (s)", fontsize=8.5)
    ax.set_ylim(-0.6, max(len(ycats), 1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
