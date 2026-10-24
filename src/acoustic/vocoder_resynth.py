"""[M3 改进] 声码器重合成：剥除说话人相位/精细结构，提升匿名化而不破坏可懂度。

动机（M3 发现）：干净的频域扭曲无法骗过 ECAPA 类说话人嵌入器（VTLN 是经典
说话人归一化技术，深度嵌入器对其近乎免疫）。本模块在扭曲保护之后追加一级
"mel → HiFi-GAN → 重采样"重合成：声码器重建会剥离原始说话人的相位细节与
精细结构（EER↑），而语音内容由谱结构决定（WER 保持）。

管线：protect(warping) → resample 22.05kHz → mel → HiFi-GAN → 16kHz
注意：声码器在 22.05kHz 上工作，输出重采样回 16kHz；重合成本身也构成一层
处理损伤（基线 WER 需先标定，见 scripts/vocoder_calibration.py）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

VOCODER_SR = 22050
MEL_PARAMS = dict(
    sample_rate=VOCODER_SR,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    f_min=0.0,
    f_max=8000.0,
    n_mels=80,
    power=1.0,
    norm="slaney",
    mel_scale="slaney",
    center=True,
    pad_mode="reflect",
)

_vocoder_cache = {}


def load_vocoder(device: str = "cuda:0", savedir: str | None = None):
    """加载 SpeechBrain HiFi-GAN（LibriTTS 22050Hz），单例缓存。"""
    import torch
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy

    key = device
    if key in _vocoder_cache:
        return _vocoder_cache[key]
    savedir = savedir or os.path.join(PROJ, "data", "pretrained", "hifigan-libritts-22050")
    voc = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-22050Hz",
        savedir=savedir,
        run_opts={"device": device},
        local_strategy=(
            LocalStrategy.COPY_SKIP_CACHE if sys.platform == "win32" else LocalStrategy.SYMLINK
        ),
    )
    _vocoder_cache[key] = (voc, torch)
    return _vocoder_cache[key]


def resynthesize(wav16k: np.ndarray, device: str = "cuda:0",
                 norm: str = "ln") -> np.ndarray:
    """16kHz 波形 → (重采样 22.05k) → mel → HiFi-GAN → 16kHz 波形。

    norm: mel 归一化方式，'ln'（自然对数）或 'log10'——训练配方的确切约定
    需实测标定（见 scripts/vocoder_calibration.py）。
    """
    import torch
    import torchaudio

    voc, _ = load_vocoder(device)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        window_fn=torch.hann_window, **MEL_PARAMS
    ).to(device)

    w = torch.as_tensor(np.asarray(wav16k, np.float32)).unsqueeze(0)
    w22 = torchaudio.functional.resample(w, 16000, VOCODER_SR).to(device)
    mel = mel_fn(w22)                                   # (1, 80, T)
    if norm == "ln":
        mel = torch.log(torch.clamp(mel, min=1e-5))
    else:
        mel = torch.log10(torch.clamp(mel, min=1e-5))
    with torch.no_grad():
        out = voc.decode_batch(mel)
    out = out.squeeze().detach().cpu().numpy().ravel()
    out16 = torchaudio.functional.resample(
        torch.as_tensor(out).unsqueeze(0), VOCODER_SR, 16000
    ).squeeze().numpy()
    # 幅度对齐：声码器输出电平可能与输入不同，按 RMS 匹配（防响度突变）
    rms_in = np.sqrt(np.mean(wav16k ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(out16 ** 2) + 1e-12)
    out16 = out16 * (rms_in / rms_out)
    return np.clip(out16, -1.0, 1.0)
