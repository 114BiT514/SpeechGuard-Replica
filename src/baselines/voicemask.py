"""[P2-1 / M5 任务5] VoiceMask 基线（论文 [11,12]，Fig.10/11/6 的四方法之一）。

VoiceMask 与其余三家（单参数/McAdams/Overo 都是频域 warp）机制不同：它是**时域**
可逆匿名化——把语音切成等长段，每段做时间反转（时间倒放），恢复即再反转一次。
参数只有段长 L（样本），天然可逆、user-agnostic，正对应论文 Table 1 里
VoiceMask 那行"User-agnostic✔ / Recoverable✔ / 无 customization✘"。

本模块提供：
  protect(wav, rng) -> (masked, params)   params=(L,)，L 从 [Lmin,Lmax] 均匀抽
  recover(masked, params) -> wav          再次段内反转 = 原信号（确定性可逆）
  attack(masked, rng) -> (guess_params, dist_to_original_proxy)
      reducing 攻击：猜段长 L'→恢复→以"恢复信号与原始信号的距离"作为 distortion
      强度代理（论文对 warp 用 dist(m_att⁻¹∘m_true)，对时域方法用等价物：
      错误 L' 使恢复残余失真更大；正确 L'→失真 0）。成功判据同 exp4：
      dist_after_attack < dist_after_protect。
  与频域三家同框对比时，attack_rate 用同一"越接近原始越成功"语义（见 exp6）。

L 用样本数，段长 8–40ms（128–640 @16k），段内反转对可懂度的破坏远大于随机
置换——VoiceMask 的匿名性主要来自反转破坏时序共振峰轨迹。
"""
from __future__ import annotations

import numpy as np

SR = 16000
L_RANGE = (128, 640)          # 段长样本数（8–40ms）


def _seg_reverse(x, L):
    y = x[: (len(x) // L) * L].reshape(-1, L)[:, ::-1].ravel()
    return np.concatenate([y, x[(len(x) // L) * L:]])   # 尾段原样保留


def protect(wav, rng=None, L=None):
    rng = np.random.default_rng() if rng is None else rng
    L = int(rng.integers(L_RANGE[0], L_RANGE[1] + 1)) if L is None else int(L)
    x = np.asarray(wav, dtype=np.float64).ravel()
    return _seg_reverse(x, L), (L,)


def recover(masked, params):
    (L,) = params
    return _seg_reverse(np.asarray(masked, dtype=np.float64).ravel(), L)


def attack_guess(rng):
    return (int(rng.integers(L_RANGE[0], L_RANGE[1] + 1)),)


def distortion_to_original(recovered, original):
    """攻击失真强度代理：归一化均方距离（越小=越接近原始=攻击越'成功'）。
    对保护态本身用相同定义（recover 用错误 L 的'伪恢复'距离）以对齐阈值语义。
    """
    n = min(len(recovered), len(original))
    e = float(np.mean((recovered[:n] - original[:n]) ** 2))
    v = float(np.mean(original[:n] ** 2)) + 1e-12
    return e / v
