"""[复现补全] 论文 Fig.5 复现：piecewise（单参数）/ Overo（双参数）/ SpeechGuard
（多参数多折点）三种扭曲函数曲线对照（x=ω∈[0,π]，y=m(ω)，恒等虚线）。

运行：E:/Miniconda/envs/speechguard/python.exe scripts/fig5_warp_functions.py
输出：assets/fig_m6_fig5_warpfuns.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from src.acoustic.warping import PI, gen_params, multi_warp, piecewise_warp  # noqa: E402

g = np.linspace(0, PI, 801)
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
a1 = float(rng.uniform(0.5, 2.0))
ax.plot(g, piecewise_warp(g, a1), lw=1.6, color="#556973",
        label=f"piecewise（公式(2)，单参数 α={a1:.2f}）")
oa = float(rng.uniform(0.1, 0.9)) * PI
ob = float(rng.uniform(0.1, 0.9)) * PI
ax.plot(g, multi_warp(g, [oa], [ob]), lw=1.6, color="#a2bac5",
        label="Overo（双参数 α,β）")
am, bm, d = gen_params(n=6, rng=rng)
ax.plot(g, multi_warp(g, am, bm), lw=1.8, color="#b45536",
        label=f"SpeechGuard（n=6 多折点，dist={d:.2f}）")
ax.plot(g, g, "k--", lw=1, label="恒等 m(ω)=ω")
ax.scatter(np.concatenate([[0], am, [PI]]), np.concatenate([[0], bm, [PI]]),
           s=18, color="#b45536", zorder=5)
ax.set_xlabel("ω (rad)")
ax.set_ylabel("m(ω)")
ax.set_xlim(0, PI)
ax.set_ylim(0, PI)
ax.set_title("Fig.5 复现：扭曲函数对照——SpeechGuard 随机参数个数、"
             "多个折点、方向可变", fontsize=10)
ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.grid(alpha=0.25, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(PROJ, "assets", "fig_m6_fig5_warpfuns.png")
fig.savefig(out)
print("saved", out)
