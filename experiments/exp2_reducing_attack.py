"""[M3 任务4] reducing 攻击仿真（论文 §5.2 / Fig.8）。

攻击者模型：已知 n∈[1,10] 与 dist∈(0.5,0.6)，不知道具体参数。攻击 = 用随机抽取的
参数对"被保护谱"施加逆扭曲，试图靠近原始谱。成功判据（论文原文）：攻击后的
扭曲强度 < 保护后的扭曲强度，即 dist(m_att⁻¹∘m_true) < dist(m_true)。

实现为纯函数仿真：m_true、m_att⁻¹ 均为分片线性映射，复合映射
comp(ω) = m_att⁻¹(m_true(ω)) 的 dist 数值积分。攻击者参数与真实参数完全一致时
comp = identity（dist=0）→ 必然成功，作为实现的健全性检查。

**双抽样对比（本实验的核心发现）**：
- raw（论文原味）：α、β 独立排序均匀 + dist 拒绝 → 成功率 ≈18%（论文 15.87% ✓）；
- constrained（本实现）：为修复 roundtrip 而引入的斜率约束 [0.5,2] 使随机逆映射
  变温和，部分抵消保护 → 成功率升到 ≈42%，且不再随 n 下降。
  即：**恢复质量约束以抗攻击性为代价**（42% vs 18%），报告与加分项素材。

产出：results/exp2_reducing_attack.csv
      results/fig_exp2_fig8.png（Fig.8 复现 + 双抽样对比）
运行：E:/Miniconda/envs/speechguard/python.exe experiments/exp2_reducing_attack.py
"""
import argparse
import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from src.acoustic.warping import (  # noqa: E402
    DIST_RANGE,
    PI,
    distortion_strength,
    gen_params,
    multi_warp,
)

GRID = np.linspace(0.0, PI, 4001)  # 复合映射 dist 的积分网格


def param_pool(n: int, count: int, rng, sampling: str = "constrained") -> list:
    """预生成参数池（避免逐 trial 拒绝采样）。

    sampling='constrained'：本实现的构造法（斜率∈[0.5,2]）；
    sampling='raw'：论文原味——α、β 独立排序均匀 + dist 拒绝，无斜率约束。
    """
    pool, tries = [], 0
    max_tries = count * 60
    while len(pool) < count and tries < max_tries:
        tries += 1
        try:
            if sampling == "constrained":
                pool.append(gen_params(n=n, dist_range=DIST_RANGE, rng=rng))
            else:
                a = np.sort(rng.uniform(1e-3, PI - 1e-3, n))
                b = np.sort(rng.uniform(1e-3, PI - 1e-3, n))
                if np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
                    continue
                d = distortion_strength(a, b)
                if DIST_RANGE[0] <= d <= DIST_RANGE[1]:
                    pool.append((a, b, d))
        except RuntimeError:
            break
    return pool


def attack_trial(y_true_grid: np.ndarray, att_alpha, att_beta) -> float:
    """对真实映射在网格上的取值 y_true = m_true(GRID)，施加攻击者逆映射
    （m_att⁻¹(ω) = m_att(ω, β, α)），返回复合映射 dist(m_att⁻¹∘m_true)。"""
    comp = multi_warp(y_true_grid, att_beta, att_alpha)
    return distortion_strength(warp=lambda _: comp, grid=GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--ns-fixed", type=int, nargs="+",
                    default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(PROJ, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    def overall_rate(sampling):
        pool_true, pool_att = [], []
        for n in range(1, 11):
            pool_true += param_pool(n, args.trials // 10 + 10, rng, sampling)
        for n in range(1, 11):
            pool_att += param_pool(n, args.trials // 10 + 10, rng, sampling)
        succ = 0
        for _ in range(args.trials):
            ta, tb, d_true = pool_true[rng.integers(len(pool_true))]
            y_true = multi_warp(GRID, ta, tb)
            aa, ab, _ = pool_att[rng.integers(len(pool_att))]
            succ += attack_trial(y_true, aa, ab) < d_true
        return succ / args.trials

    # ---- 总成功率：两种抽样对比 ----
    rate_c = overall_rate("constrained")
    print(f"[总体][constrained] 成功率 {rate_c:.2%}（本实现）", flush=True)
    rows.append(["overall_constrained", args.trials, f"{rate_c:.4f}"])
    rate_r = overall_rate("raw")
    print(f"[总体][raw 论文原味] 成功率 {rate_r:.2%}（论文基准 15.87%）", flush=True)
    rows.append(["overall_raw", args.trials, f"{rate_r:.4f}"])

    # ---- 固定 n_true：两种抽样的成功率曲线 + constrained ECDF（论文 Fig.8）----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    cmap = plt.get_cmap("viridis")
    rates = {"constrained": [], "raw": []}
    for k, n_fix in enumerate(args.ns_fixed):
        for sampling in ("constrained", "raw"):
            pool_n = param_pool(n_fix, 150, rng, sampling)
            pool_att = []
            for n in range(1, 11):
                pool_att += param_pool(n, 60, rng, sampling)
            succ = 0
            n_tr = max(args.trials // 10, 1)
            for _ in range(n_tr):
                ta, tb, d_true = pool_n[rng.integers(len(pool_n))]
                y_true = multi_warp(GRID, ta, tb)
                aa, ab, _ = pool_att[rng.integers(len(pool_att))]
                succ += attack_trial(y_true, aa, ab) < d_true
            r = succ / n_tr
            rates[sampling].append(r)
            rows.append([f"n_true={n_fix}_{sampling}", n_tr, f"{r:.4f}"])
            print(f"[Fig.8][{sampling}] n_true={n_fix:2d}: 成功率 {r:.2%}", flush=True)
        if n_fix <= 10:
            pool_ecdf = param_pool(n_fix, 200, rng, "constrained")
            pool_att = []
            for n in range(1, 11):
                pool_att += param_pool(n, 60, rng, "constrained")
            d_all = []
            for _ in range(1000):
                ta, tb, _ = pool_ecdf[rng.integers(len(pool_ecdf))]
                y_true = multi_warp(GRID, ta, tb)
                aa, ab, _ = pool_att[rng.integers(len(pool_att))]
                d_all.append(attack_trial(y_true, aa, ab))
            xs = np.sort(d_all)
            c = cmap(k / max(len(args.ns_fixed) - 1, 1))
            ax1.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, label=f"n={n_fix}")

    ax1.set_xlabel("Distortion strength after reducing attacks")
    ax1.set_ylabel("ECDF")
    ax1.set_title("(a) ECDF of dist after attack (constrained)")
    ax1.legend(fontsize=7, ncol=2)
    ax2.plot(args.ns_fixed, rates["constrained"], "o-", label="constrained (ours)")
    ax2.plot(args.ns_fixed, rates["raw"], "s--", label="raw (paper-style)")
    ax2.set_xlabel("n (number of parameter pairs)")
    ax2.set_ylabel("Attack success rate")
    ax2.set_title("(b) Success rate vs n")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_exp2_fig8.png"), dpi=150)

    with open(os.path.join(out_dir, "exp2_reducing_attack.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "trials", "success_rate"])
        w.writerows(rows)
    print("saved:", os.path.join(out_dir, "fig_exp2_fig8.png"))
    print("saved:", os.path.join(out_dir, "exp2_reducing_attack.csv"))


if __name__ == "__main__":
    main()
