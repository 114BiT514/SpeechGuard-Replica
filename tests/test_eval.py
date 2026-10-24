"""评估工具单元测试：EER 计算的正确性（不加载模型）。"""
import numpy as np

from experiments.eval_asv import eer_from_scores

rng = np.random.default_rng(0)


def test_eer_perfect_separation():
    gen = np.ones(100) * 0.9
    imp = np.ones(100) * 0.1
    assert eer_from_scores(gen, imp) < 0.01


def test_eer_identical_distributions():
    gen = rng.normal(0.5, 0.1, 500)
    imp = rng.normal(0.5, 0.1, 500)
    eer = eer_from_scores(gen, imp)
    assert 0.4 < eer < 0.6


def test_eer_shifted_distributions():
    """gen 均值高 1σ：EER ≈ Φ(-0.5) ≈ 0.309。"""
    gen = rng.normal(0.6, 0.1, 4000)
    imp = rng.normal(0.5, 0.1, 4000)
    eer = eer_from_scores(gen, imp)
    assert 0.25 < eer < 0.37


def test_eer_monotone_in_shift():
    e1 = eer_from_scores(rng.normal(0.55, 0.1, 2000), rng.normal(0.5, 0.1, 2000))
    e2 = eer_from_scores(rng.normal(0.65, 0.1, 2000), rng.normal(0.5, 0.1, 2000))
    assert e2 < e1
