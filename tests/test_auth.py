"""M5 权限层单测：CP-ABE 等价实现的访问语义 + 资产序列化 + 三组典型权限。"""
import numpy as np
import pytest

from src.acoustic.warping import SegmentParams
from src.auth.cp_abe import CPABE
from src.auth.permission import (AssetPackage, PermissionVault, seg_from_dict,
                                 seg_to_dict, typical_groups)


def test_cp_abe_satisfy_and_deny():
    abe = CPABE.setup()
    msg = "Mi={Si,Wi} 资产包".encode()
    ct = abe.encrypt([["role_admin"]], msg)
    sk_admin = abe.keygen({"role_admin"})
    assert abe.decrypt(sk_admin, ct) == msg
    sk_other = abe.keygen({"role_collab"})
    with pytest.raises(PermissionError):
        abe.decrypt(sk_other, ct)
    with pytest.raises(PermissionError):
        abe.decrypt({}, ct)


def test_cp_abe_dnf_and_conjunction():
    abe = CPABE.setup()
    msg = b"assets"
    ct = abe.encrypt([["a", "b"], ["c"]], msg)      # (a∧b) ∨ c
    assert abe.decrypt(abe.keygen({"a", "b"}), ct) == msg
    assert abe.decrypt(abe.keygen({"c"}), ct) == msg
    assert abe.decrypt(abe.keygen({"c", "d"}), ct) == msg       # 超集仍可解（单调）
    with pytest.raises(PermissionError):                          # 半条路不行
        abe.decrypt(abe.keygen({"a"}), ct)
    with pytest.raises(PermissionError):
        abe.decrypt(abe.keygen({"d", "e"}), ct)


def test_cp_abe_tamper_detection():
    abe = CPABE.setup()
    ct = abe.encrypt([["x"]], b"secret")
    import json
    d = json.loads(ct)
    d["wraps"][0]["term"] = ["y"]                  # 篡改策略项
    sk = abe.keygen({"x", "y"})
    with pytest.raises(PermissionError):           # AAD 绑定 → 校验失败
        CPABE(abe.master).decrypt(sk, json.dumps(d))


def test_segment_serialization_exact():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0.1, 3.0, 5))
    b = np.sort(rng.uniform(0.1, 3.0, 5))
    seg = SegmentParams(10, 42, a, b, 0.57)
    back = seg_from_dict(seg_to_dict(seg))
    np.testing.assert_array_equal(back.alpha, seg.alpha)
    np.testing.assert_array_equal(back.beta, seg.beta)
    assert (back.frame_start, back.frame_end, back.dist) == (10, 42, 0.57)


def test_typical_three_groups():
    abe = CPABE.setup()
    vault = PermissionVault(abe)
    rng = np.random.default_rng(1)
    archive = [SegmentParams(0, 20, np.sort(rng.uniform(0.2, 2.9, 3)),
                             np.sort(rng.uniform(0.2, 2.9, 3)), 0.55)]
    keys = [b"\x11" * 16, b"\x22" * 16]
    vault.register(archive, keys)
    sks = vault.grant(typical_groups())
    assert set(sks) == {"admin", "collab"}          # public 不发资产

    admin = vault.open(sks["admin"], "admin")
    assert admin.voice and len(admin.content_keys) == 2
    collab = vault.open(sks["collab"], "collab")
    assert collab.voice and collab.content_keys == []
    with pytest.raises(PermissionError):
        vault.open(sks["collab"], "admin")          # collab 打不开 admin 的包

    # vault JSON 迁移到另一实例：持有同 master 的授权方仍可用
    v2 = PermissionVault.from_json(vault.to_json(), CPABE(abe.master))
    assert v2.open(sks["admin"], "admin").content_keys == keys
