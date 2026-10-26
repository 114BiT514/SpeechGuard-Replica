"""[M5 任务2] 权限组数据结构：G = {G1..Gg}，每组资产子集 Mi = {Si, Wi}。

论文 §4.4：
- Si：声音参数集（warping 的 SegmentParams 档案，恢复声纹的密钥材料）；
- Wi：内容加密密钥集（每个敏感加密对象一把 AES 密钥）；
- 不同权限组凭各自属性集解出各自可见的 Mi → 听到/恢复不同程度的声纹与内容。

典型三组（执行计划 M5 场景，pipeline/前端直接消费）：
- admin（管理员）：Si + 全部 Wi → 完整恢复（听者=音频主人授权端）；
- collab（合作者）：仅 Si → 逆扭曲恢复声纹，但敏感词仍是加密态（听不出内容）；
- public（公开）：什么都不给 → 只有静音分发版 MP3（§4.3.3）。

资产包序列化：SegmentParams 含 numpy 数组，转 base64（JSON 安全）；
AES 密钥同样 base64。加密分发用 cp_abe.CPABE（方法等价实现见其文档）。
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

import numpy as np

from src.acoustic.warping import SegmentParams
from src.auth.cp_abe import CPABE


def _b64(v: bytes) -> str:
    return base64.b64encode(v).decode()


def _ub64(s: str) -> bytes:
    return base64.b64decode(s)


def seg_to_dict(seg: SegmentParams) -> dict:
    return {"frame_start": int(seg.frame_start), "frame_end": int(seg.frame_end),
            "alpha": _b64(seg.alpha.astype(np.float64).tobytes()),
            "beta": _b64(seg.beta.astype(np.float64).tobytes()),
            "n": int(seg.n), "dist": float(seg.dist)}


def seg_from_dict(d: dict) -> SegmentParams:
    n = d["n"]
    return SegmentParams(
        d["frame_start"], d["frame_end"],
        np.frombuffer(_ub64(d["alpha"]), dtype=np.float64).copy().reshape(n),
        np.frombuffer(_ub64(d["beta"]), dtype=np.float64).copy().reshape(n),
        d["dist"])


@dataclass
class AssetPackage:
    """Mi = {Si, Wi}——发给某权限组的全部资产（可子集）。"""
    voice: bool = False                      # 是否含 Si
    archive: list[SegmentParams] = field(default_factory=list)
    content_keys: list[bytes] = field(default_factory=list)   # Wi 子集

    def to_bytes(self) -> bytes:
        return json.dumps({
            "voice": self.voice,
            "archive": [seg_to_dict(s) for s in self.archive],
            "keys": [_b64(k) for k in self.content_keys],
        }, ensure_ascii=False).encode()

    @classmethod
    def from_bytes(cls, b: bytes) -> "AssetPackage":
        d = json.loads(b)
        return cls(voice=d["voice"],
                   archive=[seg_from_dict(s) for s in d["archive"]],
                   content_keys=[_ub64(k) for k in d["keys"]])


@dataclass
class PermissionGroup:
    name: str
    attributes: set[str]                 # KeyGen 给该组成员的属性
    policy_term: list[str]               # Encrypt 的合取项（通常 == attributes）
    sees: tuple[bool, bool]              # (Si?, Wi?) → 论文 "不同程度的恢复"


def typical_groups() -> list[PermissionGroup]:
    """执行计划 M5 的三组典型权限。"""
    return [
        PermissionGroup("admin", {"role_admin"}, ["role_admin"], (True, True)),
        PermissionGroup("collab", {"role_collab"}, ["role_collab"], (True, False)),
        PermissionGroup("public", set(), [], (False, False)),   # 不发放资产
    ]


class PermissionVault:
    """owner 侧：全量资产 (S, W) + 各组的 CP-ABE 密文；授权者凭属性解密。"""

    def __init__(self, abe: CPABE):
        self.abe = abe
        self.archive: list[SegmentParams] = []
        self.keys: list[bytes] = []
        self.ciphertexts: dict[str, str] = {}   # group name -> CT

    def register(self, archive: list[SegmentParams], keys: list[bytes]) -> None:
        self.archive, self.keys = archive, keys

    def grant(self, groups: list[PermissionGroup]) -> dict[str, dict[str, bytes]]:
        """给每个可见组发 sk 并生成该组 Mi 的密文。返回 {group: sk}。"""
        sks: dict[str, dict[str, bytes]] = {}
        for g in groups:
            voice = g.sees[0]
            if not (voice or g.sees[1]):
                continue
            pkg = AssetPackage(voice=voice,
                               archive=self.archive if voice else [],
                               content_keys=list(self.keys) if g.sees[1] else [])
            self.ciphertexts[g.name] = self.abe.encrypt([g.policy_term], pkg.to_bytes())
            sks[g.name] = self.abe.keygen(g.attributes)
        return sks

    def to_json(self) -> str:
        return json.dumps({"dom": self.abe.pub, "ct": self.ciphertexts})

    @classmethod
    def from_json(cls, s: str, abe: CPABE) -> "PermissionVault":
        d = json.loads(s)
        v = cls(abe)
        v.ciphertexts = d["ct"]
        return v

    def open(self, sk: dict[str, bytes], group: str) -> AssetPackage:
        """listener/recover 侧：组内成员用 sk 解出 Mi={Si,Wi}。"""
        return AssetPackage.from_bytes(self.abe.decrypt(sk, self.ciphertexts[group]))
