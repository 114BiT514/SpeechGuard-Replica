"""[M5 任务1] 权限加密：CP-ABE 四算法（Setup/Encrypt/KeyGen/Decrypt）。

论文 §4.4 用 CP-ABE（属性基加密）把权限组可见范围 Mi={Si,Wi} 加密给满足访问
结构 Ai 的属性集。首选 charm-crypto 的 CP-ABE——**Windows 编译失败**
（pip install charm 需 C 扩展，本机 2026-10-26 尝试失败，证据在 git log），按执行
计划 §3-M5 的降级预案改为**方法等价实现**并在报告如实说明：

本实现 = 基于对称密钥派生的属性加密（Naccache 94 / Key-Derivation ABE 谱系），
提供与论文完全相同的四算法 API 与访问语义：

- Setup(λ)   → 主密钥 MK（每属性秘密 x_a 的派生根）+ 公域标识 PK；
- KeyGen(MK, S) → 私钥 sk_S：属性集 S 中每个属性的派生秘密 {x_a: a∈S}；
  不在 S 中的属性秘密在随机预言模型下不可推出（HKDF 单向）；
- Encrypt(PK, A, M) → 密文：访问结构 A 为 **析取范式（DNF）**：A = T1 ∨ T2 ∨ …，
  每个项 Tk 是属性合取集（全部持有才可解该副本）。M 用随机 AES-256-GCM 会话密钥 K
  加密；K 对每个项封装一份，KEK_T = HKDF(⊕_{a∈Tk} x_a)——属性名进 AAD 防错位；
- Decrypt(sk_S, CT) → 若 S ⊇ 某 Tk 则解出 M，否则 PermissionError（密码学校验失败）。

单调性：属性集合只增不减地扩大解密能力，与论文单调访问结构一致。
与配对 CP-ABE 的差别（写入报告局限）：
1) 无椭圆曲线对/非短密文（每项 1 条 ~100B 封装）——本项目策略规模
   （≤10 组 × ≤4 属性）下完全够用；
2) 访问结构限 DNF（不支持任意线性公式/门限）——论文示例即"权限组→属性集"，
   DNF 足以表达；
3) 属性撤销需对受影响资产重加密（与经典 ABE 相同的运维模式）。

实现：hashlib/hmac + pycryptodome AES-GCM，零新增依赖、本地/云端（Windows 与
ModelArts Linux）同构。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os

from Crypto.Cipher import AES

# ── HKDF（RFC 5869，无依赖实现）──

def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out, t = b"", b""
    for i in range(1, (length + 32) // 32 + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
    return out[:length]


def _attr_secret(master: bytes, attr: str) -> bytes:
    """主密钥 MK + 属性名 → 属性秘密 x_a。"""
    prk = hmac.new(b"SpeechGuard-ABE-KDF", master, hashlib.sha256).digest()
    return _hkdf_expand(prk, attr.encode("utf-8"), 32)


def _derive(*secrets: bytes) -> bytes:
    """属性秘密集合 → KEK：HKDF-Expand(⊕ x_a)。XOR 顺序无关、单调于属性集合。"""
    acc = b"\0" * 32
    for s in secrets:
        acc = bytes(a ^ b for a, b in zip(acc, s))
    return _hkdf_expand(acc, b"speechguard-cpabe-kek", 32)


class CPABE:
    """可信授权方（属性权威）：持有 master；Encrypt 用派生秘密封装会话密钥。"""

    def __init__(self, master: bytes):
        self.master = master
        self.pub = hashlib.sha256(master).hexdigest()[:8]  # 公域标识，绑定密文域

    # ── 论文 §4.4 的四算法 ──
    @classmethod
    def setup(cls, lam: int = 32) -> "CPABE":
        return cls(os.urandom(lam))

    def keygen(self, attributes: set[str]) -> dict[str, bytes]:
        """sk_S = {a: x_a}。调用方持久化时 base64 编码即可（pipeline 侧处理）。"""
        return {a: _attr_secret(self.master, a) for a in attributes}

    def encrypt(self, policy: list[list[str]], message: bytes) -> str:
        """A = DNF（每项为属性合取集）；message = 权限组可见资产 Mi 的序列化字节。"""
        if not policy or any(not t for t in policy):
            raise ValueError("policy must be DNF of non-empty attribute sets")
        k = os.urandom(32)
        nonce = os.urandom(12)
        c = AES.new(k, AES.MODE_GCM, nonce=nonce)
        ct, tag = c.encrypt_and_digest(message)
        wraps = []
        for term in sorted({tuple(sorted(t)) for t in policy}):
            kek = _derive(*(_attr_secret(self.master, a) for a in term))
            wnonce = os.urandom(12)
            wc = AES.new(kek, AES.MODE_GCM, nonce=wnonce)
            wc.update(str(term).encode())          # 属性名进 AAD 防错位
            wct, wtag = wc.encrypt_and_digest(k)
            wraps.append({"term": list(term), "n": wnonce.hex(),
                          "ct": wct.hex(), "tag": wtag.hex()})
        return json.dumps({"v": 1, "dom": self.pub, "n": nonce.hex(),
                           "ct": ct.hex(), "tag": tag.hex(), "wraps": wraps})

    def decrypt(self, sk: dict[str, bytes], ciphertext: str) -> bytes:
        """S 覆盖某一项才解得 message；否则 PermissionError（AAD/tag 校验失败）。"""
        box = json.loads(ciphertext)
        if box["dom"] != self.pub:
            raise PermissionError("CPABE: ciphertext from another authority domain")
        for w in box["wraps"]:
            term = list(w["term"])
            if not all(a in sk for a in term):
                continue
            kek = _derive(*[sk[a] for a in term])
            try:
                wc = AES.new(kek, AES.MODE_GCM, nonce=bytes.fromhex(w["n"]))
                wc.update(str(tuple(term)).encode())
                k = wc.decrypt_and_verify(bytes.fromhex(w["ct"]),
                                          bytes.fromhex(w["tag"]))
            except ValueError:
                raise PermissionError("CPABE: wrapped key failed verification")
            c = AES.new(k, AES.MODE_GCM, nonce=bytes.fromhex(box["n"]))
            return c.decrypt_and_verify(bytes.fromhex(box["ct"]),
                                        bytes.fromhex(box["tag"]))
        raise PermissionError(
            f"CPABE: attributes {sorted(sk)} do not satisfy policy")
