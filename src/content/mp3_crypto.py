"""[M4 任务6] MP3 有损格式适配（论文 §4.3.3，公式(5)）。

前提：AES-CTR 是流密码，密文与明文等长 → 帧结构不变。本模块在 **MP3 文件字节层**
直接加密"帧体"：

- 公式(5)：敏感词时间戳 (ts, te) 换算 MP3 帧号  fs=⌊ts/Td⌋, fe=⌈te/Td⌉，
  Td = 每帧样本数/采样率（MPEG-1 L3 = 1152/sr ≈ 26.1ms @44.1k；MPEG-2/2.5 L3 = 576/sr）。
- **跳过**：ID3v2 tag、帧头（含 CRC）——这些字节不动，播放器照常解码、时长不变；
- **只加密**：帧体 → 无密钥者听到噪声（论文："harsher sound"）；
- **无权限监听**：把加密帧帧体替换为静音帧（同长零填充，对应论文简易 decoder 行为）。

解析策略：顺序解析（找到首帧后按 framesize 逐帧推进），不重新扫同步字——
帧体内容（包括加密后的伪随机字节）不会被误认成帧边界。

实测行为（ffmpeg 解码验证，tests/test_mp3_crypto.py 固化）：
- quiet 分发版：时长保持、加密窗外 PCM 与原文件逐样本一致、窗内纯静音——
  第三方播放器完全无损可播；
- encrypt 版：加密帧的密文大概率不满足 Huffman 约束，解码器整帧丢弃 →
  加密段之后的音频在第三方播放器里前移（前端区间仍逐样本一致）。论文为此
  自带简易 decoder；无密钥的对外分发应使用 quiet 版，持有密钥的应用先解密再播。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from Crypto.Cipher import AES

# 帧头字段表（ISO/IEC 11172-3 / 13818-3，Layer III）
_BITRATE_MPEG1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BITRATE_MPEG2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
# ver -> (采样率表, 每帧样本数)
_VER = {
    0b11: ([44100, 48000, 32000], 1152),   # MPEG-1
    0b10: ([22050, 24000, 16000], 576),    # MPEG-2
    0b00: ([11025, 12000, 8000], 576),     # MPEG-2.5
}


@dataclass
class Mp3Frame:
    off: int       # 帧起点（字节）
    hdr_len: int   # 帧头长度（4，含 CRC 时 6）
    size: int      # 整帧长度
    samples: int   # 每帧样本数
    sr: int        # 采样率

    @property
    def body_off(self) -> int:
        return self.off + self.hdr_len

    @property
    def body_len(self) -> int:
        return self.size - self.hdr_len

    @property
    def dur(self) -> float:
        return self.samples / self.sr


@dataclass
class Mp3Doc:
    data: bytes
    frames: list[Mp3Frame]   # 数据帧（不含 tag）
    tag_bytes: int           # ID3v2 前缀长度
    tag_v1_bytes: int        # ID3v1 后缀长度（128 或 0）

    def frame_times(self) -> list[tuple[float, float]]:
        """每个数据帧的时间区间 (t0, t1) 秒（按帧时长顺序累加）。"""
        out, acc = [], 0.0
        for f in self.frames:
            out.append((acc, acc + f.dur))
            acc += f.dur
        return out


def parse(data: bytes) -> Mp3Doc:
    """解析 MP3 → Mp3Doc。支持前置 ID3v2 / 后置 ID3v1，tag 之后字节须构成连续帧。"""
    n = len(data)
    tag2 = 0
    if data[:3] == b"ID3" and n >= 10:
        size = ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 |
                (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
        tag2 = 10 + size
    tag1 = 128 if n >= 128 and data[-3:] == b"TAG" else 0

    frames: list[Mp3Frame] = []
    i, body_end = tag2, n - tag1
    while i < body_end - 3:
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            raise ValueError(f"MP3 parse: bad sync at byte {i}")
        ver = (data[i + 1] >> 3) & 0x03      # 11=MPEG1 10=MPEG2 00=MPEG2.5 01=保留
        layer = (data[i + 1] >> 1) & 0x03    # 01=Layer3
        prot = data[i + 1] & 0x01            # 1=无CRC
        if layer != 0x01 or ver not in _VER:
            raise ValueError(f"MP3 parse: unsupported layer/version at {i}")
        br_i = (data[i + 2] >> 4) & 0x0F
        sr_i = (data[i + 2] >> 2) & 0x03
        pad = (data[i + 2] >> 1) & 0x01
        if sr_i == 3 or br_i in (0, 15):
            raise ValueError(f"MP3 parse: bad bitrate/sr index at {i}")
        sr_tab, n_per_frame = _VER[ver]
        sr = sr_tab[sr_i]
        kbps = (_BITRATE_MPEG1_L3 if ver == 0b11 else _BITRATE_MPEG2_L3)[br_i]
        size = n_per_frame // 8 * kbps * 1000 // sr + pad   # Layer III 填充单位=1字节
        hdr = 4 + (0 if prot else 2)
        if size <= hdr or i + size > body_end:
            raise ValueError(f"MP3 parse: frame at {i} size {size} invalid")
        frames.append(Mp3Frame(i, hdr, size, n_per_frame, sr))
        i += size
    return Mp3Doc(data, frames, tag2, tag1)


def time_spans_to_frames(doc: Mp3Doc,
                         spans: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """公式(5)：[(ts, te) 秒] → [(fs, fe) 数据帧号闭区间]，fs=⌊ts/Td⌋，fe=⌈te/Td⌉-1。"""
    if not doc.frames:
        raise ValueError("MP3 has no data frames")
    Td = doc.frames[0].dur
    last = len(doc.frames) - 1
    out = []
    for ts, te in spans:
        fs = min(max(int(np.floor(ts / Td)), 0), last)
        fe = min(max(int(np.ceil(te / Td)) - 1, fs), last)
        out.append((fs, fe))
    return out


def _ctr(key: bytes, obj_index: int, data: bytes, decrypt: bool = False) -> bytes:
    """与 encrypt.py 相同的 nonce 约定：8 字节对象序号 + 8 字节零起始计数器。"""
    nonce = obj_index.to_bytes(8, "big")
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce, initial_value=b"\0" * 8)
    return cipher.decrypt(data) if decrypt else cipher.encrypt(data)


def _apply(doc: Mp3Doc, ranges: list[tuple[int, int]], keys: list[bytes],
           mode: str) -> bytes:
    """mode: encrypt / decrypt / quiet。帧头与 tag 永不动。

    跨多帧的区间：帧头穿插在帧体之间，须**只取帧体拼接**做 CTR 运算再写回，
    否则中间帧的帧头会被一并加密 → 结构损坏。
    """
    buf = bytearray(doc.data)
    for j, (fs, fe) in enumerate(ranges):
        slices = [(doc.frames[i].body_off, doc.frames[i].body_off + doc.frames[i].body_len)
                  for i in range(fs, fe + 1)]
        if mode == "quiet":
            for a, b in slices:
                buf[a:b] = bytes(b - a)
            continue
        plain = b"".join(bytes(buf[a:b]) for a, b in slices)
        res = _ctr(keys[j % len(keys)], j, plain, decrypt=(mode == "decrypt"))
        off = 0
        for a, b in slices:
            buf[a:b] = res[off:off + (b - a)]
            off += b - a
    return bytes(buf)


def encrypt_mp3(data: bytes, ranges: list[tuple[int, int]],
                keys: list[bytes]) -> bytes:
    return _apply(parse(data), ranges, keys, "encrypt")


def decrypt_mp3(data: bytes, ranges: list[tuple[int, int]],
                keys: list[bytes]) -> bytes:
    return _apply(parse(data), ranges, keys, "decrypt")


def quiet_mp3(data: bytes, ranges: list[tuple[int, int]]) -> bytes:
    """无权限分发版：加密帧帧体替换为静音帧（保留帧头 → 播放时长/结构不变）。"""
    return _apply(parse(data), ranges, [], "quiet")


def encode_mp3(wav_path: str, out_path: str, kbps: int = 128, sr: int = 44100,
               ffmpeg: str | None = None) -> str:
    """16k 单声道 WAV → MP3（publish 阶段用）。"""
    ff = ffmpeg or shutil.which("ffmpeg") or \
        r"E:/Miniconda/envs/speechguard/Library/bin/ffmpeg.exe"
    subprocess.run([ff, "-y", "-v", "error", "-i", wav_path,
                    "-ar", str(sr), "-ac", "1", "-b:a", f"{kbps}k", out_path],
                   check=True)
    return out_path
