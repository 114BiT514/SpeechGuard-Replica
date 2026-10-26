"""M4 任务6 单测：MP3 帧体加密（公式(5)）/解密还原/静音分发版可播放性。

需要 ffmpeg（环境内已装）。测试合成一段 3s 音频：0-0.5s 正弦（前段）、
0.5-1.2s 高频音（"敏感词"段）、1.2-3s 正弦（尾段）。
"""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from src.content import mp3_crypto as mc
from src.content.encrypt import gen_key

SR_MP3 = 44100

FFMPEG = (shutil.which("ffmpeg")
          or r"E:/Miniconda/envs/speechguard/Library/bin/ffmpeg.exe")


def _decode(data: bytes, tmp_path) -> np.ndarray:
    p = tmp_path / "dec.mp3"
    p.write_bytes(data)
    r = subprocess.run([FFMPEG, "-v", "error", "-i", str(p),
                        "-f", "f32le", "-ac", "1", "-"],
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return np.frombuffer(r.stdout, np.float32)


@pytest.fixture(scope="module")
def mp3_case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mp3")
    t = np.arange(16000 * 3) / 16000
    wav = 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    wav[8000:19200] = 0.3 * np.sin(2 * np.pi * 800 * t[8000:19200])
    p16 = tmp / "t.wav"
    sf.write(str(p16), wav, 16000)
    out = tmp / "t.mp3"
    mc.encode_mp3(str(p16), str(out))
    data = out.read_bytes()
    doc = mc.parse(data)
    return data, doc, tmp


def test_parse_full_coverage(mp3_case):
    data, doc, _ = mp3_case
    assert len(doc.frames) > 50
    assert doc.frames[-1].off + doc.frames[-1].size == len(data) - doc.tag_v1_bytes
    assert abs(doc.frame_times()[-1][1] - 3.0) < 0.1


def test_formula5_span(mp3_case):
    _data, doc, _ = mp3_case
    Td = doc.frames[0].dur
    (fs, fe), = mc.time_spans_to_frames(doc, [(0.5, 1.2)])
    assert fs == int(np.floor(0.5 / Td))
    assert int(np.ceil(1.2 / Td)) - 1 == fe
    t0, _ = doc.frame_times()[fs]
    _, t1 = doc.frame_times()[fe]
    assert t0 <= 0.5 + 1e-6 and t1 >= 1.2 - 1e-6   # 闭区间覆盖整个词


def test_encrypt_structure_and_roundtrip(mp3_case):
    data, doc, _ = mp3_case
    r = mc.time_spans_to_frames(doc, [(0.5, 1.2)])
    keys = [gen_key()]
    enc = mc.encrypt_mp3(data, r, keys)
    assert len(enc) == len(data)                       # 帧结构不变
    fs, fe = r[0]
    for i, f in enumerate(doc.frames):
        hdr = enc[f.off:f.off + f.hdr_len] == data[f.off:f.off + f.hdr_len]
        body_enc = enc[f.body_off:f.off + f.size] != data[f.body_off:f.off + f.size]
        if fs <= i <= fe:
            assert hdr and body_enc                     # 帧头保留、帧体加密
        else:
            assert enc[f.off:f.off + f.size] == data[f.off:f.off + f.size]
    assert mc.decrypt_mp3(enc, r, keys) == data         # 逐字节还原
    assert mc.decrypt_mp3(enc, r, [gen_key()]) != data  # 错密钥解不回


def test_encrypted_front_plays(mp3_case, tmp_path):
    """加密版第三方播放器可解码，加密段之前的音频逐样本无损。"""
    data, doc, _ = mp3_case
    r = mc.time_spans_to_frames(doc, [(0.5, 1.2)])
    d0 = _decode(data, tmp_path)
    d1 = _decode(mc.encrypt_mp3(data, r, [gen_key()]), tmp_path)
    n = int(0.4 * SR_MP3)                                # 加密窗之前的样本
    assert len(d0) > n and len(d1) > n
    np.testing.assert_allclose(d1[:n], d0[:n], atol=0.0)


def test_quiet_distribution_lossless(mp3_case, tmp_path):
    """无权限静音分发版：时长保持、存在一段与加密窗等长的真静音、
    静音段之前/之后（去边缘）逐样本无损。

    注：字节时间轴与解码流时间轴之间有解码器延迟/padding 固定偏移，
    且相邻帧重叠窗(~576样本)+比特池使边缘数百样本受扰——静音边界
    直接在解码流上测量，不依赖帧号换算。
    """
    data, doc, _ = mp3_case
    r = mc.time_spans_to_frames(doc, [(0.5, 1.2)])
    d0 = _decode(data, tmp_path)
    d2 = _decode(mc.quiet_mp3(data, r), tmp_path)
    assert abs(len(d2) - len(d0)) < 0.05 * SR_MP3        # 时长不漂移

    near_zero = np.abs(d2) < 1e-4
    idx = np.nonzero(~near_zero)[0]                    # 非静音样本位置
    gaps = np.diff(idx)                                 # 相邻非静音样本间的静音间隙
    k = int(np.argmax(gaps))
    a = int(idx[k])                                     # 静音段前一非静音样本
    b = int(idx[k + 1])                                 # 静音段后一非静音样本
    assert (b - a) / SR_MP3 > 0.05, "无显著静音段"
    seg_len = (b - a - 1) / SR_MP3
    win = doc.frame_times()[r[0][1]][1] - doc.frame_times()[r[0][0]][0]
    assert abs(seg_len - win) < 0.1, f"silent seg {seg_len:.3f}s vs window {win:.3f}s"

    edge = 3200                                         # overlap/reservoir 实测扰动 ~2.2k 样本
    tail0 = b + edge
    tail = min(len(d0), len(d2))
    assert a > edge and tail0 < tail - 2000
    np.testing.assert_allclose(d2[a - edge:a - 2200], d0[a - edge:a - 2200], atol=1e-6)
    np.testing.assert_allclose(d2[tail0:tail], d0[tail0:tail], atol=1e-6)
