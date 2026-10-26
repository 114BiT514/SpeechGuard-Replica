"""[M4 DoD] 加密前后 MP3 样例生成。

选一条含 NER 实体的 LibriSpeech 语音 → 128kbps MP3 → 对敏感词帧区间：
  mp3_demo_original.mp3   原始
  mp3_demo_encrypted.mp3  加密帧体（有密钥方可解回；常见播放器可播，加密段噪声）
  mp3_demo_quiet.mp3      无权限分发版（加密段替换静音，论文 §4.3.3）

运行：E:/Miniconda/envs/speechguard/python.exe scripts/m4_mp3_demo.py
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.eval_asr import librispeech_ref  # noqa: E402
from src.content.annotation import Annotation  # noqa: E402
from src.content.detect import detect_sensitive  # noqa: E402
from src.content.mp3_crypto import (  # noqa: E402
    encode_mp3, parse, quiet_mp3, encrypt_mp3, decrypt_mp3,
    time_spans_to_frames)
from src.content.transcribe import transcribe_with_times  # noqa: E402
from src.content.encrypt import gen_key  # noqa: E402

SR = 16000


def main():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/dev-clean/*/*/*.flac")))
    wav = ctc = hits = path = None
    for f in files:
        w, _ = sf.read(f)
        w = w.astype(np.float32)
        c = transcribe_with_times(w, device="cuda:0")
        h = detect_sensitive(c)
        if len(h) >= 2 and any(s.startswith("ner:") for hi in h for s in hi.sources):
            wav, ctc, hits, path = w, c, h, f
            break
    assert wav is not None, "no suitable utterance"
    print(f"utt: {os.path.basename(path)}  ref: {librispeech_ref(path)}")
    print(f"敏感项: {[(h.word, h.sources[0]) for h in hits]}")

    ann = Annotation.from_ctc(path, ctc)
    groups, cur = [], [hits[0].idx]
    for h in hits[1:]:
        if h.idx == cur[-1] + 1:
            cur.append(h.idx)
        else:
            groups.append(cur)
            cur = [h.idx]
    groups.append(cur)
    by_idx = {h.idx: h for h in hits}
    for g in groups:
        ann.mark_words(g, by_idx[g[0]].sources[0])

    outdir = os.path.join(PROJ, "assets")
    tmp_wav = os.path.join(outdir, "_demo16k.wav")
    sf.write(tmp_wav, wav, SR)
    orig = os.path.join(outdir, "mp3_demo_original.mp3")
    encode_mp3(tmp_wav, orig)
    os.remove(tmp_wav)
    data = open(orig, "rb").read()
    doc = parse(data)

    # 36ms 帧区间 → 秒 → 公式(5) MP3 帧号
    spans_sec = [(fs * 576 / SR, (fe + 1) * 576 / SR)
                 for fs, fe in ann.merge_spans(margin_frames=8)]
    ranges = time_spans_to_frames(doc, spans_sec)
    keys = [gen_key() for _ in ranges]
    enc = encrypt_mp3(data, ranges, keys)
    open(os.path.join(outdir, "mp3_demo_encrypted.mp3"), "wb").write(enc)
    open(os.path.join(outdir, "mp3_demo_quiet.mp3"), "wb").write(quiet_mp3(data, ranges))
    assert decrypt_mp3(enc, ranges, keys) == data, "MP3 解密还原失败"
    print(f"加密对象: {len(ranges)} 组 / MP3 帧 {ranges}")
    print("已生成: assets/mp3_demo_{original,encrypted,quiet}.mp3")


if __name__ == "__main__":
    main()
