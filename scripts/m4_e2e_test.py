"""[M4 集成测试] 端到端内容保护验证（LibriSpeech 真实语音，不等自录集）。

流程：参考文本 NER 找实体 → forced_align 词级时间戳 → 实体帧区间 AES-CTR 加密
→ 重新转写算 FNR（论文 §5.1：加密后 ASR 仍识别出的敏感词比例，越低越私密）
→ 解密还原逐字节一致。

运行：E:/Miniconda/envs/speechguard/python.exe scripts/m4_e2e_test.py
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from experiments.eval_asr import librispeech_ref, normalize  # noqa: E402
from src.content.align import align_words  # noqa: E402
from src.content.encrypt import decrypt_spans, encrypt_spans, gen_key  # noqa: E402
from src.content.ner import extract_entities  # noqa: E402
from src.content.sensitive_lists import SensitiveLists  # noqa: E402
from src.content.transcribe import transcribe  # noqa: E402

SR = 16000
FRAME_LEN = 576
MARGIN_FRAMES = 8   # 加密区间两侧余量（288ms：覆盖协同发音渗漏；实测 ±4 帧时
                    # 短词 US 仍被上下文脑补，±8 帧完全抹除——代价是邻近词损伤）


def main():
    files = sorted(glob.glob(os.path.join(
        PROJ, "data/librispeech/LibriSpeech/dev-clean/*/*/*.flac")))
    # 选含 NER 实体的语音（前 8 条里找）
    chosen = []
    for f in files:
        ref = librispeech_ref(f)
        ents = extract_entities(ref, device="cpu")
        if ents:
            chosen.append((f, ref, ents))
        if len(chosen) >= 6:
            break
    print(f"含实体语音 {len(chosen)} 条", flush=True)

    sl = SensitiveLists.build()
    fnr_hits, fnr_total = 0, 0
    for f, ref, ents in chosen:
        wav, sr = sf.read(f)
        wav = wav.astype(np.float64)
        word_times = {w: (t0, t1) for w, t0, t1 in align_words(wav, ref)}
        spans = []
        for ent, lab, s0, e0 in ents:
            for w in ent.split():
                if w in word_times:
                    t0, t1 = word_times[w]
                    fs = max(int(np.floor(t0 * SR / FRAME_LEN)) - MARGIN_FRAMES, 0)
                    fe = int(np.ceil(t1 * SR / FRAME_LEN)) + MARGIN_FRAMES
                    spans.append((fs, fe))
        # 合并重叠区间（同一密钥段）
        spans.sort()
        merged = []
        for fs, fe in spans:
            if merged and fs <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], fe))
            else:
                merged.append((fs, fe))
        spans = merged
        keys = [gen_key() for _ in spans]
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        enc = encrypt_spans(pcm, spans, keys)
        dec = decrypt_spans(enc, spans, keys)
        assert np.array_equal(dec, pcm), "解密还原失败"
        # FNR：加密后 ASR 是否仍认出实体词
        hyp = transcribe([enc.astype(np.float64) / 32767.0])[0]
        for ent, lab, _, _ in ents:
            for w in ent.split():
                fnr_total += 1
                still = w in hyp.split()
                fnr_hits += still
        n_frames = int(np.ceil(len(pcm) / FRAME_LEN))
        cov = sum(fe - fs + 1 for fs, fe in spans) / n_frames
        print(f"  {os.path.basename(f)}: 实体 {[e[0] for e in ents]} "
              f"帧覆盖 {cov:.1%} 解密✓", flush=True)

    # FNR = ASR 认不出的敏感词比例（论文 Table 7：Ori≈0.6%，Enc≈95.9%，Dec≈0.6%）
    fnr_miss = 1.0 - fnr_hits / max(fnr_total, 1)
    print(f"\nEnc FNR（加密后敏感词 ASR 识别率）= {fnr_hits/max(fnr_total,1):.2%} "
          f"({fnr_hits}/{fnr_total}) → 论文口径 FNR(认不出)={fnr_miss:.2%}（论文 Enc 95.88%）")


if __name__ == "__main__":
    main()
