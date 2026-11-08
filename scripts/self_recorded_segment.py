"""[Volunteers 列前置] 自录集逐句切分入库：8 人连续朗读 → utterance 粒度 WAV + manifest。

动机：论文 volunteers 集是 18 人/1300 条 ~8s utterance；我们的录音是每人一份连续朗读
（A30 短句 + B3 段落 + C20 敏感句 + D10 数字句）。Table 5/7 的评估协议以 utterance
为单位（EER 配对、WER 逐条、加密按条），必须先切句。

切分依据：scripts/self_recorded_quality_eval.py 的 _sentence_windows——
边界=上句 t1 与下句 t0 的中点吸附静音段（录音规范句间停 1-2s），段末句用"t1 后
第一段静音中心"封顶；该窗口逻辑已在预审/加分项里经过 person4 对照校准。

分组标签（进 manifest，供分层报告）：
  native   person8（母语者）
  colloquial person6（口语组：自然连读缩约，双模型 WER 高但人评口语最佳）
  clear    person1/2/3/4/5/7（清晰非母语）

输出：
  data/self_recorded/processed/utterances/{person}/{sid}.wav   16k 16bit mono
  data/self_recorded/processed/volunteers_manifest.json        元数据（含参考文本）

运行（纯 CPU，秒级）：
  E:/Miniconda/envs/speechguard/python.exe scripts/self_recorded_segment.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.self_recorded_quality_eval import (  # noqa: E402
    REF, SR, _sentence_windows, evaluate)

SELF = os.path.join(PROJ, "data", "self_recorded")
OUT = os.path.join(SELF, "processed", "utterances")
MANIFEST = os.path.join(SELF, "processed", "volunteers_manifest.json")

GROUP = {"person8": "native", "person9": "native", "person10": "native",
         "person11": "native", "person6": "colloquial"}


def main():
    people = sorted(f[:-4] for f in os.listdir(os.path.join(SELF, "processed", "full16k"))
                    if f.endswith(".wav"))
    ref_text = {sid: t for _p, sid, t in REF}
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in people:
        r = evaluate(p, "cpu")                      # CTC 词时间戳走缓存，不碰 GPU
        wins = _sentence_windows(r["sentences"], r["_wav"], r["signal"]["noise_floor_db"])
        pdir = os.path.join(OUT, p)
        os.makedirs(pdir, exist_ok=True)
        n_cut = 0
        for part, sid, _t in REF:
            if sid not in wins:
                continue
            lo, hi = wins[sid]
            if hi - lo < 1.0:
                print(f"  跳过 {p}/{sid}（窗口 {hi - lo:.1f}s 过短）")
                continue
            wav = r["_wav"][max(int(lo * SR), 0):int(hi * SR)]
            if len(wav) < SR:                       # 越界/空窗保护
                print(f"  跳过 {p}/{sid}（有效音频 {len(wav) / SR:.1f}s 过短）")
                continue
            peak = float(np.abs(wav).max())
            if peak > 0.999:                        # 防切分后极小越界
                wav = wav / peak * 0.999
            sf.write(os.path.join(pdir, f"{sid}.wav"), wav, SR)
            rows.append({"utt": f"{p}_{sid}", "person": p, "sid": sid,
                         "part": part, "group": GROUP.get(p, "clear"),
                         "ref": ref_text[sid],
                         "t0": round(lo, 2), "t1": round(hi, 2),
                         "dur_s": round(len(wav) / SR, 2)})
            n_cut += 1
        print(f"{p}: {n_cut} 句 / {sum(x['dur_s'] for x in rows if x['person'] == p):.0f}s",
              flush=True)
    json.dump(rows, open(MANIFEST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    acd = [x for x in rows if x["part"] in "ACD"]
    print(f"\nmanifest: {len(rows)} 条（A/C/D 主协议 {len(acd)} 条 / "
          f"{len({x['person'] for x in acd})} 人 / "
          f"{sum(x['dur_s'] for x in acd) / 60:.1f} min）-> {MANIFEST}")


if __name__ == "__main__":
    main()
