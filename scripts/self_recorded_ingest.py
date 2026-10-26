"""[M4] 自录集入库：格式统一 + 盘点 + 转写辅助（内容审核用）。

背景（README/用户确认）：7 人朗读音频格式混杂（wav/m4a/MP3、44.1/48k、立体声），
发音不标准且内容未经人工审核 → 本脚本只做入库与辅助盘点，**不进主评估**
（主评估继续用 LibriSpeech/VCTK，自录集待人工审听后再定）。

运行：
  E:/Miniconda/envs/speechguard/python.exe scripts/self_recorded_ingest.py            # 转码+盘点
  E:/Miniconda/envs/speechguard/python.exe scripts/self_recorded_ingest.py --transcribe  # 追加转写辅助

输出：
  data/self_recorded/processed/full16k/{personN}.wav   16kHz 16bit 单声道
  data/self_recorded/inventory.md / inventory.json      盘点表
  data/self_recorded/transcripts/{personN}.txt          分块转写（仅辅助审内容）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "data", "self_recorded", "raw")
OUT16K = os.path.join(PROJ, "data", "self_recorded", "processed", "full16k")
TRANS = os.path.join(PROJ, "data", "self_recorded", "transcripts")
SR = 16000

ENV_BIN = r"E:/Miniconda/envs/speechguard/Library/bin"


def _tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(ENV_BIN, name + ".exe")
    if os.path.exists(cand):
        return cand
    raise RuntimeError(f"{name} not found")


def probe(path):
    ff = _tool("ffprobe")
    r = subprocess.run(
        [ff, "-v", "error", "-show_entries",
         "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", path],
        capture_output=True, text=True)
    d = json.loads(r.stdout)
    st = d["streams"][0]
    return {"codec": st["codec_name"], "sr": int(st["sample_rate"]),
            "ch": int(st["channels"]), "dur": float(d["format"]["duration"])}


def convert(path, dst):
    ff = _tool("ffmpeg")
    subprocess.run(
        [ff, "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", dst], check=True)


def chunk_bounds(wav):
    """能量找静音点，把长音频切成 ≤30s 块（转写辅助用，避免 CTC 超长退化）。"""
    frame = int(0.03 * SR)
    n = len(wav) // frame
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    bounds, pos = [0], 0
    while (n - pos) * 0.03 > 30:
        lo = pos + int(22 / 0.03)
        hi = pos + int(30 / 0.03)
        cut = lo + int(np.argmin(rms[lo:hi]))
        bounds.append(cut)
        pos = cut
    bounds.append(n)
    return [(b * frame, e * frame) for b, e in zip(bounds[:-1], bounds[1:])]


def transcribe_person(wav_path, txt_path, device):
    import soundfile as sf
    sys.path.insert(0, PROJ)
    from src.content.transcribe import transcribe

    wav, sr = sf.read(wav_path)
    wav = wav.astype(np.float32)
    assert sr == SR
    out = []
    for s, e in chunk_bounds(wav):
        text = transcribe([wav[s:e]], device=device)[0]
        out.append(f"[{s // SR // 60:02d}:{s // SR % 60:02d}-"
                   f"{e // SR // 60:02d}:{e // SR % 60:02d}] {text}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcribe", action="store_true",
                    help="追加 wav2vec2 分块转写（辅助人工审内容）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(OUT16K, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW)
                   if os.path.splitext(f)[1].lower() in
                   (".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"))
    rows = []
    for f in files:
        src = os.path.join(RAW, f)
        stem = os.path.splitext(f)[0]
        dst = os.path.join(OUT16K, stem + ".wav")
        info = probe(src)
        convert(src, dst)
        out_dur = os.path.getsize(dst) / (SR * 2) - 0.001  # s16 mono 减 wav 头
        rows.append({**info, "src": os.path.relpath(src, PROJ),
                     "wav16k": os.path.relpath(dst, PROJ), "out_dur": out_dur})
        print(f"{f}: {info['codec']} {info['sr']}Hz x{info['ch']} "
              f"{info['dur']:.0f}s -> {stem}.wav", flush=True)

    md = ["# 自录集盘点（M4 · 谨慎使用）", "",
          "> 状态：**未进入主评估**。发音不标准、内容未经人工审听；转写仅供对照朗读文本。",
          "> 主 FNR/EER/WER 评估继续使用 LibriSpeech test-clean/dev-clean 与 VCTK。", "",
          "| 文件 | 原始格式 | 原始采样率/声道 | 时长(s) | 转码后(16k mono) | 时长(s) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {os.path.basename(r['src'])} | {r['codec']} | "
                  f"{r['sr']} Hz x{r['ch']} | {r['dur']:.0f} | "
                  f"{os.path.basename(r['wav16k'])} | {r['out_dur']:.0f} |")
    total = sum(r["out_dur"] for r in rows)
    md += ["", f"合计 {len(rows)} 人 / {total / 60:.1f} 分钟。", ""]
    with open(os.path.join(PROJ, "data", "self_recorded", "inventory.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(PROJ, "data", "self_recorded", "inventory.json"),
              "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"inventory: {len(rows)} persons, {total / 60:.1f} min")

    if args.transcribe:
        os.makedirs(TRANS, exist_ok=True)
        for r in rows:
            stem = os.path.splitext(os.path.basename(r["src"]))[0]
            tp = os.path.join(TRANS, stem + ".txt")
            if os.path.exists(tp):
                print(f"skip {stem} (exists)")
                continue
            print(f"transcribe {stem} ...", flush=True)
            transcribe_person(os.path.join(PROJ, r["wav16k"]), tp, args.device)
        print("transcripts ->", TRANS)


if __name__ == "__main__":
    main()
