"""[M5 任务3] 端到端四阶段管线（论文 §5.1）：

  protect  录音实时保护：warping 匿名化 → CTC 转写+词时间戳 → 敏感项自动检测
           → 帧级 AES-CTR 加密（生成参数集 S 与密钥集 W）
  publish  发布：人工增删敏感项（Annotation）→ 重加密 → MP3 编码
           （公式(5) 帧体加密分发版 + 无权限静音分发版）+ 权限保险库（CP-ABE 密文）
  listen   听众：public=静音版；collab=加密版（无 Wi，敏感处噪声）；
           admin=解出帧体后完整 MP3
  recover  恢复：凭组密钥解出 Mi={Si,Wi} → 内容解密 +（持 Si 时）逆扭曲声纹

CLI 一键演示：
  E:/Miniconda/envs/speechguard/python.exe -m src.pipeline demo <wav>
      [--outdir data/probe/pipeline] [--device cuda:0]
产物：publish.wav / publish.mp3 / publish.enc.mp3 / publish.quiet.mp3 /
     vault.json / sk_{admin,collab}.json / listen_*.mp3|wav /
     recovered_{full,voice_only}.wav / summary.json

云端兼容：只用 torch/transformers 通用 API；设备经 --device 传入（npu 亦可）；
模型权重路径不硬编码（transformers 缓存 + speechbrain savedir 由配置决定）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf

from src.acoustic import warping
from src.auth.cp_abe import CPABE
from src.auth.permission import PermissionVault, typical_groups
from src.content.annotation import Annotation
from src.content.detect import detect_sensitive
from src.content.encrypt import FRAME_LEN, decrypt_spans, encrypt_spans, gen_key
from src.content.mp3_crypto import (encode_mp3, encrypt_mp3, parse, quiet_mp3,
                                    time_spans_to_frames)

SR = 16000


def _ffmpeg() -> str:
    return (shutil.which("ffmpeg")
            or r"E:/Miniconda/envs/speechguard/Library/bin/ffmpeg.exe")


def _to_pcm(wav) -> np.ndarray:
    return (np.clip(np.asarray(wav, dtype=np.float64), -1, 1) * 32767).astype(np.int16)


def _to_float(pcm) -> np.ndarray:
    return np.asarray(pcm, dtype=np.float32) / 32768.0


def _consecutive_groups(idxs):
    groups, cur = [], []
    for i in sorted(idxs):
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                groups.append(cur)
            cur = [i]
    if cur:
        groups.append(cur)
    return groups


# ────────────────────────────── 阶段1 protect ──────────────────────────────

def protect(wav16k: np.ndarray, device: str = "cuda:0",
           rng: np.random.Generator | None = None,
           dist_range: tuple[float, float] | None = None) -> dict:
    from src.content.transcribe import transcribe_with_times

    t0 = time.time()
    wav = np.asarray(wav16k, dtype=np.float64).ravel()
    kw = {} if dist_range is None else {"dist_range": dist_range}
    prot, archive = warping.protect(wav, rng=rng, **kw)             # 论文参数集 S
    # OLA 合成输出可能超出 [-1,1]：int16 量化前按峰值归一，recover 端还原增益
    pk = float(np.abs(prot).max())
    gain = 0.98 / pk if pk > 0.98 else 1.0
    prot = prot * gain
    prot_pcm = _to_pcm(prot)
    # 检测在【原始音频】上做（论文 §4.3：SD 转写录音原始音频；与 eval_fnr 评估
    # 协议一致）。36ms 帧网格 warping 前后对齐，CTC 时间戳可直接用于保护态加密。
    ctc = transcribe_with_times(wav.astype(np.float32), device=device)
    hits = detect_sensitive(ctc, device=device)
    ann = Annotation.from_ctc("in-memory", ctc)
    by_idx = {h.idx: h for h in hits}
    for g in _consecutive_groups(h.idx for h in hits):
        ann.mark_words(g, by_idx[g[0]].sources[0])
    spans = ann.merge_spans()
    keys = [gen_key() for _ in spans]                          # 论文密钥集 W
    enc_pcm = encrypt_spans(prot_pcm, spans, keys)
    return {"wav": enc_pcm, "prot_pcm": prot_pcm, "archive": archive,
            "ctc": ctc, "ann": ann, "spans": spans, "keys": keys, "gain": gain,
            "rtc": {"audio_s": wav.size / SR, "protect": time.time() - t0}}


# ────────────────────────────── 阶段2 publish ──────────────────────────────

def publish(bundle: dict, outdir: str, bitrate: int = 128) -> dict:
    """人工修正（bundle["ann"] 已被前端编辑）后：重算区间→重加密→MP3 三态发布。"""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    spans = bundle["ann"].merge_spans()
    keys = [gen_key() for _ in spans]
    enc_pcm = encrypt_spans(bundle["prot_pcm"], spans, keys)
    bundle["spans"], bundle["keys"], bundle["wav"] = spans, keys, enc_pcm

    paths = {}
    paths["wav"] = os.path.join(outdir, "publish.wav")
    sf.write(paths["wav"], _to_float(enc_pcm), SR)

    tmp16 = os.path.join(outdir, "_tmp16.wav")
    shutil.copy(paths["wav"], tmp16)
    paths["mp3"] = os.path.join(outdir, "publish.mp3")
    encode_mp3(tmp16, paths["mp3"], kbps=bitrate)
    os.remove(tmp16)
    data = open(paths["mp3"], "rb").read()
    doc = parse(data)
    spans_sec = [(fs * FRAME_LEN / SR, (fe + 1) * FRAME_LEN / SR)
                 for fs, fe in spans]
    ranges = time_spans_to_frames(doc, spans_sec)
    paths["mp3_enc"] = os.path.join(outdir, "publish.enc.mp3")
    open(paths["mp3_enc"], "wb").write(encrypt_mp3(data, ranges, keys))
    paths["mp3_quiet"] = os.path.join(outdir, "publish.quiet.mp3")
    open(paths["mp3_quiet"], "wb").write(quiet_mp3(data, ranges))
    bundle["mp3"] = {"data": data, "ranges": ranges}

    abe = bundle.get("abe") or CPABE.setup()
    bundle["abe"] = abe
    vault = PermissionVault(abe)
    vault.register(bundle["archive"], keys)
    sks = vault.grant(typical_groups())
    bundle["vault"], bundle["sks"] = vault, sks
    paths["vault"] = os.path.join(outdir, "vault.json")
    open(paths["vault"], "w", encoding="utf-8").write(vault.to_json())
    for name, sk in sks.items():
        p = os.path.join(outdir, f"sk_{name}.json")
        json.dump({k: v.hex() for k, v in sk.items()},
                  open(p, "w", encoding="utf-8"))
        paths[f"sk_{name}"] = p
    bundle["paths"] = paths
    bundle["rtc"]["publish"] = time.time() - t0
    return paths


# ────────────────────────────── 阶段3 listen ──────────────────────────────

def listen(bundle: dict, role: str) -> str:
    """返回可播放文件路径。public→静音分发版；collab→加密版（敏感处噪声）；
    admin→用 Wi 解出帧体的完整 MP3。"""
    outdir = os.path.dirname(bundle["paths"]["mp3"])
    if role == "public":
        src = bundle["paths"]["mp3_quiet"]
    elif role == "admin":
        src = bundle["paths"]["mp3_enc"]
        from src.content.mp3_crypto import decrypt_mp3
        dec = decrypt_mp3(open(src, "rb").read(), bundle["mp3"]["ranges"],
                          bundle["keys"])
        out = os.path.join(outdir, "listen_admin.mp3")
        open(out, "wb").write(dec)
        return out
    elif role == "collab":
        src = bundle["paths"]["mp3_enc"]
    else:
        raise ValueError(f"unknown role {role}")
    out = os.path.join(outdir, f"listen_{role}.mp3")
    shutil.copy(src, out)
    return out


def decode_for_play(path: str, outdir: str) -> str:
    """mp3 → 16k wav（前端播放器统一走 wav）。"""
    out = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".wav")
    if os.path.exists(out):
        return out
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    return out


# ────────────────────────────── 阶段4 recover ──────────────────────────────

def recover(bundle: dict, role: str) -> str:
    """WAV 态完整恢复：解 Wi → 内容还原；持 Si → 逆扭曲声纹还原。"""
    outdir = os.path.dirname(bundle["paths"]["wav"])
    sk = bundle["sks"].get(role)
    if sk is None:
        raise PermissionError(f"role {role} has no credential")
    pkg = bundle["vault"].open(sk, role)
    pcm = decrypt_spans(bundle["wav"], bundle["spans"],
                        pkg.content_keys) if pkg.content_keys else bundle["wav"]
    if pkg.voice:
        wav = warping.recover(_to_float(pcm).astype(np.float64), pkg.archive)
        wav = np.asarray(wav) / bundle.get("gain", 1.0)
        name = "recovered_full" if pkg.content_keys else "recovered_voice_only"
    else:
        wav = _to_float(pcm)
        name = "recovered_content_only"
    out = os.path.join(outdir, f"{name}.wav")
    sf.write(out, np.asarray(wav, dtype=np.float32), SR)
    return out


# ────────────────────────────── CLI 演示 ──────────────────────────────

def run_pipeline(wav_path: str, outdir: str, device: str = "cuda:0") -> dict:
    wav, sr = sf.read(wav_path)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SR:
        os.makedirs(outdir, exist_ok=True)
        tmp = os.path.join(outdir, "_resample.wav")
        subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", wav_path,
                        "-ar", str(SR), "-ac", "1", tmp], check=True)
        wav, sr = sf.read(tmp)
        os.remove(tmp)

    bundle = protect(wav, device=device)
    paths = publish(bundle, outdir)
    summary = {
        "audio_s": bundle["rtc"]["audio_s"],
        "sensitive_items": [
            {"text": it.text, "category": it.category,
             "t0": round(it.t0, 2), "t1": round(it.t1, 2)}
            for it in bundle["ann"].items if not it.deleted],
        "archive_segments": len(bundle["archive"]),
        "outputs": paths,
        "listen": {},
        "recover": {},
    }
    for role in ("public", "collab", "admin"):
        summary["listen"][role] = listen(bundle, role)
    for role in ("collab", "admin"):
        summary["recover"][role] = recover(bundle, role)

    rec = sf.read(summary["recover"]["admin"])[0].astype(np.float64)
    n = min(len(rec), len(wav))
    err = rec[:n] - wav[:n].astype(np.float64)
    snr = 10 * float(np.log10(max((wav[:n].astype(np.float64) ** 2).mean(), 1e-12)
                              / max((err ** 2).mean(), 1e-12)))
    summary["recover_snr_db"] = snr
    summary["rtc"] = bundle["rtc"]
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="protect→publish→listen→recover 一键演示")
    d.add_argument("wav")
    d.add_argument("--outdir", default="data/probe/pipeline")
    d.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.cmd == "demo":
        s = run_pipeline(args.wav, args.outdir, device=args.device)
        print(json.dumps(s, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
