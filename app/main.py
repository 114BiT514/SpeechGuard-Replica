"""[M6 任务1] SpeechGuard Gradio 前端：音频主人 / 听众 双角色。

四阶段（protect → publish → listen → recover）直接复用 src.pipeline，本文件只做
界面编排。启动：

  E:/Miniconda/envs/speechguard/python.exe -m app.main            # 默认 7860
  E:/Miniconda/envs/speechguard/python.exe -m app.main --device cuda:0

浏览器：录音/上传 → 一键保护+检测 → 敏感项人工增删 → 发布（MP3 三态+权限保险库）
→ 听众按角色试听 → 按权限恢复下载。
"""
from __future__ import annotations

import argparse
import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf

from src import pipeline
from src.viz import sensitive_timeline, spectrogram_row

DIST_LEVELS = {"轻度 (0.3–0.4)": (0.3, 0.4),
               "标准 (0.5–0.6)": (0.5, 0.6),
               "强 (0.7–0.8)": (0.7, 0.8)}
DEVICE = os.environ.get("SG_DEVICE", "cuda:0")
WORKDIR = os.environ.get("SG_WORKDIR",
                         os.path.join(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))), "data", "probe", "app"))


def _items_text(ann):
    rows = ["**敏感项列表**（发布前可增删）", "",
            "| ID | 内容 | 类别 | 时间(s) | 来源 | 状态 |",
            "|---|---|---|---|---|---|"]
    for it in ann.items:
        rows.append(f"| {it.iid} | {it.text} | {it.category} | "
                    f"{it.t0:.2f}–{it.t1:.2f} | {it.source} | "
                    f"{'已删除' if it.deleted else '有效'} |")
    return "\n".join(rows)


def _word_choices(ctc):
    return [f"{i}: {w} [{t0:.2f}-{t1:.2f}]" for i, (w, t0, t1) in enumerate(ctc)]


def _item_choices(ann):
    return [f"{it.iid}: {it.text}" for it in ann.items if not it.deleted]


# ───────────────────────── 主人：保护 + 检测 ─────────────────────────

def load_sample():
    """内置示例语音（真实 LibriSpeech 句，免上传直接走流程）。"""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "sample_utt.wav")
    if not os.path.exists(p):
        raise gr.Error("示例语音缺失（assets/sample_utt.wav）")
    return p


def protect_detect(audio, level_name):
    if audio is None:
        raise gr.Error("请先录音或上传音频")
    sr, wav = audio
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav[:, 0]
    # 兼容 Gradio 各版本 type="numpy" 的幅度约定（float±1 / int16±32767）
    if float(np.abs(wav).max()) > 4.0:
        wav = wav.astype(np.float32) / 32768.0
    else:
        wav = wav.astype(np.float32)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    bundle = pipeline.protect(wav, device=DEVICE,
                              dist_range=DIST_LEVELS[level_name])
    tmpdir = tempfile.mkdtemp(prefix="sess_", dir=_ensure_workdir())
    bundle["tmpdir"] = tmpdir
    orig_path = os.path.join(tmpdir, "original.wav")
    prot_path = os.path.join(tmpdir, "protected.wav")
    sf.write(orig_path, wav, 16000)
    sf.write(prot_path, pipeline._to_float(bundle["wav"]), 16000)
    spec = spectrogram_row([(wav, "原始"), (pipeline._to_float(bundle["wav"]),
                                          "保护后（匿名化+敏感词加密）")],
                           os.path.join(tmpdir, "spec_protect.png"))
    tl = sensitive_timeline(bundle["ann"], wav.size / 16000,
                           os.path.join(tmpdir, "timeline.png"))
    transcript = " ".join(w for w, _, _ in bundle["ctc"])
    return (bundle, prot_path, spec, tl,
            f"**转写（wav2vec2，词时间戳同源）**\n\n{transcript}\n\n" + _items_text(bundle["ann"]),
            gr.update(choices=_word_choices(bundle["ctc"])),
            gr.update(choices=_item_choices(bundle["ann"])))


# ───────────────────────── 主人：人工增删 ─────────────────────────

def add_words(bundle, start_sel, end_sel):
    if bundle is None:
        raise gr.Error("请先在①执行保护+检测")
    if not start_sel:
        raise gr.Error("请先在左侧下拉中选择一个起始词（下拉无选项说明转写为空）")
    i0 = int(start_sel.split(":")[0])
    i1 = int(end_sel.split(":")[0]) if end_sel else i0
    if i1 < i0:
        i0, i1 = i1, i0
    bundle["ann"].mark_words(list(range(i0, i1 + 1)), "manual", source="manual")
    bundle["wav"] = _reencrypt(bundle)
    return (bundle, _items_text(bundle["ann"]),
            gr.update(choices=_item_choices(bundle["ann"])),
            os.path.join(bundle["tmpdir"], "timeline.png")
            if _save_timeline(bundle) else None)


def remove_item(bundle, sel):
    if bundle is None:
        raise gr.Error("请先在①执行保护+检测")
    if not sel:
        raise gr.Error("请先在「删除敏感项」下拉中选择要删除的项")
    iid = sel.split(":")[0].strip()
    bundle["ann"].remove(iid)
    bundle["wav"] = _reencrypt(bundle)
    return (bundle, _items_text(bundle["ann"]),
            gr.update(choices=_item_choices(bundle["ann"])),
            _save_timeline(bundle))


def _save_timeline(bundle):
    dur = bundle["wav"].size / 16000
    p = os.path.join(bundle["tmpdir"], "timeline.png")
    sensitive_timeline(bundle["ann"], dur, p)
    return p


def _reencrypt(bundle):
    spans = bundle["ann"].merge_spans()
    keys = [pipeline.gen_key() for _ in spans]
    bundle["spans"], bundle["keys"] = spans, keys
    return pipeline.encrypt_spans(bundle["prot_pcm"], spans, keys)


# ───────────────────────── 主人：发布 ─────────────────────────

def publish(bundle):
    if bundle is None:
        raise gr.Error("先执行保护+检测")
    paths = pipeline.publish(bundle, bundle["tmpdir"])
    n_items = len([i for i in bundle["ann"].items if not i.deleted])
    info = (f"**发布完成**：敏感项 {n_items} 个 / 加密对象 {len(paths and bundle['spans'])} 个"
            f"（每组独立 AES 密钥）\n\n"
            f"- 加密分发版 MP3：`{os.path.basename(paths['mp3_enc'])}`（持 Wi 可解）\n"
            f"- 静音分发版 MP3：`{os.path.basename(paths['mp3_quiet'])}`（无权限：敏感段静音）\n"
            f"- 权限保险库：admin / collab 两组凭属性证书解出各自资产；public 组无资产\n")
    return bundle, info, paths["mp3_enc"], paths["mp3_quiet"]


# ───────────────────────── 听众 ─────────────────────────

def listen_role(bundle, role):
    if bundle is None:
        raise gr.Error("还没有发布任何音频（主人 Tab 先走流程）")
    f = pipeline.listen(bundle, {"公开听众": "public", "合作者": "collab",
                                 "管理员": "admin"}[role])
    return f


def recover_role(bundle, role):
    if bundle is None:
        raise gr.Error("还没有发布任何音频")
    if role == "公开听众":
        raise gr.Error("public 组无任何资产：只听到静音分发版，无法恢复")
    rmap = {"管理员": "admin", "合作者": "collab"}
    out = pipeline.recover(bundle, rmap[role])
    prot = pipeline._to_float(bundle["wav"])
    rec, _sr = sf.read(out)
    title = ("按权限恢复完成（管理员：内容解密 + 声纹逆扭曲）" if role == "管理员"
             else "按权限恢复完成（合作者：仅声纹逆扭曲，敏感词保持加密）")
    spec = spectrogram_row([(prot, "发布版（保护态）"),
                            (np.asarray(rec, np.float64), "恢复后")],
                           os.path.join(bundle["tmpdir"],
                                        f"spec_recover_{role}.png"))
    return out, spec, title


def _ensure_workdir():
    os.makedirs(WORKDIR, exist_ok=True)
    return WORKDIR


# ───────────────────────── 界面 ─────────────────────────

def build_ui():
    with gr.Blocks(title="SpeechGuard 语音隐私保护演示") as demo:
        gr.Markdown("# SpeechGuard：可恢复、可定制的语音隐私保护（论文复现演示）\n"
                    "论文：*SpeechGuard*, USENIX Security 2025。声学隐私（多参数频域 "
                    "warping）+ 内容隐私（敏感词帧级 AES-CTR）+ 权限组（属性基分发）。")
        state = gr.State()

        with gr.Tab("🎙 音频主人（protect → publish）"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(sources=["upload", "microphone"],
                                         type="numpy", label="原始语音")
                    level = gr.Radio(sorted(DIST_LEVELS, key=lambda k:
                                             {"轻度 (0.3–0.4)": 0,
                                              "标准 (0.5–0.6)": 1,
                                              "强 (0.7–0.8)": 2}[k]),
                                     value="标准 (0.5–0.6)", label="保护强度 dist")
                    with gr.Row():
                        btn_sample = gr.Button("载入示例语音")
                        btn_prot = gr.Button("① 保护 + 敏感项自动检测",
                                             variant="primary")
                with gr.Column():
                    prot_player = gr.Audio(type="filepath", label="保护后语音（试听）")
            spec_prot = gr.Image(label="频谱对比：原始 vs 保护", type="filepath")
            tl_prot = gr.Image(label="敏感词时间轴", type="filepath")
            items_md = gr.Markdown()
            with gr.Row():
                word_dd = gr.Dropdown(choices=[], label="增补敏感词（起始词）")
                word_dd2 = gr.Dropdown(choices=[], label="（结束词，可空）")
                btn_add = gr.Button("② 人工增补敏感项")
            item_dd = gr.Dropdown(choices=[], label="删除敏感项")
            btn_del = gr.Button("② 人工删除敏感项")
            btn_pub = gr.Button("③ 发布（MP3 三态 + 权限保险库）", variant="primary")
            pub_info = gr.Markdown()
            enc_player = gr.Audio(type="filepath", label="加密分发版 MP3")
            quiet_player = gr.Audio(type="filepath", label="静音分发版 MP3（无权限）")

            btn_sample.click(load_sample, [], audio_in)
            btn_prot.click(protect_detect, [audio_in, level],
                           [state, prot_player, spec_prot, tl_prot, items_md,
                            word_dd, item_dd])
            btn_add.click(add_words, [state, word_dd, word_dd2],
                          [state, items_md, item_dd, tl_prot])
            btn_del.click(remove_item, [state, item_dd],
                          [state, items_md, item_dd, tl_prot])
            btn_pub.click(publish, [state], [state, pub_info, enc_player,
                                             quiet_player])

        with gr.Tab("🎧 听众（listen → recover）"):
            role = gr.Radio(["管理员", "合作者", "公开听众"], value="公开听众",
                            label="我的权限组（由 owner 分发属性证书）")
            btn_listen = gr.Button("在线试听", variant="primary")
            listen_player = gr.Audio(type="filepath", label="播放器")
            btn_recover = gr.Button("按权限恢复并下载")
            rec_player = gr.Audio(type="filepath", label="恢复结果")
            spec_rec = gr.Image(label="频谱对比：发布态 vs 恢复态", type="filepath")
            rec_info = gr.Markdown()
            btn_listen.click(listen_role, [state, role], listen_player)
            btn_recover.click(recover_role, [state, role],
                              [rec_player, spec_rec, rec_info])

        gr.Markdown("---\n本地开发模式（云端 ModelArts 部署见 deploy/，M7）。"
                    "同一浏览器会话内主人发布后切到听众 Tab 即可体验三种权限。")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    if args.device:
        globals()["DEVICE"] = args.device
    demo = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port,
                        share=False, inbrowser=False)


if __name__ == "__main__":
    main()
