"""M1 环境冒烟测试：验证 torch/GPU 可用，ECAPA-TDNN 与 Conformer ASR 各跑通一条推理。

模型会缓存到 data/pretrained/（SpeechBrain savedir），结果追加写入
data/downloads/smoke_test_out.txt。运行方式：
    python scripts/smoke_test.py
"""
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import numpy as np
import soundfile as sf
import torch

LINES = []


def log(msg):
    print(msg, flush=True)
    LINES.append(str(msg))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "-"
    log(f"[1/3] torch {torch.__version__} | device={device} | gpu={gpu}")

    # 3 秒测试音频（440Hz 正弦），供 ASR/ASV 推理
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    test_wav = os.path.join(PROJ, "assets", "test_tone.wav")
    sf.write(test_wav, wav, sr)
    # SpeechBrain 内部处理 Windows 反斜杠路径会丢字符，统一用正斜杠
    test_wav = os.path.abspath(test_wav).replace("\\", "/")

    # ECAPA-TDNN 说话人嵌入
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # Windows 无符号链接特权：用 COPY 策略直接把模型文件落到 savedir
    strategy = (
        LocalStrategy.COPY_SKIP_CACHE if sys.platform == "win32" else LocalStrategy.SYMLINK
    )

    t0 = time.time()
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(PROJ, "data", "pretrained", "ecapa-voxceleb"),
        run_opts={"device": device},
        local_strategy=strategy,
    )
    with torch.no_grad():
        emb = ecapa.encode_batch(torch.tensor(wav).unsqueeze(0))
    log(f"[2/3] ECAPA ok | embedding shape={tuple(emb.shape)} | {time.time() - t0:.1f}s")

    # Conformer ASR 转写
    from speechbrain.inference.ASR import EncoderDecoderASR

    t0 = time.time()
    asr = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-conformer-transformerlm-librispeech",
        savedir=os.path.join(PROJ, "data", "pretrained", "conformer-asr"),
        run_opts={"device": device},
        local_strategy=strategy,
    )
    text = asr.transcribe_file(test_wav)
    log(f"[3/3] Conformer ASR ok | transcription={text!r} | {time.time() - t0:.1f}s")

    log("SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"SMOKE_FAILED: {type(e).__name__}: {e}")
    finally:
        with open(
            os.path.join(PROJ, "data", "downloads", "smoke_test_out.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(LINES))
