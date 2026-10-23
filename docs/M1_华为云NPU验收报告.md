# M1 华为云昇腾 NPU 环境验收报告

> 对应执行计划：M1 任务 8「华为云开卡（部署前置）」
> 验收日期：2026-10-23
> 状态：**已通过**（ECAPA 与 Conformer 均已在昇腾 NPU 上完成推理）
> 执行人：项目组（工程线）

---

## 0. 一页结论（TL;DR）

- 在华为云 ModelArts 上创建了 **1× Ascend 910B4（32GB HBM）** 的 Notebook 实例，区域 **西南-贵阳一**。
- 预置镜像自带 **CANN 8.0.0 + torch 2.1.0 + torch_npu 2.1.0.post10**，`torch.npu.is_available() == True`。
- **ECAPA-TDNN** 与 **Conformer ASR** 两个预训练模型均已在 `npu:0` 上跑通，ASR 转写结果正确。
- 复现命令见 §4，踩坑与解决见 §5。
- 计费参考价 **¥18.689/小时**，用完务必停止实例。

---

## 1. 验收目标与标准

执行计划中 M1 的华为云任务要求：

1. 实名认证，开通 ModelArts 与 OBS；
2. 创建一个 Ascend（910B/310P）Notebook 实例；
3. 验证 `import torch, torch_npu; torch.npu.is_available()`；
4. 把 ECAPA-TDNN 在 NPU 上跑通一条音频推理。

本报告在上述基础上额外完成了 **Conformer ASR** 的 NPU 推理，覆盖 M1 DoD 中「ECAPA 与 Conformer 各跑通一条音频推理」的要求。

---

## 2. 资源配置

| 项目 | 实际配置 |
|---|---|
| 云平台 | 华为云 ModelArts |
| 区域 | **西南-贵阳一**（`cn-southwest-2`） |
| 入口 | ModelArts → 模型开发与训练 → Notebook → 创建 |
| 资源池 | 公共资源池 |
| 实例规格 | `modelarts.bm.npu.arm.1snt9b1`（控制台显示：`1 * ascend-snt9b1 \| 24 vCPUs \| 192 GiB \| ARM`） |
| 实际芯片（`npu-smi`） | **Ascend 910B4**，HBM 32768 MB（32GB） |
| 镜像 | `pytorch_2_1_ascend: pytorch_2.1.0-cann_8.0.0-py_3.10-euler_2.10.11-aarch64-snt9b-20250305173557-cb53968` |
| 系统盘 | 50 GiB 本地盘（免费，删除实例时释放） |
| 数据盘 | **EVS 50 GiB**，挂载到 `/home/ma-user/work`（停止实例后保留，持续计费） |
| 网络 | 公共网络 + 允许实例访问公网（实际出口走平台代理） |
| SSH 远程开发 | 已开启（可选） |
| 自动停止 | 1 小时 |
| 计费方式 | 按需 |
| 参考价 | **¥18.689/小时**（实际以账单为准） |

> 说明：控制台规格名是 `snt9b1`，但 `npu-smi` 实际识别为 **910B4**。写报告时以 `npu-smi` 为准，并保留控制台规格名，方便他人复现。
>
> 910B 系列在部分账号下属于**受限购买**，可能看不到规格或显示售罄。遇到时提交工单申请「西南-贵阳一 ModelArts 昇腾 910B 单卡 Notebook 规格」，或换区域重试，不要改租 8 卡整机。

---

## 3. 环境版本

| 组件 | 版本 |
|---|---|
| Python | 3.10（conda 环境名 `PyTorch-2.1.0`） |
| PyTorch | 2.1.0 |
| torch_npu | 2.1.0.post10 |
| CANN | 8.0.0 |
| npu-smi | 25.5.1 |
| torchaudio | 2.1.0（手动安装） |
| SpeechBrain | **1.0.3**（必须锁版本，见 §5.2） |
| transformers | 4.45.0（镜像预装） |
| torchvision | 0.16.0（镜像预装） |
| numpy | 1.23.0 |
| librosa | 0.10.2.post1 |
| soundfile | 0.13.1 |

---

## 4. 复现步骤与验证证据

### 4.1 创建实例

在 ModelArts 控制台按 §2 的配置创建 Notebook，等待状态变为「运行中」，打开 JupyterLab，进入 Terminal。

**注意顺序**：先把「实例规格」切到 **NPU → 1× Ascend 910B**，再点「选择镜像」。否则镜像列表只会显示 x86 + CUDA 的 GPU 镜像。

### 4.2 确认 NPU 可用

```bash
npu-smi info

python -c "import torch, torch_npu; \
print('torch', torch.__version__); \
print('torch_npu', torch_npu.__version__); \
print('npu', torch.npu.is_available(), torch.npu.device_count())"
```

实测输出（关键行）：

```text
torch 2.1.0
torch_npu 2.1.0.post10
npu True 1
```

再做一次 NPU 矩阵乘法，确认算子真正执行在卡上：

```bash
python - <<'PY'
import torch, torch_npu
x = torch.randn(4096, 4096, device="npu")
y = x @ x
torch.npu.synchronize()
print("matmul ok:", y.device, float(y[0, 0]))
PY
```

实测输出：

```text
matmul ok: npu:0 14.001969337463379
```

### 4.3 安装推理依赖

镜像已带 torch / torch_npu / transformers，**不要重装 torch**。只需要补 torchaudio 和 SpeechBrain：

```bash
python -m pip install torchaudio==2.1.0 --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install "speechbrain==1.0.3" --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install hyperpyyaml joblib sentencepiece tqdm \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python -c "import torch, torchaudio, speechbrain; \
print(torch.__version__, torchaudio.__version__, speechbrain.__version__)"
```

期望：

```text
2.1.0 2.1.0 1.0.3
```

### 4.4 下载预训练权重（代理注意事项）

ModelArts 实例**不能直连公网**，必须走平台代理：

```text
http_proxy = https_proxy = http://127.0.0.1:9999
PROXY_COMMON = http://proxy-notebook.modelarts.com:8083
no_proxy 包含 *.myhuaweicloud.com、pip.modelarts.private.com
```

因此**不要 unset 代理**。HuggingFace 直连会超时，需要 `HF_ENDPOINT=https://hf-mirror.com`。下载时用单线程 + 重试，多线程容易被代理掐断：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

python - <<'PY'
import time
from huggingface_hub import snapshot_download

repo = "speechbrain/asr-conformer-transformerlm-librispeech"
local = "pretrained_models/conformer"
last = None
for i in range(10):
    try:
        p = snapshot_download(repo_id=repo, local_dir=local, max_workers=1)
        print("downloaded to", p)
        break
    except Exception as e:
        last = e
        print(f"attempt {i+1} failed: {type(e).__name__}")
        time.sleep(5)
else:
    raise SystemExit(f"all attempts failed: {last}")
PY
```

本次实测：大文件（`asr.ckpt` 442MB、`lm.ckpt` 381MB）下载成功；最后的小文件 `tokenizer.ckpt` 被代理挂住，改用 curl 补齐：

```bash
curl -L --fail --retry 30 --retry-all-errors --retry-delay 3 \
     --connect-timeout 30 --max-time 600 \
     -o pretrained_models/conformer/tokenizer.ckpt \
     "https://hf-mirror.com/speechbrain/asr-conformer-transformerlm-librispeech/resolve/main/tokenizer.ckpt"
```

ECAPA 权重同理，可直接用 curl 逐个下载（5 个文件）：

```bash
mkdir -p pretrained_models/ecapa_manual
cd pretrained_models/ecapa_manual
for f in hyperparams.yaml embedding_model.ckpt mean_var_norm_emb.ckpt classifier.ckpt label_encoder.txt; do
  echo "== downloading $f =="
  curl -L --fail --retry 30 --retry-all-errors --retry-delay 3 \
       --connect-timeout 30 --max-time 1800 \
       -o "$f" "https://hf-mirror.com/speechbrain/spkrec-ecapa-voxceleb/resolve/main/$f"
done
cd ../..
```

### 4.5 ECAPA-TDNN 推理（声纹嵌入）

```bash
python - <<'PY'
import torch, torch_npu
from speechbrain.inference.speaker import EncoderClassifier

clf = EncoderClassifier.from_hparams(
    source="pretrained_models/ecapa_manual",
    run_opts={"device": "npu"},
)
print("model loaded, device =", clf.device)

wav = torch.randn(1, 16000 * 3)  # 3 秒测试音频
with torch.no_grad():
    emb = clf.encode_batch(wav)
torch.npu.synchronize()
print("ECAPA OK:", tuple(emb.shape), emb.device)
PY
```

实测输出：

```text
model loaded, device = npu
ECAPA OK: (1, 1, 192) npu:0
```

结论：ECAPA-TDNN 的 fbank 特征提取与卷积/注意力主干在 910B4 上原生可用，无需 CPU fallback。

### 4.6 Conformer ASR 推理（语音转写）

```bash
python - <<'PY'
import os, torch, torch_npu
from speechbrain.inference.ASR import EncoderDecoderASR

asr = EncoderDecoderASR.from_hparams(
    source="pretrained_models/conformer",
    run_opts={"device": "npu"},
)
print("model loaded, device =", asr.device)

path = "pretrained_models/conformer/example.wav"
import torchaudio
wav, sr = torchaudio.load(path)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav.mean(0, keepdim=True)

with torch.no_grad():
    text, scores = asr.transcribe_batch(wav, torch.tensor([1.0]))
torch.npu.synchronize()
print("Conformer OK:", text)
PY
```

实测输出：

```text
model loaded, device = npu
using pretrained_models/conformer/example.wav (1, 52173)
Conformer OK: ['THE BIRCH CANOE SLID ON THE SMOOTH PLANKS']
```

结论：Conformer + Transformer LM 的编码器、注意力与自回归解码在 910B4 上可运行，转写结果正确。

---

## 5. 踩坑与解决

| # | 现象 | 原因 | 解决方法 |
|---|---|---|---|
| 5.1 | 「选择镜像」里只有 `x86_64 + CUDA` 的 PyTorch 镜像 | 实例规格还停留在 CPU/GPU，镜像列表按规格过滤 | 先选 **NPU → 1× Ascend 910B**，再选镜像；Ascend 镜像名一定带 `aarch64-snt9b` |
| 5.2 | `speechbrain 1.1.2 requires torch>=2.7.1` | SpeechBrain 最新版把 torch 下限提到了 2.7.1，与镜像的 torch 2.1 冲突 | 锁定 **`speechbrain==1.0.3`**（要求 `torch>=1.9`，无上限） |
| 5.3 | `pip install torchaudio` 可能把 torch 顶掉 | torchaudio 默认依赖会触发 torch 重装 | 一律加 **`--no-deps`**，并显式指定 `torchaudio==2.1.0` |
| 5.4 | 直连 `hf-mirror.com` / `huggingface.co` 超时 | 实例无公网直连，必须走平台代理 | 保留 `http(s)_proxy`，设置 `HF_ENDPOINT=https://hf-mirror.com`；不要 unset 代理 |
| 5.5 | `snapshot_download` 多线程时报 `ReadTimeout` / `ProxyError` / `LocalEntryNotFoundError` | 共享代理并发连接不稳定 | `max_workers=1` + 外层重试循环；已下载文件会断点续传；极小文件改用 `curl --retry` |
| 5.6 | `UserWarning: ... owner does not match the current owner` | CANN 目录属主与当前用户不一致 | 无害，忽略 |
| 5.7 | `Support for mismatched key_padding_mask and attn_mask is deprecated` | PyTorch 2.1 的兼容性提示 | 无害，忽略 |
| 5.8 | 规格列表里找不到 910B / 显示售罄 | 受限购买或公共资源池售罄 | 提交工单开通单卡 910B 配额；或换区域重试；不要租 8 卡整机 |

---

## 6. 费用、停机与数据保留

- 计费参考价：**¥18.689/小时**（1×910B，按需），实际以账单为准。
- 本次验证约 1 小时；**验证完成后立即在控制台「停止」实例**。
- 停止后：
  - 计算资源不再计费；
  - EVS 数据盘继续按 GiB 计费；
  - `/home/ma-user/work` 下内容保留，包括本次下载的 `pretrained_models/ecapa_manual` 与 `pretrained_models/conformer`。
- 如需彻底零费用，可删除实例（数据盘一并释放），但下次需要重新下载约 900MB 权重。
- 后续 M2/M3 主要在本地 CPU 完成，云端实例保持停止即可；M7 部署前再启动。

---

## 7. 结论与下一步

**已完成：**

- ModelArts 昇腾 Notebook 创建与计费链路跑通；
- CANN 8.0.0 + torch 2.1.0 + torch_npu 2.1.0.post10 环境验证通过；
- ECAPA-TDNN、Conformer ASR 两个模型在 910B4 上推理成功；
- 形成可复现的依赖安装、权重下载、NPU 推理命令。

**M1 仍未完成（不依赖云端）：**

- LibriSpeech test-clean / dev-clean、VCTK 数据就位；
- WavLM、BERT NER 权重下载（可在 M4 前完成）；
- 自录语音采集；
- 论文 §4/§5 精读与「公式-表号」对照笔记。

**M7 需要补做的 NPU 验证：**

- WavLM、BERT-base NER 逐个 `to('npu')`；
- 记录不支持的算子与 CPU fallback；
- 权重与数据上传 OBS，完成 ModelArts 在线服务 / 公网访问部署。

---

## 8. 附：最短复验清单

```bash
# 1. NPU 是否可用
npu-smi info
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"

# 2. 依赖版本
python -c "import torch, torchaudio, speechbrain; print(torch.__version__, torchaudio.__version__, speechbrain.__version__)"

# 3. 权重目录
ls -lh pretrained_models/ecapa_manual
ls -lh pretrained_models/conformer

# 4. 跑 ECAPA（见 §4.5）与 Conformer（见 §4.6）
```
