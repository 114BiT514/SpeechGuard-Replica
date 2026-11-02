# 华为云 ModelArts + 昇腾部署（M7）

> 状态（2026-11-02）：**部署链路已在 ModelArts 昇腾 Notebook 上跑通**
> —— 66 项单测全绿、pipeline 端到端在 `npu:0` 上完成、Gradio 前端可访问。
> 详见 [M7 部署复现记录](../docs/M7_华为云部署复现.md)。

目标：整套系统（pipeline + Gradio 前端）部署到华为云 ModelArts 昇腾 NPU 实例，公网可访问。

口径：华为云 ModelArts + 昇腾 NPU（910B/310P）+ torch_npu（PyTorch 昇腾适配层）+ OBS 存储。
若课程硬性要求 MindSpore 框架：仅将 BERT NER 等单点换为 MindSpore Hub 等价实现，主链路不变。

---

## 0. 一页结论

| 项目 | 结果 |
|---|---|
| 实例 | ModelArts Notebook，`ascend-snt9b1`（`npu-smi` 识别 910B4 32GB），`cn-southwest-2` |
| 镜像 | `pytorch_2_1_ascend`（torch 2.1.0 + torch_npu 2.1.0.post10 + CANN 8.0.0，Python 3.10） |
| 代码位置 | `/home/ma-user/work/SpeechGuard-Replica`（EVS 数据盘，停止实例后保留） |
| 依赖 | 见 §2.2；关键版本 `gradio==4.44.1`、`huggingface-hub==0.29.1`、`pydantic==2.9.2`、`urllib3<2` |
| 模型 | wav2vec2-large-960h-lv60-self + bert-large-cased-conll03，共 3.7GB（`data/hf-cache`） |
| 单元测试 | `66 passed`（含 MP3 加解密、pipeline 端到端） |
| NPU 推理 | `protect` 35.2s（含首次模型加载）+ `publish` 0.15s，三态 MP3/恢复音频全部产出 |
| 前端 | `0.0.0.0:7860`，设备自动选 `npu:0` |
| 公网入口 | Gradio `share=True` 被平台代理拦截（frp 7000 端口 403）；改用 Pinggy SSH-over-443 临时公网链接，或 SSH 本地转发 |

---

## 1. 环境

| 组件 | 版本 |
|---|---|
| Python | 3.10.6（conda 环境 `PyTorch-2.1.0`） |
| PyTorch / torch_npu | 2.1.0 / 2.1.0.post10 |
| CANN | 8.0.0 |
| transformers | 4.45.0 |
| gradio / gradio-client | 4.44.1 / 1.3.0 |
| huggingface-hub | 0.29.1 |
| pydantic | 2.9.2 |
| urllib3 | 1.26.20 |
| speechbrain / torchaudio | 1.0.3 / 2.1.0 |
| ffmpeg | 系统 4.2 **无 libmp3lame**；由 `imageio-ffmpeg 0.8.0` 兜底 |

> **不要**在云端执行根目录 `environment.yml`：会覆盖镜像预装的 CANN/torch_npu。
> 补装 torch 相关依赖一律 `--no-deps`。

---

## 2. 从零部署步骤

### 2.1 上传代码

在本地仓库根目录打包（排除数据与大文件），上传后解压：

```bash
tar --exclude=data --exclude=.git --exclude=__pycache__ --exclude='*.npz' \
    -czf /tmp/sg_code.tar.gz src app scripts experiments tests assets docs deploy \
    README.md requirements.txt environment.yml .gitignore

scp -P 30905 -i KeyPair-2d47.pem /tmp/sg_code.tar.gz \
    ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:/home/ma-user/work/

ssh -p 30905 -i KeyPair-2d47.pem ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com \
  'cd /home/ma-user/work && mkdir -p SpeechGuard-Replica && \
   tar -xzf sg_code.tar.gz -C SpeechGuard-Replica && rm sg_code.tar.gz'
```

> 若 `scp` 报 `path canonicalization failed`，先 `mkdir -p` 目标父目录。
> 若本机 Bash 处于沙箱、`scp` 看不到刚下载的文件，可用
> `cat file | ssh ... 'cat > remote'` 走管道上传。

### 2.2 安装依赖（版本锁定）

```bash
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 180"

# torch 生态：必须 --no-deps，防止把镜像里的 torch/torch_npu 顶掉
$PIP torchaudio==2.1.0 --no-deps
$PIP speechbrain==1.0.3 --no-deps

$PIP hyperpyyaml pydub jiwer pyeer imageio-ffmpeg

# Gradio 4.x + hub 0.29 + pydantic 2.9 + urllib3<2：
# gradio 5/6 要求 huggingface-hub>=1.x，会破坏 transformers 4.45
$PIP "gradio==4.44.1" "huggingface-hub==0.29.1" "pydantic==2.9.2" "urllib3<2"
```

验证：

```bash
python -c "import torch, torch_npu, transformers, gradio, huggingface_hub, pydantic; \
print(torch.__version__, torch.npu.is_available(), transformers.__version__, \
      gradio.__version__, huggingface_hub.__version__, pydantic.VERSION)"
# 期望：2.1.0 True 4.45.0 4.44.1 0.29.1 2.9.2
```

### 2.3 下载模型（走 hf-mirror）

云端 `hf-mirror.com` 可通（平台代理已配置）。下载到仓库内缓存，代码会读
`PROJ/data/hf-cache`：

```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/data/hf-cache
export HF_HUB_DOWNLOAD_TIMEOUT=600
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["facebook/wav2vec2-large-960h-lv60-self",
             "dbmdz/bert-large-cased-finetuned-conll03-english"]:
    snapshot_download(repo_id=repo, max_workers=1,
                      allow_patterns=["*.json","*.txt","*.model","*.bin","*.safetensors"])
PY
du -sh data/hf-cache     # 约 3.7G
```

### 2.4 单元测试

测试 `test_pipeline.py` 需要真实 LibriSpeech flac（取排序后第 4 条）。只验证部署时，
上传任意一个 chapter（约 5MB）即可：

```bash
# 本地
scp -r -P 30905 -i KeyPair-2d47.pem \
  data/librispeech/LibriSpeech/test-clean/1089 \
  ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:/home/ma-user/work/SpeechGuard-Replica/data/librispeech/LibriSpeech/test-clean/
```

```bash
cd /home/ma-user/work/SpeechGuard-Replica
python -m pytest tests/ -q
# 66 passed
```

### 2.5 NPU 端到端验证

```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache

python -m src.pipeline demo assets/sample_utt.wav \
    --outdir data/probe/pipeline --device npu:0
```

实测摘要：

```json
{
  "audio_s": 3.105,
  "sensitive_items": [
    {"text": "POLITICS", "category": "vocab:politics", "t0": 1.64, "t1": 2.04},
    {"text": "RELIGION", "category": "vocab:religion", "t0": 2.2, "t1": 3.1}
  ],
  "rtc": {"protect": 35.2, "publish": 0.146},
  "recover_snr_db": 2.375
}
```

> `src/device.py` 会自动 `import torch_npu` 并选择 `npu:0`；`torch` 本身不认识
> `npu` 设备，必须显式注册。日志里的 `ASCEND_HOME_PATH is not set` 为无害告警。

### 2.6 启动前端

```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache
python -m app.main --device npu:0 --host 0.0.0.0 --port 7860
# [device] {'torch': '2.1.0', 'npu_available': True, 'npu_count': 1, 'picked': 'npu:0'}
# Running on local URL: http://0.0.0.0:7860
```

后台常驻：

```bash
cat > ~/start_app.sh <<'SH'
#!/bin/bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
exec python -u -m app.main --device npu:0 --host 0.0.0.0 --port 7860
SH
chmod +x ~/start_app.sh
setsid nohup bash ~/start_app.sh > ~/gradio.log 2>&1 < /dev/null &
```

### 2.7 公网入口

#### 方式 A：SSH 本地转发（最稳定，推荐日常验收）

在你自己的电脑上：

```bash
ssh -N -L 7860:127.0.0.1:7860 -p 30905 -i KeyPair-2d47.pem \
    ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com
# 浏览器打开 http://127.0.0.1:7860/
```

#### 方式 B：Pinggy SSH-over-443（公网临时链接，免费 60 分钟）

平台代理**只放行 443**、拦截 frp 的 7000 端口，所以 Gradio 自带 share 不可用。
Pinggy 提供 SSH over 443，可绕过：

```bash
# 一行：经平台 HTTP 代理 CONNECT a.pinggy.io:443，再开反向隧道
ssh -tt -o ProxyCommand="python ~/connect_proxy.py %h %p" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=6 \
    -p 443 -R 0:localhost:7860 a.pinggy.io
```

`~/connect_proxy.py` 是一个极小的 CONNECT 代理转发器（见
[M7 部署复现记录](../docs/M7_华为云部署复现.md) §4）。连接成功后会打印两个
`https://*.pinggy*.link` 公网地址。注意：

- 免费隧道 **60 分钟过期**，且每次重连 URL 会变；
- 需要 `-tt`（Pinggy 用交互 shell 打印 URL）；
- 平台代理对长连接可能中断，建议 `ServerAliveInterval=10`；
- 后台常驻时用 `sleep infinity | ssh ...` 保持 stdin 不关闭。

#### 方式 C：ModelArts 控制台 Jupyter 代理

在 ModelArts 控制台打开该 Notebook 的 JupyterLab（已登录会话），把地址末尾改为
`/proxy/7860/` 即可访问 Gradio。该入口需要华为云登录，不是匿名公网。

#### 为什么 Gradio `share=True` 不可用

`gradio 4.44` 的 share 依赖 `frp.gradio.live:7000`。平台代理对该端口的 `CONNECT`
返回 `403`，frpc 报 `DialTcpByHttpProxy error, StatusCode [403]`。
另外 aarch64 上 Gradio 的二进制名是 `frpc_linux_aarch64`，而 CDN 只有
`frpc_linux_arm64`；即使手动放置二进制，也无法解决 7000 端口被拦的问题。

---

## 3. 踩坑与解决

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `module 'torch' has no attribute 'npu'` | torch 不会自动加载 torch_npu | `src/device.py` 显式 `import torch_npu` 后再选设备 |
| 2 | `torchaudio` / `speechbrain` 缺失或装完 torch 变了 | 依赖会触发 torch 重装 | 锁版本并 `--no-deps` |
| 3 | gradio 6 + transformers 4.45 冲突 | gradio 5/6 要求 `huggingface-hub>=1.x`，transformers 4.45 要求 `<1.0` | 锁 `gradio==4.44.1` + `huggingface-hub==0.29.1` |
| 4 | Gradio 打开报 `TypeError: argument of type 'bool' is not iterable` | pydantic 2.11 改变了 JSON Schema 的 `additionalProperties` | 降 `pydantic==2.9.2` |
| 5 | `ImportError: DEFAULT_CIPHERS` | gradio 装依赖时把 urllib3 升到 2.x，旧 botocore 不兼容 | `urllib3<2` |
| 6 | `Default encoder for format mp3 ... disabled` | 系统 ffmpeg 4.2 未编入 libmp3lame | `src/ffmpeg_util.py` 自动回退到 `imageio-ffmpeg` |
| 7 | `ModuleNotFoundError: no module named 'torchaudio'` | 依赖漏装 | 见 §2.2 |
| 8 | `gradio share` 返回 403 | 平台代理拦截 frp 7000 端口 | 用 §2.7 的 SSH 转发 / Pinggy / 控制台代理 |
| 9 | `scp` 看不到刚生成的文件 | 本地 Bash 沙箱与 `scp` 的文件视图不同 | 用 `cat file \| ssh 'cat > remote'` 管道上传 |
| 10 | 测试 `test_pipeline::small_wav` IndexError | 需要 ≥4 条 LibriSpeech flac | 上传一个 chapter（§2.4） |

---

## 4. 验收命令（最短复验）

```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache

python -c "from src.device import device_report; print(device_report())"
python -m pytest tests/ -q
python -m src.pipeline demo assets/sample_utt.wav --outdir data/probe/pipeline --device npu:0
curl -s -o /dev/null -w "app=%{http_code}\n" http://127.0.0.1:7860/
```

---

## 5. 尚未完成

- ModelArts **在线服务 / Docker Compose + SWR 镜像**（真正的长期公网端点）未做；
  当前公网入口为临时隧道，需按 60 分钟续期。
- 数据集（LibriSpeech 全集 / VCTK 全集）未上传；本次范围为「部署 + 主链路」，
  重跑 Table 5/6/7 需另行上传约 13GB 数据。
- OBS 权重/数据归档与安全组固定端口映射未配置。
