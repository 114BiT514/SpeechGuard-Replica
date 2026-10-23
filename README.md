# SpeechGuard-Replica：可恢复、可定制的语音隐私保护系统复现

复现论文 **SpeechGuard: Recoverable and Customizable Speech Privacy Protection**（USENIX Security 2025）。

系统在无需任何训练的前提下，对语音同时提供：
- **声学隐私**：多参数频域扭曲（warping）抹去声纹，持密钥者可精确恢复；
- **内容隐私**：敏感文本自动检测 + 帧级 AES-CTR 加密（适配 MP3 有损压缩）；
- **权限组**：不同听众凭权限恢复不同程度的声纹与内容（CP-ABE 密钥分发）。

## 仓库结构

```
SpeechGuard-Replica/
├── src/
│   ├── preprocess/   # 36ms 分帧、动态阈值 VAD、STFT
│   ├── acoustic/     # 多参数 warping：protect / recover / dist
│   ├── content/      # 转写、NER、MFA 对齐、AES 加密、MP3 适配
│   ├── auth/         # CP-ABE 权限组、密钥分发
│   ├── baselines/    # 对比基线（单参数分段线性 / McAdams / Overo）
│   ├── pipeline.py   # 端到端：protect → publish → listen → recover
│   └── device.py     # 昇腾 NPU / CUDA / CPU 自动选择
├── experiments/      # 复现实验脚本 + results/ 图表
├── app/              # Gradio 前端（音频主人 / 听众双角色）
├── deploy/           # 华为云 ModelArts + 昇腾部署
├── docs/             # 报告、验收 PPT 大纲
└── assets/           # demo 音频与截图
```

## 快速开始（M8 完善）

```bash
# 1. 环境（见 environment.yml）
conda env create -f environment.yml && conda activate speechguard

# 2. 端到端演示（M5 完成后可用）
python -m src.pipeline demo assets/demo.wav

# 3. 前端（M6 完成后可用）
python -m app.main
```

## 模型权重与数据声明（M8 完善）

- 大文件（预训练权重、数据集）存放于百度网盘：`【待填写链接】`
- 华为云 OBS 地址：`【待填写】`
- 华为云在线演示入口：`【待填写】`

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M1 环境/数据/仓库 | ✅ 环境就绪+冒烟通过；LibriSpeech/VCTK 就位；剩余：华为云开卡、自录集、精读笔记 |
| M2 声音匿名化核心 | ☐ |
| M3 匿名化效果验证 | ☐ |
| M4 内容保护 | ☐ |
| M5 权限组 + 端到端 | ☐ |
| M6 前端 + 可视化 | ☐ |
| M7 华为云部署 + 加分项 | ☐ |
| M8 报告 + 验收 | ☐ |

## 环境备忘（重要，后续里程碑开工前先读）

- **环境**：Miniconda 位于 `E:\Miniconda`（未加入 PATH，用全路径调用），环境名 `speechguard`（python 3.10）。调用方式：`E:/Miniconda/envs/speechguard/python.exe`；ffmpeg 已由 conda 装入该环境。
- **深度学习栈**：torch **2.8.0+cu128** + torchaudio 2.8.0（torchaudio 已停止更新、最高 2.8.0，必须与 torch 版本配对；RTX 4060 驱动 617.42 支持 cu128）。本地 GPU 推理已验证可用。
- **HuggingFace 访问**：hf-mirror 的 API 现返回 401，改走 Clash 代理直连官方源：
  `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 HF_HOME=E:/project/SpeechGuard-Replica/data/hf-cache`
- **模型替换说明（WavLM → wav2vec2）**：论文用 WavLM 做离线转写，但 `facebook/wavlm-large` 在 Hub 已被 gate（401）。改用 `facebook/wav2vec2-large-960h-lv60-self`——论文 Table 3 本身就把 wav2vec2 列为对比的 SD 模型（clean 错误率 5.00% vs WavLM 4.04%），报告中如实注明即可。
- **已缓存模型**：`data/pretrained/` 内 ECAPA-TDNN（85MB）、Conformer ASR（786MB，SpeechBrain COPY 策略落盘）；`data/hf-cache/` 内 BERT-NER（dbmdz）、wav2vec2-960h。
- **SpeechBrain on Windows 两个坑**：① 无符号链接特权 → 所有 `from_hparams` 必须传 `local_strategy=LocalStrategy.COPY_SKIP_CACHE`；② 反斜杠路径会被内部处理吃掉 → 传给 speechbrain 的文件路径一律 `replace("\\","/")`。参考 `scripts/smoke_test.py`。
- **数据集**：LibriSpeech test-clean（2620 条✓）/ dev-clean（2703 条✓）已解压至 `data/librispeech/LibriSpeech/`；VCTK 0.92（silence-trimmed FLAC 版，109 个说话人、88,328 条 flac + `txt/` 转写 + `speaker-info.txt`）解压于 `data/vctk/wav48_silence_trimmed/`（官方源直连仅 ~20KB/s，取自 hf-mirror 镜像 `duanyu027/vctk_0.92`；注意 `wav48_silence_trimmed/` 下的 `log.txt` 和 `s5/` 不是说话人，加载时过滤）。
- **MFA**：本机无 WSL，且 MFA 不支持 Windows 原生 → M4 开工时先装 WSL2 + Ubuntu 建 mfa 环境；备选方案 whisperX 词级时间戳。
- **Clash 代理**：`127.0.0.1:7890`；国际源（OpenSLR、HF）国内直连极慢，务必走代理；curl 走代理需加 `--ssl-no-revoke`（Schannel 证书吊销检查会失败）。
