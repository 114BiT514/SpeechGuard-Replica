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
| M1 环境/数据/仓库 | 🚧 进行中 |
| M2 声音匿名化核心 | ☐ |
| M3 匿名化效果验证 | ☐ |
| M4 内容保护 | ☐ |
| M5 权限组 + 端到端 | ☐ |
| M6 前端 + 可视化 | ☐ |
| M7 华为云部署 + 加分项 | ☐ |
| M8 报告 + 验收 | ☐ |
