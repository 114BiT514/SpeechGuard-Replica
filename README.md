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

## 快速开始

```bash
# 1. 环境（见 environment.yml；Windows 本机用 E:/Miniconda/envs/speechguard）
conda env create -f environment.yml && conda activate speechguard

# 2. 单元测试
python -m pytest tests/ -q          # 57 项

# 3. 端到端演示（protect → publish → listen → recover，产物含三态 MP3）
python -m src.pipeline demo assets/sample_utt.wav

# 4. 前端（Gradio 双角色：音频主人 / 听众；含"载入示例语音"按钮）
python -m app.main                  # http://127.0.0.1:7860
```

> 模型权重需先按"环境备忘"下载缓存（`data/pretrained/`、`data/hf-cache/`）。
> 指标的唯一权威口径：`experiments/results/main_table.md`（Table 5/6/7 样式）。

## 模型权重与数据声明（M8 完善）

- 大文件（预训练权重、数据集）存放于百度网盘：`【待填写链接】`
- 华为云 OBS 地址：`【待填写】`
- 华为云在线演示入口：`【待填写】`

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M1 环境/数据/仓库 | ✅ 环境就绪+冒烟通过；LibriSpeech/VCTK 就位；华为云 NPU 已验收；精读笔记已交付（`docs/精读笔记_公式表号对照.md`）；剩余：自录集人工审听（7 人已入库+转写辅助） |
| M2 声音匿名化核心 | ✅ 完成（framing/VAD/warping + 29 测试全绿 + 演示音频人工试听通过） |
| M3 匿名化效果验证 | ✅ **论文 Table 5 成功复现**（single 模式：EER 40.01%/WER 5.44% vs 论文 34.50%/4.80%），证据链见 `docs/M3_reproduction_findings.md`；**P1 补全**：Fig.9 五 n 矩阵复现、TitaNet-Large（论文选参 ASV）全集重做 38.66%≈ECAPA 40.01%（修正 24 句子集伪影，§4.1） |
| M4 内容保护 | ✅ **FNR 复现论文 Table 7**（margin=8：**98.57%** vs 论文 95.88%，60句/142敏感词，Conformer 评估；检测收紧=虚词过滤+date正则收窄，幸存2词均为NER误标普通名词）；全链路完成：CTC词时间戳（免MFA）+检测流水线（词表/模式/NER）+手动标注JSON+AES-CTR帧加密+**MP3适配（公式(5)+静音分发版）**；旧"FNR 75%"根因=插值时间戳失配，已解决；证据链见 `docs/M4_content_protection_findings.md`。自录集 7 人已入库+转写辅助（发音不标准、待人工审听，**不进主评估**） |
| M5 权限组 + 端到端 | ✅ 完成：CP-ABE 权限组（charm-crypto Windows 编译失败→按计划降级为对称密钥派生属性加密，四算法 API + DNF 访问结构，报告标注与配对 CP-ABE 差异）+ 三组典型权限（admin/collab/public）+ `pipeline.py` 四阶段端到端 CLI + **主表 Table 5/6/7 复现**（LS test-clean + VCTK 全达标）+ 基线对比四方法齐全+VoiceMask（Fig.10：SG raw 14.7% 最低 < McAdams 40.2 < Overo 49.7 < single 57.3 < VoiceMask† 62.2）。证据链 `docs/M5_permission_pipeline_findings.md` |
| M6 前端 + 可视化 | ✅ 完成：Gradio 双角色前端（主人 protect→人工增删→publish MP3 三态+保险库；听众三角色试听/恢复）+ 可视化素材全套（频谱四联图/时间轴/warping曲线/Fig.7/8/10）；**浏览器端到端全流程验证通过**，过程中定位并修复 2 个音频电平 bug。`docs/M6_frontend_findings.md` |
| M7 华为云部署 + 加分项 | ☐ |
| M8 报告 + 验收 | ☐ |

## 环境备忘（重要，后续里程碑开工前先读）

- **环境**：Miniconda 位于 `E:\Miniconda`（未加入 PATH，用全路径调用），环境名 `speechguard`（python 3.10）。调用方式：`E:/Miniconda/envs/speechguard/python.exe`；ffmpeg 已由 conda 装入该环境。
- **深度学习栈**：torch **2.8.0+cu128** + torchaudio 2.8.0（torchaudio 已停止更新、最高 2.8.0，必须与 torch 版本配对；RTX 4060 驱动 617.42 支持 cu128）。本地 GPU 推理已验证可用。
- **HuggingFace 访问**：hf-mirror 的 API 现返回 401，改走 Clash 代理直连官方源：
  `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 HF_HOME=E:/project/SpeechGuard-Replica/data/hf-cache`
- **模型替换说明（WavLM → wav2vec2）**：论文用 WavLM 做离线转写+日志（SD），但 `facebook/wavlm-large` 在 Hub 已被 gate（401）。改用 `facebook/wav2vec2-large-960h-lv60-self`。注意论文 Table 3 是**两列**：NER 列（BERT）Clean1/2/Noisy=5.00/4.68/10.31%，SD 列（diarization 误差）wav2vec2/HuBERT/WavLM Clean1=6.09/5.93/4.04%——**5.00% 是 BERT 的数字，不是 wav2vec2**（此前本备忘误引，已更正）。我们系统单说话人数据无 speaker-transition 可分，SD 列超出系统范围（书面结论见 `experiments/results/table3_error_rates.json`）；wav2vec2 替换论证改用其**转写 WER 实测**。
- **已缓存模型**：`data/pretrained/` 内 ECAPA-TDNN（85MB）、Conformer ASR（786MB，SpeechBrain COPY 策略落盘）；`data/hf-cache/` 内 BERT-NER（dbmdz）、wav2vec2-960h。
- **SpeechBrain on Windows 两个坑**：① 无符号链接特权 → 所有 `from_hparams` 必须传 `local_strategy=LocalStrategy.COPY_SKIP_CACHE`；② 反斜杠路径会被内部处理吃掉 → 传给 speechbrain 的文件路径一律 `replace("\\","/")`。参考 `scripts/smoke_test.py`。
- **数据集**：LibriSpeech test-clean（2620 条✓）/ dev-clean（2703 条✓）已解压至 `data/librispeech/LibriSpeech/`；VCTK 0.92（silence-trimmed FLAC 版，109 个说话人、88,328 条 flac + `txt/` 转写 + `speaker-info.txt`）解压于 `data/vctk/wav48_silence_trimmed/`（官方源直连仅 ~20KB/s，取自 hf-mirror 镜像 `duanyu027/vctk_0.92`；注意 `wav48_silence_trimmed/` 下的 `log.txt` 和 `s5/` 不是说话人，加载时过滤）。
- **云端/本地版本差异（代码须两端兼容）**：云端 NPU 环境是 torch 2.1.0 + torch_npu 2.1.0.post10 + **speechbrain 1.0.3**（详见验收报告），本地是 torch 2.8.0 + speechbrain 1.1.2。写代码时：① 不要用 speechbrain 1.0.3 没有的新 API；② 模型权重路径不硬编码（本地 `data/pretrained/`，云端 `pretrained_models/`），由配置传入；③ 云端补装依赖一律 `--no-deps`，防止 torch 被顶掉。
- **MFA**：本机无 WSL，且 MFA 不支持 Windows 原生 → M4 开工时先装 WSL2 + Ubuntu 建 mfa 环境；备选方案 whisperX 词级时间戳。
- **Clash 代理**：`127.0.0.1:7890`；国际源（OpenSLR、HF）国内直连极慢，务必走代理；curl 走代理需加 `--ssl-no-revoke`（Schannel 证书吊销检查会失败）。

> M1 云端部分已完成（2026-10-23）：华为云 ModelArts 昇腾 910B4 环境、ECAPA 与 Conformer 的 NPU 推理均已验证通过，详见 [M1 华为云 NPU 验收报告](docs/M1_华为云NPU验收报告.md)。
