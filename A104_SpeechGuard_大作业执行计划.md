# A104《SpeechGuard》复现与探索 —— 大作业执行计划

> 用途：本文档是整个大作业的"施工总纲"。每次开工时把对应里程碑（M1–M9）的章节交给 AI agent 执行，按 DoD（完成标准）验收后再进入下一阶段。
> 选题：A104 SpeechGuard: Recoverable and Customizable Speech Privacy Protection（USENIX Security 2025）
> 论文：全文 PDF 已下载在 `C:\Users\33588\Downloads\speechguard.pdf`（另有 prepub 版与官方 Slides，见文末资源清单）
> 关键时间：**2026-12-22 线上验收**，报告在验收通过后 5 个工作日内提交。

---

## 0. 评分对照表（一切工作围绕这张表展开）

| 评分项 | 分值 | 对应产出 |
|---|---|---|
| 项目报告 | 30 | `docs/report`：①论文背景与核心问题 ②方法理解与关键模块说明 ③复现实验设计与结果 ④结果分析、局限与思考 |
| 接入系统前端 | 10 | `app/`：Gradio 双角色界面（音频主人 / 听众） |
| 论文核心方法正确复现 | 25 | `src/`：warping 匿名化 + 敏感文本加密 + 权限组恢复，端到端可跑 |
| 实验结果验证 | 10 | `experiments/`：EER / WER / FNR / RTC 四指标复现表 |
| 案例可视化与结果分析 | 10 | 频谱图对比、恢复效果波形、四指标图表 + 文字分析 |
| 华为云部署 | 10 | `deploy/`：整套系统部署到华为云 ModelArts（昇腾 NPU 实例）并可公网访问 |
| README 文档 | 5 | `README.md`：环境、运行步骤、网盘权重链接声明 |
| **加分项（可选冲分）** | **15** | 见 M7：问题发现(+3) → 实验验证(+4) → 原因分析(+3) → 改进方案(+5) |
| 验收汇报 | — | 9 分钟 PPT，每人轮流讲，准备"问题→方法→结果"式提问 |

行政事项（勿忘）：组长/组员互评打分问卷；每人贡献简述（50 字内）；项目传 GitHub 且保证可运行，大文件（模型权重、数据集）传百度网盘并在 README 声明链接；提交命名 `组员1姓名学号后4位_组员2姓名学号后4位_SpeechGuard语音隐私保护复现`，发至助教邮箱 2325557558@qq.com。

**组队与分工（2 人组，工作量对半，里程碑 Owner 见各节标题）**：
- **成员 A（建议组长）**：声学线——M2 warping 核心、M3 匿名化验证、M5 评估矩阵、M7 加分项实验；报告"背景/复现实验/分析思考"（①③④）主笔。
- **成员 B**：工程线——M4 内容保护、M6 前端、M7 华为云部署、README；报告"方法与关键模块"（②）主笔 + 验收 PPT。
- **共同**：M1 环境、M5 权限组与整合、M8 报告互审与演练、M9 验收（每人轮讲约 4.5 分钟）。
- 两人都要能跑通全流程：互相 review 对方模块的代码与实验数字——验收提问不分模块，答不上来会扣印象分。

---

## 1. 论文技术拆解（复现的"图纸"，均已在论文中核实）

SpeechGuard 是一个**全程无需训练**的语音隐私保护系统，全部是推理/信号处理/密码学操作。论文用 P100 16GB 完成所有推理，算力门槛很低。系统分四个模块：

### 1.1 预处理（§4.1）
- 输入 16kHz 16bit 单声道 WAV，按 **36ms 帧**处理。
- **动态阈值 VAD**（公式(1)）区分有声/无声帧并分段，参数论文已给全：`T=1, at=0.7, ai=0.25, δ=1.99, Φ(-1)=max(|X_k(e^jω)|)+δ`。
- 对每帧做 STFT。

### 1.2 声音匿名化（§4.2，核心）
- **多参数分段线性频域扭曲**（公式(3)）：随机参数对数 `n∈[1,10]`，参数集 `α={α1..αn}, β={β1..βn}`，边界 `α0=β0=0, α_{n+1}=β_{n+1}=π`，逐段（segment）使用独立参数集 `s={α,β}`。
- **逆函数零成本**：`m⁻¹(ω,α,β) = m(ω,β,α)`，即 α、β 互换再查表一次——这就是"可恢复"的来源。
- **扭曲强度** dist 定义为扭曲曲线与恒等曲线之间的面积（公式(4)），论文在 LibriSpeech dev-clean 上扫参后限定 **dist∈(0.5,0.6)**：此时 ASV 错误率 80.18%（听不出是谁），ASR 错误率仅 3.58%（内容照样能听懂）。选参实验用的模型：ASR 用 SpeechBrain Transformer，ASV 用 NVIDIA TitaNet-Large。
- 对比基线：单参数分段线性（公式(2)）、McAdams、VoiceMask、Overo（双参数）。

### 1.3 敏感文本保护（§4.3）
- 保护单位是"词"；敏感词表来源：GDPR 类预定义词表 + 用户自定义词表 + 正则/NER 模式（电话号码、地址等）；词表在录音前本地存 hash 表。
- **自动检测流水线**：离线语音转写 + 说话人分段（WavLM）→ **MFA 强制对齐**拿到每个词的起止时间 → 与敏感词表/模式匹配定位。
- **手动标注**界面：录音后增删敏感词（弥补离线模型 4–10% 的转写错误率）。
- **AES-CTR 加密**：对敏感词覆盖的帧区间直接加密；c 个加密对象各自有密钥 `W={w1..wc}`。
- **MP3 有损压缩适配**（公式(5)）：由敏感词时间戳 `ts,te` 换算 MP3 帧号 `fs=⌊ts/Td⌋, fe=⌈te/Td⌉`，只加密数据帧体、跳过 tag 帧与帧头；无权限监听时把加密帧替换为静音帧（论文自带简易 decoder）。

### 1.4 权限组与恢复（§4.4）
- 声音参数集 `S={s1..sv}` 与加密密钥 `W={w1..wc}` 是两类"资产"。
- 听众划分权限组 `G={G1..Gg}`，每组分配子集 `Mi={Si,Wi}` → 不同人听到/恢复不同程度的声纹与内容。
- **密钥分发用 CP-ABE（属性基加密）**：Setup/Encrypt/KeyGen/Decrypt 四算法，把权限组可见范围 `Mi` 作为消息、访问结构 `Ai` 嵌入密文。

### 1.5 评估协议（§5，照抄即可复现）
| 指标 | 含义 | 工具 |
|---|---|---|
| EER | 说话人验证等错误率，越高越匿名（0.5 最优） | ECAPA-TDNN（VoxCeleb 训练）+ 余弦相似度 |
| WER | 词错误率，越低音质/可懂度越好 | Conformer ASR（LibriSpeech 960h 训练） |
| FNR | 敏感词加密后 ASR 仍识别出的比例，越高越私密 | 同上 ASR |
| RTC | 处理时长/音频时长，越低越实时 | 计时 |

- 数据集：LibriSpeech test-clean（40 人/2620 条）、VCTK（109 人/8163 条）、自录集（论文 18 人 1300 条，含噪声）。
- 关键复现数字：LS test-clean 上 EER 原始 1.47% → 保护后 34.50% → reducing 攻击后 36.77%；WER 原始 2.46% → 4.80% → 18.39%；10000 次 reducing 攻击成功率 15.87%，n 越大越抗攻击（论文 Fig.8/9/10、Table 2/5/6）。
- dist 选参实验在 dev-clean 上做：n=1，扫 dist∈{0.0..1.0} 十个区间，画 ASR/ASV 错误率曲线（论文 Fig.7、Table 2 的 EER/WER 矩阵）。

---

## 2. 代码仓库结构（M1 就按这个建）

```
SpeechGuard-Replica/
├── README.md                  # M8：环境、一键运行、网盘链接声明
├── environment.yml / requirements.txt
├── assets/                    # demo 音频、截图（进 git）
├── data/                      # 数据集（.gitignore，大文件走网盘）
├── src/
│   ├── preprocess/            # framing.py, vad.py, stft.py
│   ├── acoustic/              # warping.py: protect/recover/dist/n参数生成
│   ├── content/               # transcribe.py, ner.py, align.py, encrypt.py, mp3_crypto.py
│   ├── auth/                  # cp_abe.py, permission.py
│   ├── pipeline.py            # 端到端：protect → publish → listen → recover
│   └── device.py              # 昇腾 NPU / CUDA / CPU 设备自动选择（torch_npu 适配）
├── experiments/               # exp1_dist.py ... exp5_bonus.py + results/（图表）
├── app/                       # Gradio 前端（owner / listener 两个 Tab）
├── deploy/                    # ModelArts + 昇腾部署：打包推理服务/容器、OBS 拉取脚本、说明
└── docs/                      # report.md、验收PPT大纲、提问预案
```

> 环境提示：MFA 与若干音频库在 Linux 下最省事，**强烈建议全程用 WSL2 + conda**；本机网络访问 GitHub 不稳定，克隆/装包优先走代理或 gitee 镜像。深度模型的推理开发在本地 CPU 就能做（信号处理与加密模块本来就与框架无关），**NPU 适配统一放 ModelArts 上做**——torch_npu 仅支持 Linux，且 ModelArts 镜像已预装 CANN + torch_npu，本地 Windows 不要折腾。

---

## 3. 里程碑计划（今天 2026-10-22 → 验收 2026-12-22，共 61 天）

> 每个里程碑都给了：目标 / 交给 agent 的任务清单 / 交付物 / DoD（自检通过才算完）。标 ★ 的是直接对应分数的核心路径，宁可延期也不能砍。

---

### M1 ★ 开题与环境（10.22–10.26，5天）｜Owner：共同

**目标**：环境可用、论文精读完成、仓库与数据就位。

任务：
1. 按上文分工块确认两人职责（声学线 / 内容与工程线），建立每周一次的进度同步（对齐 DoD 与 blocker）。
2. 精读论文 §4、§5（设计+实验），把本文档 §1 的拆解逐条在论文里找到对应公式和表号，做一份"公式-表号"对照笔记（报告素材）。
3. 建仓库（按 §2 结构）、git 初始化、写 `.gitignore`。
4. conda 建环境（python 3.10），装：`torch`、`speechbrain`（ECAPA-TDNN、Conformer、预训练模型自动下载）、`pydub`、`pydantic`、`ffmpeg`、`nemo_toolkit`（TitaNet-Large，仅选参实验用）、MFA（`conda install -c conda-forge montreal-forced-aligner`）+ 英文声学模型与字典、`pycryptodome`（AES-CTR）、`charm-crypto`（CP-ABE，装不上先跳过，见风险表）。
5. 下载数据集：LibriSpeech test-clean + dev-clean（选参实验用）、VCTK（VCTK 有损坏文件清单，网上找 skip 列表）。
6. 下载预训练权重：SpeechBrain 的 ECAPA-TDNN（`speechbrain/spkrec-ecapa-voxceleb`）与 Conformer ASR（`speechbrain/asr-conformer-transformerlm-librispeech`）、WavLM、BERT-base NER（`dbmdz/bert-large-cased-finetuned-conll03-english`）。全部走 HuggingFace/网盘缓存，README 里登记。
7. 录制自录集：组员两人各录 20 分钟左右朗读（英文为主，另录少量中文做扩展讨论），模拟论文的 volunteers 集及键盘/空调底噪。
8. **华为云开卡（部署前置，务必提前，不要拖到 M7）**：实名认证 → 开通 ModelArts 与 OBS → 创建一个 **Ascend（910B/310P）Notebook** 实例 → 验证 `import torch, torch_npu; torch.npu.is_available()` → 把 ECAPA-TDNN 在 NPU 上跑通一条音频推理。CANN 与 torch_npu 在 ModelArts 预装镜像里已带好，本地无需（也无法在 Windows 上）安装。

**交付物**：仓库骨架、environment.yml、数据与权重就位、精读笔记。
**DoD**：`python -c "import speechbrain, torch; print('ok')"` 通过；ECAPA 与 Conformer 各跑通一条音频推理；LS test-clean 能列出 2620 条。

---

### M2 ★ 预处理 + 声音匿名化核心（10.27–11.8，13天）—— 全项目技术核心｜Owner：A 主导，B 协作

**目标**：36ms 分帧 + VAD 分段 + 多参数 warping 的"保护→恢复"闭环，能算 dist、能出频谱图。

任务（按顺序，每个都是独立可交给 agent 的包）：
1. `framing.py`：36ms 帧切分（帧长 576 采样点 @16kHz），支持帧级索引与时间戳换算。
2. `vad.py`：公式(1) 动态阈值 VAD，参数照抄 §4.1（T=1, at=0.7, ai=0.25, δ=1.99）；输出有声段列表（起止帧/时间），与静音段区分；画分段示意图（论文 Fig.3 风格）。
3. `stft.py`：STFT/ISTFT 封装（注意窗长与 36ms 对齐、相位保留，恢复后要 ISTFT 回时域）。
4. `warping.py`：
   - 实现公式(2) 单参数分段线性（基线用）；
   - 实现公式(3) 多参数 warping `m(ω,α,β)` 与逆函数 `m⁻¹(ω,α,β)=m(ω,β,α)`；
   - 实现 dist 计算（公式(4)，数值积分即可）；
   - `gen_params(n, dist_range)`：随机生成满足 `dist∈(0.5,0.6)`、单调约束的 α/β 参数对（拒绝采样即可）；
   - `protect(wav, n=None)`：逐段独立参数集扭曲 + 返回参数档案 `S`；`recover(wav_protected, S)`：逆扭曲。
5. 频谱可视化脚本：扭曲前后 spectrogram 对比（论文 Fig.6 风格）。
6. 单元测试：`recover(protect(x)) ≈ x`（浮点误差内）；dist 单调性；参数边界条件（α0<α1<…<α_{n+1}）。

**交付物**：`src/preprocess` + `src/acoustic` + tests + 演示音频对（原始/保护/恢复）。
**DoD**：拿一条 LibriSpeech 音频，保护→恢复后与人耳听原音频几乎一致；保护版人耳听不出原说话人但内容可懂；`pytest` 全绿。

---

### M3 ★ 匿名化效果验证 + dist 选参复现（11.9–11.15，7天）｜Owner：A

**目标**：复现论文 Fig.7 / Table 2 的选参依据，并给出保护后的 EER/WER 初步数字。

任务：
1. `exp1_dist_scan.py`：dev-clean 子集（如 10 人 × 50 条，控制跑批时间），n=1，扫 dist 十个区间，每个区间测 ASR 错误率（SpeechBrain Transformer 或轻量 Whisper 对齐说明）与 ASV 错误率（TitaNet-Large 或 ECAPA，报告中注明与论文的模型差异），画论文 Fig.7 同款双曲线图。
2. `eval_asv.py`：ECAPA-TDNN + 余弦相似度，实现 EER 计算（用 `pyeer` 或手写 FAR/FRR 扫描）。在 LS test-clean 上跑：原始 EER ≈ 1.47%（论文 Table 5），保护后 EER 落到 30–40% 区间即视为方向正确。
3. `eval_asr.py`：Conformer ASR 跑 WER（`jiwer` 算 WER）；原始 WER ≈ 2.46% 对照。
4. `exp2_reducing_attack.py`：按 §5.2 实现 reducing 攻击仿真——攻击者已知 n∈[1,10]、dist∈(0.5,0.6)，随机生成参数做逆扭曲 10000 次，统计"攻击后 dist < 保护后 dist"的比例（论文基准 15.87%）；画不同固定 n 的 ECDF 与成功率图（Fig.8）。

**交付物**：`experiments/` 前三个脚本 + results/ 图表 + 一页初步数字表。
**DoD**：EER 保护后 ≥30%、WER 保护后 ≤6%、攻击成功率 10–20% 区间；图表可直接进报告。

---

### M4 ★ 内容保护：检测 + 加密 + MP3 适配（11.16–11.24，9天）｜Owner：B

**目标**：敏感词"自动检测 → 定位 → 帧级加密 → 有损格式安全"全链路。

任务：
1. `transcribe.py`：WavLM 离线转写（HF `facebook/wavlm-large` + CTC 或用 wav2vec2 ASR 头），输出带词级时间的候选转写。
2. `align.py`：MFA 强制对齐，得到每个词的精确起止时间戳（论文用 MFA[47]）；对 36ms 帧网格做帧索引映射。
3. `ner.py`：BERT NER 识别人名/地点/机构等模式；正则模式匹配电话号码、地址、日期。
4. `sensitive_lists.py`：预定义敏感词表（按 GDPR 类别起一套英文词表）+ 自定义词表加载 + hash 表存储。
5. `encrypt.py`：AES-CTR；`encrypt_frames(wav, spans, keys)` 对敏感词覆盖的帧区间加密；每个加密对象独立密钥 `W`。
6. `mp3_crypto.py`：公式(5) 帧号换算；只加密数据帧体、保留 tag 帧与帧头；无权限解码时替换静音帧（论文 §4.3.3 的简易 decoder）。用 `pydub`/`lameenc` 生成 MP3 并验证：加密后的 MP3 在常见播放器可正常播放未加密部分。
7. 手动标注数据结构：JSON 记录敏感词 span 增删（前端在 M6 消费）。
8. `eval_fnr.py`：加密后跑 ASR，统计敏感词 FNR（论文基准：保护后敏感词基本不可识别）。

**交付物**：`src/content` 全套 + 加密前后 MP3 样例 + FNR 初步数字。
**DoD**：一句话含敏感词的音频，加密后 ASR 转不出该词、人耳听是噪声/静音；未加密部分播放无损；恢复端凭密钥解回。

---

### M5 ★ 权限组 + 端到端管线 + 评估矩阵（11.25–12.1，7天）｜Owner：共同（A 评估矩阵 / B 权限组）

**目标**：CP-ABE 权限分发跑通，系统四模块串成完整输入输出流程，复现主表。

任务：
1. `cp_abe.py`：优先用 `charm-crypto` 的 CP-ABE 方案实现论文四算法（Setup/Encrypt/KeyGen/Decrypt）；**若装库受阻**，降级方案：AES 密钥按权限组加密分发（RSA/混合加密），报告中如实说明与论文 CP-ABE 的差异——这是允许的"方法等价实现"，但要在验收时说清楚。
2. `permission.py`：权限组数据结构 `G/Mi={Si,Wi}`；owner 给三组典型权限：管理员（全恢复）/ 合作者（恢复声音不恢复敏感词）/ 公开（全保护）。
3. `pipeline.py`：四阶段端到端——`protect`（录音实时保护）→ `publish`（人工增删 + MP3 发布）→ `listen`（无密钥播放）→ `recover`（按权限解密+逆扭曲）。CLI 可一键演示。
4. `exp3_main_table.py`：三个数据集（LS test-clean / VCTK / 自录）跑全系统，产出论文 Table 5 同款表（Ori./Pro./Att. 三态的 EER、WER）+ FNR + RTC（Table 6 风格，含 VoiceMask/McAdams/Overo 基线的 RTC 对比可后置到 M7）。
5. 基线实现：`baselines/` 里实现单参数分段线性、McAdams 系数变换（公式简单）；Overo 双参数扭曲按其论文公式实现；对比它们的攻击成功率与 EER（论文 Fig.10/11）。

**交付物**：可一键运行的端到端 CLI + 主结果表（论文复现数字对照）。
**DoD**：`python -m src.pipeline demo assets/demo.wav` 走完四阶段；三数据集数字进表；与论文数字的偏差在报告中给出解释（模型版本差异等）。

---

### M6 ★ 前端 + 案例可视化（12.2–12.8，7天）｜Owner：B

**目标**：Gradio 双角色界面（10分）+ 报告用的可视化素材（10分）。

任务：
1. `app/`（Gradio 多 Tab）：
   - **音频主人端**：录音/上传 → 实时保护 → 自动检测结果高亮列表 → 手动增删敏感词 → 选择保护强度 dist → 发布（下载 MP3）；
   - **听众端**：选择身份（权限组）→ 在线试听（有权限处自动解密）→ 一键恢复下载；
   - 每步展示中间产物（频谱图、敏感词时间轴）。
2. 可视化素材脚本：扭曲前后/攻击后频谱三联图（Fig.6 风格）、权限组演示动图或录屏、EER/WER 随 dist 曲线（Fig.7）、攻击成功率柱状图（Fig.10）、n 的 ECDF（Fig.8）。
3. 案例分析文案：每个图配 3–5 句结论（直接是报告和 PPT 素材）。

**交付物**：`app/` 可本地跑 + `assets/` 全套图 + 案例分析文字。
**DoD**：浏览器里走完"主人发布 → 三种听众分别听到不同版本 → 管理员完整恢复"；所有报告配图成稿。

---

### M7 ★ 华为云部署 + 加分项实验（12.9–12.16，8天）｜Owner：共同（B 部署 / A 加分项）

**目标**：华为云 10 分落袋；15 分加分项按四层做满。

部署任务（`deploy/`，华为生态硬要求：华为云平台 + 昇腾算力 + 华为工具链）：
1. **先和老师确认"华为生态"的口径**：默认口径 = ModelArts 平台 + 昇腾 NPU（910B/310P 实例）+ torch_npu（PyTorch 昇腾适配层）+ OBS 存储；若老师明确要求 MindSpore 框架，则把 BERT NER 等单点模型换成 MindSpore Hub 等价实现——信号处理、加密、权限组模块与框架无关不受影响，主链路不必重写。结论写进报告。
2. 代码适配：`src/device.py` 自动选择 `npu/cuda/cpu`；四个深度模型（ECAPA、Conformer、WavLM、BERT）逐个 `to('npu')` 全量迁移验证，记录不支持的算子及 CPU fallback（M1 已完成 ECAPA 首轮验证，这里补齐全部）。
3. 权重与数据全部存 OBS，部署实例通过 obs-sdk / moxing 拉取；`deploy/README` 写清从零复现部署的步骤（建实例→拉权重→启动→访问）。
4. 部署形态（二选一）：**A（推荐）** ModelArts 在线服务——把整套 pipeline（含 Gradio 界面）打包成推理服务部署到昇腾实例；**B** 昇腾 Notebook/ECS + Docker Compose 跑 Gradio，安全组放行端口。验收时用公网 URL 现场演示。
5. 前端加"云端模式"入口：与 M6 的 Gradio 复用同一套代码，本地/云端仅切换后端地址。

加分项实验（`experiments/exp4_bonus*.py`，按 PPT 四层递进，全部推理级、成本低）：
1. **问题发现 (+3)**：真实信道下的保护/恢复鲁棒性——微信语音（Silk/AMR 转码模拟：低码率 mp3/amr 重编码）、8k 重采样、翻录（扬声器→麦克风，可用数据增强模拟）、加性噪声，各自对"恢复质量（EER 回升/WER 变化）"和"敏感词加密定位准确性（帧号漂移）"的影响案例。
2. **实验验证 (+4)**：控制变量——固定其余条件，扫描码率 {32/64/128/192kbps}、SNR {0/5/10/20dB}、重采样率 {8/16/22k}，画"信道条件 vs 恢复后 EER/WER"曲线族，证明问题普遍存在非偶发。
3. **机理分析 (+3)**：解释为什么——如 MP3 帧边界与 36ms 保护帧边界错位导致加密区间渗漏；低码率量化把高频扭曲"抹平"导致逆扭曲失配；噪声掩盖了 warping 留下的高频特征等。结合频谱图给出证据。
4. **改进方案 (+5)**：至少做一个并验证——如 (a) 信道感知的参数选择（在 dist 相同下选择低频段扭曲权重更高的参数集）；(b) 加密区间外扩 + 边界余量；(c) 恢复端先做信道均衡再逆扭曲。对照组：原方法在信道 C 失效 → 改进方法在 C 下明显改善。**不必全部解决，前三层做扎实已有 10 分。**

**交付物**：可公网访问的云端系统 + 加分项实验四件套（案例/曲线/机理图/改进对照表）。
**DoD**：手机 4G 打开云端链接完整演示；加分项每层有图有数有结论。

---

### M8 ★ 报告 + README + 验收材料（12.17–12.21，5天）｜Owner：共同

任务：
1. `docs/report`（对照 30 分四块）：
   - 背景与核心问题：云上语音的声纹/内容双重隐私，现有方法不可恢复、不可定制（引用论文 Table 1）；
   - 方法与关键模块：本文档 §1 的拆解直接改写，配模块图与公式；
   - 复现实验设计与结果：M3/M5 的实验协议 + 主表 + 与论文数字的偏差分析（写明所用模型版本差异，如 ECAPA vs TitaNet）；
   - 结果分析、局限与思考：偏差原因、系统局限（英语语料、单说话人假设、离线模型错误率）、加分项的发现。
2. `README.md`：环境/数据/权重获取、三条命令跑通（测试、端到端 demo、前端）、**百度网盘链接声明**、华为云入口、目录结构、复现数字一览。
3. 验收 PPT（9 分钟，每人轮流 4–5 分钟）：选题动机 → 系统总览 → 复现结果 → 前端/云端演示（录屏兜底）→ 加分项 → 分工与贡献。
4. 提问预案（"问题→方法→结果"句式）：预演 10 个可能问题，如"为什么选 warping 而不是神经声码器""EER 为什么到不了 0.5""MP3 加密为什么不破坏播放""与 Overo 的本质区别""CP-ABE 是否必要"。
5. GitHub 仓库整理：删调试残留、加 demo GIF、确认 clone 后可运行；打 tag。

**DoD**：报告四块齐全且数字与 experiments 输出一致；README 按助教"clone→跑通"视角自查通过；PPT 排练一遍不超时。

---

### M9 验收与提交（12.22 起）

1. 12.22 线上验收：屏幕共享演示云端系统 + 汇报 + 答问。
2. 验收通过后 5 个工作日内：报告 PDF 发 2325557558@qq.com，命名 `组员1姓名学号后4位_组员2姓名学号后4位_SpeechGuard语音隐私保护复现`。
3. 完成互评问卷与 50 字贡献简述。

---

## 4. 风险与预案（开工前读一遍）

| 风险 | 影响 | 预案 |
|---|---|---|
| MFA 在 Windows 装不上 | 卡住 M4 | 全程 WSL2/conda；实在不行用 whisperX 的词级时间戳替代并在报告说明 |
| `charm-crypto` 编译失败 | 卡住 M5 | 降级为"AES 密钥 + 权限组加密分发"，报告注明与论文 CP-ABE 的实现差异 |
| GitHub 访问不稳定 | 依赖拉取 | 代理/镜像站；权重用 HF 镜像或提前下好放 OBS/网盘 |
| 复现数字与论文对不上 | 复现 25 分 | 论文用 TitaNet 做 ASV 选参、ECAPA 做评估；模型版本差异会导致偏差——**如实报告偏差并解释**，方向性正确（EER 大幅上升、WER 基本不变）即可 |
| VCTK 部分文件损坏 | 数据问题 | 使用社区维护的 skip 文件清单 |
| 中文自录音频在英文 ASV/ASR 上失真 | 自录集评估 | 主实验用英文自录；中文部分只做定性展示并说明评估模型是英文的 |
| torch_npu 算子不兼容 | 某模型在 NPU 上报错 | M1/M7 逐模型迁移验证；不兼容算子回退 CPU 执行（信号处理与加密模块本来就在 CPU，不受影响）；必要时换等价模型 |
| ModelArts 配额/计费 | 进度与预算 | Notebook 用完即停、按量付费；提前实名认证并申请配额；"本地 CPU 开发 + 云端 NPU 验证"节奏分离，NPU 只跑迁移验证和最终部署 |
| 老师硬性要求 MindSpore 框架 | 改造范围 | M1 先确认口径；仅把 NER 等单点换 MindSpore Hub 模型，主链路 torch_npu 保持不动，报告说明工程权衡 |
| 12 月初进度落后 | 验收 | 非开源项目**不要求完整实现、按工作量给分**：优先保 M2/M3/M5/M6 核心闭环，加分项和基线对比可裁剪；遇到不可行处，邮件论文原作者（邮箱在 PDF 首页）并同步助教 |

---

## 5. 怎么把本文档喂给 agent（使用说明）

1. **每次会话开场**，把这句话+对应里程碑章节发给 agent：
   > 请阅读 `C:\Users\33588\Desktop\A104_SpeechGuard_大作业执行计划.md` 的 §1 技术拆解与 M{n} 里程碑，完成其中任务列表。约束：代码风格与仓库现有代码一致；每个模块写单元测试；完成后按 DoD 自检并逐条报告证据。
2. **一次只派一个任务包**（如"只做 M2 的 vad.py"），产出立即 `pytest` + 人工抽听，通过再合并。不要一次让 agent 写完整系统。
3. **给 agent 的固定上下文**：论文 PDF 路径、本文档路径、仓库根目录。涉及公式时让 agent 先复述公式再写代码（公式(1)(3)(4)(5) 已在 §1 给全）。
4. **实验脚本**单独派发：要求 agent 产出"脚本 + 图表 + 一段结论文字"，三件套齐才算完。
5. **每个里程碑收尾**，让 agent 输出"里程碑报告"：完成了什么、DoD 逐条证据、遗留问题——这份报告直接攒成最终报告的素材。

---

## 6. 资源清单

- 论文 PDF（正式版，已下载）：`C:\Users\33588\Downloads\speechguard.pdf`
- 论文页面（含 prepub PDF、官方 Slides）：https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-jingmiao
- arXiv 版：https://arxiv.org/abs/2502.06139
- 选题信息（难度 11 = 方法实现3 + 环境修复2 + 依赖整合3 + 复现材料3，部署难度：困难）：`语音识别与合成大作业2026.xlsx` 非开源表第 21 行
- 预训练模型：ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb`；Conformer ASR `speechbrain/asr-conformer-transformerlm-librispeech`；TitaNet-Large（NVIDIA NeMo）；WavLM `facebook/wavlm-large`；BERT NER `dbmdz/bert-large-cased-finetuned-conll03-english`
- 工具：MFA（montreal-forced-aligner）、pycryptodome、charm-crypto、pydub/ffmpeg、jiwer、pyeer
- 华为生态：ModelArts（https://www.huaweicloud.com/product/modelarts.html）；昇腾 PyTorch 适配层 torch_npu（https://gitee.com/ascend/pytorch）；昇腾社区（https://www.hiascend.com/）；OBS 对象存储文档
- 数据集：LibriSpeech（test-clean + dev-clean）、VCTK、自录集
- 加分项参考：微信语音 Silk/AMR 转码、MP3 码率扫描、SNR 扫描

---

*计划版本：v1.1（2026-10-22 制定；同日修订：明确 2 人分工与里程碑 Owner、部署要求改为华为云 + 昇腾生态）。每完成一个里程碑更新一次本文档的勾选状态。*

**进度勾选**：M1 ☐ / M2 ☐ / M3 ☐ / M4 ☐ / M5 ☐ / M6 ☐ / M7 ☐ / M8 ☐ / M9 ☐
