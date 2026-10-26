# M4 结论：内容保护全链路完成，FNR 成功复现论文 Table 7

> 复跑：`E:/Miniconda/envs/speechguard/python.exe experiments/eval_fnr.py`
> 结果：`experiments/results/fnr_eval.json`（转写缓存 `fnr_hyp_cache.json`，GPU 结果可复用）

## 1. 头条结果：FNR 复现（test-clean 60 句 / 142 敏感词，检测收紧后）

评估协议：wav2vec2 CTC 词时间戳（代 WavLM+MFA）→ 自动检测（GDPR 词表+正则模式+BERT NER，
**虚词过滤 + date 正则收窄**，见 §2.5）→ 帧级 AES-CTR 加密 → Conformer ASR（beam=4，
与 M3 WER 同配置）重新转写。FNR 口径同论文 Table 7：**加密后 ASR 认不出的敏感词比例**
（越高越私密）。

| 状态 | 我们 | 论文 | 判定 |
|---|---|---|---|
| Ori（干净音频基线 FNR） | 8.57% | ≈0.6% | 口径差异（见 §3） |
| Enc margin=4 | 96.43% | — | 余量不足端 |
| **Enc margin=8 帧（默认）** | **98.57%** | **95.88%** | ✓ 达标且超过（2/140 幸存） |
| Enc margin=12 | 97.86% | — | 无收益，整句 WER 继续升 |
| Enc margin=16 | 98.57% | — | 同上 |
| Dec（解密还原） | = Ori | ≈0.6% | ✓ 逐字节还原，每句 assert 通过 |
| 加密帧覆盖（margin=8） | 33.5% | 未公布 | 报告披露 |

### 余量扫描结论
- **8 帧（288ms）定为默认**（annotation.MARGIN_FRAMES=8）：FNR=98.57%，且再加大余量
  无收益、只伤害整句 WER（36.4%→44.8%）；
- 收紧检测后 margin=4 也能到 96.4%，说明旧版 FNR 损失主因是**误命中虚词从上下文
  被脑补**，而非加密覆盖不足。

## 2. "FNR 75%" 问题的根因与解决（M4 最大教训）

| | 旧版（scripts/m4_e2e_test.py） | 新版（experiments/eval_fnr.py） |
|---|---|---|
| 词时间戳 | 参考文本 ↔ CTC 词流 **difflib 对齐+插值** | **直接对 CTC 词流检测**，时间戳与词同源 |
| 检测面 | 仅 NER 实体 | 词表+模式+NER（论文 §4.3.1 全三类） |
| 评估 ASR | wav2vec2（兼转写与检测） | **Conformer**（论文评估协议指定，与检测解耦） |
| FNR | 75%（6 句冒烟） | v1 94.38% → v2 **98.57%**（60 句 142 词） |

根因：旧版把敏感词定位寄托在"参考文本插值时间戳"上，插值误差使部分敏感词的加密区间没盖住
真实发音段；且用同一个 wav2vec2 既加密又检验，指标口径偏离论文。改为"检测-时间戳同源 + 
Conformer 独立评估"后命中论文数字。

**幸存词诊断（检测收紧后，margin=8 仅 2/140）**：剩余幸存者全部是 NER 对普通名词的
误标（`WIZARD`、`LAMP` 被标 ORG）——本身不是敏感内容，ASR 从上下文认出属正常；
真正的敏感实体（人名、电话、日期、机构专名）无一幸存。

### 检测收紧（v2，两刀）
1. **虚词过滤**（detect.py `FUNCTION_WORDS`）：模式/NER 命中落在英语封闭类虚词
   （THE/AND/MAY/US…）时只保留词表命中——LibriSpeech 语料里 NER 把 "TELLS US"
   标 LOC、date 正则尾随吞词都属此类误命中；国家名等多词实体不受影响；
2. **date 正则收窄**：删除 `(?:\s+\w+){0,3}` 尾随贪婪捕获，月名锚定单词即算命中。

v1（未收紧）：160 词、margin=8 FNR=94.38%，12 幸存全是虚词误命中——归档于
`experiments/results/fnr_eval_v1_baseline.json`；v2：142 词、FNR=98.57%。
转写缓存 `fnr_hyp_cache_v2.json`（v2 密文与 v1 不可混用）。

## 3. 与论文剩余偏差
- Ori FNR 6.87%（论文 0.6%）：论文 Ori 是"其检测系统标注的真敏感词在干净音频上的漏识别率"，
  我们的分母含 11 个 CTC 转写本身就不准/Conformer 不认的词（两套 ASR 的差异），
  属**口径差异**而非保护失效；Enc 94.38% 与论文 95.88% 差 1.5pp 同源于此。
- 加密帧覆盖 25–43%（margin 依赖）：论文未公布其覆盖比例；LibriSpeech 语料含
  大量 PER/LOC/ORG 日期类，检测面比论文 GDPR 场景宽。

## 3.5 MP3 发布层 FNR（P0-5 补测，`experiments/eval_fnr_mp3.py`）

单变量隔离：**WAV 层不加密敏感词**，只测公式(5) 的 MP3 帧体加密/静音替换。
25 句 / 62 敏感词（与 eval_fnr 同采样同打分口径），publish 真实链路（128kbps
44.1k MP3 → 帧体操作 → ffmpeg 容忍坏帧解码 → Conformer beam=4）：

| 态 | FNR（认不出） | 说明 |
|---|---|---|
| clean MP3（只编解码，未加密） | 9.68% | 有损格式本身 + 双模型口径的基线（≈WAV 层 Ori 8.57%） |
| **enc MP3（公式(5) 帧体加密）** | **100.00%** | 62 词无一复原；第三方解码器对密文帧整帧丢弃（连时间前移一起消灭内容） |
| quiet MP3（静音分发版） | 98.39% | 1 个幸存者（上下文脑补） |

论文 Table 7 Enc=95.88%——MP3 层两种形态均达标。结果并入
`main_table.json.table7_style_fnr.ours_mp3_layer` 与 `main_table.md`。
解码细节：ffmpeg 对含密文帧的 MP3 返回非零码但输出窗外解码样本——评估取
可解码部分（这正是 M4 §4 的第三方播放器行为，论文简易 decoder 的动机）。

## 4. MP3 适配（任务6）实测行为- `src/content/mp3_crypto.py`：顺序解析帧（跳过 ID3v2/v1 与帧头/CRC），公式(5) 秒→帧号，
  **只加密帧体**（跨帧区间按帧体拼接做 CTR，帧头穿插字节不动）。
- **quiet 无权限分发版**：加密段帧体清零 → ffmpeg 解码时长保持、窗外逐样本一致、窗内纯静音
  （test 固化）。这是给第三方播放器的正确分发形态。
- **enc 版在第三方播放器的行为**（有诊断价值，M7 加分项素材）：密文大概率不满足 Huffman
  约束，解码器整帧丢弃 → 加密段之后的音频前移（段前逐样本无损）。论文自带简易 decoder
  做静音替换正是此因。`assets/mp3_demo_{original,encrypted,quiet}.mp3` 三个样例。

## 5. 交付物清单（DoD 对照）
- `src/content/`：transcribe / align / ner / sensitive_lists / detect / encrypt / mp3_crypto / annotation ✅
- 手动标注 JSON（annotation.py，增删软删除/改时间/帧区间合并），M6 前端直接消费 ✅
- `tests/test_annotation.py` + `tests/test_mp3_crypto.py` + `tests/test_app_handlers.py`
  + MP3 层 FNR（`eval_fnr_mp3.py`），全库 **60/60 通过** ✅（P0 修复后计数）
- MP3 样例三态（assets/）+ FNR 数字（本文档） ✅
- DoD 逐条：加密后 ASR 转不出该词 ✓（94.4%）；未加密部分播放无损 ✓（quiet 分发版逐样本一致）；
  凭密钥解回 ✓（逐字节相等）
- 自录集 7 人已转码入库 + 转写辅助（`data/self_recorded/inventory.md`）：**发音不标准、
  内容未审，不进主评估**，人工审听后再定用途。
