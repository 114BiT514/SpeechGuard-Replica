# 华为云剩余工作清单（M7 收尾 + 验收准备）

> 状态日期：2026-11-02（M7 部署链路跑通当日）｜ 依据：`docs/M7_华为云部署验收报告.md` §8 与代码审计结果
> 实例：ModelArts Notebook `ascend-snt9b1`（910B4 32GB），区域 `cn-southwest-2`，按需 **¥18.689/小时**
> EVS 保留物：`/home/ma-user/work/SpeechGuard-Replica`（代码）、`data/hf-cache`（wav2vec2+BERT 3.7GB）、
> `~/start_app.sh`、`~/start_tunnel.sh`、`~/connect_proxy.py`
> 原则：**先做零成本的文档/代码项（§1、§4-§6），算力项集中在验收前一周做（§2、§3），做完即停实例。**

## 优先级总览

| # | 事项 | 优先级 | 需要开机 | 预估 |
|---|---|---|---|---|
| 1 | 重跑云端 demo，修正 `archive_segments` 摘录（审计 P1） | 🔴 必须 | ✅ | 开机 ~30 分钟 |
| 2 | 长期公网端点（或确认临时隧道口径） | 🔴 必须 | 验收周 | 方案见 §2 |
| 3 | 四模型 NPU 状态表（ECAPA/Conformer 补验） | 🟡 应做 | ✅ | 开机 ~1 小时 |
| 4 | OBS 归档 或 口径措辞修正 | 🟡 应做 | 仅方案 B | 措辞 10 分钟 / OBS 半天 |
| 5 | `deploy/requirements-cloud.txt` 版本固化 | 🟢 顺手 | ❌ | 10 分钟 |
| 6 | 文档卫生（SSH 端口易变注记等 3 处） | 🟢 顺手 | ❌ | 15 分钟 |
| 7 | 云端重跑 Table 5/6/7（全集） | ⚪ 默认不做 | ✅ | 13GB 上传 + 数小时 |
| 8 | 验收日演示 runbook | 🔴 必须 | 验收日 | 彩排 1 次 |

---

## 0. 通用前置：开机、SSH、环境恢复

每次需要云端算力时按此流程（§1、§3、§7 共用）：

1. **控制台开机**：ModelArts →  Notebook → 该实例 →「启动」。启动后到概览页
   重新复制 SSH 命令——**端口每次重启可能变**（上次是 30905），文档里的旧端口不要盲用。
2. **验证环境**（约 1 分钟）：
   ```bash
   cd /home/ma-user/work/SpeechGuard-Replica
   python -c "from src.device import device_report; print(device_report())"
   # 期望 {'torch': '2.1.0', 'npu_available': True, 'npu_count': 1, 'picked': 'npu:0'}
   ```
3. **确认代码是仓库最新版**（本地打包上传，方法同 `deploy/README.md` §2.1；
   沙箱下 scp 不可见时用 `cat file | ssh ... 'cat > file'` 管道法）。
4. **干完活立刻停机**（控制台「停止」）：计算停止计费；EVS 保留费约 ¥25/月量级（以账单为准）。

---

## 1. 🔴 重跑云端 demo，修正 archive_segments 摘录

**问题**：M7 三份文档摘录的 `"archive_segments": 89` 与当前代码矛盾——`warping.protect`
默认 single 模式，整句恒为 **1 段**；3.1s 音频总帧数才 86，任何模式都产不出 89。
助教按附录 A 复验会得到 1，与"实测摘要"对不上，属于必须消除的实证瑕疵。

**步骤**（开机后执行）：
```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache
python -m src.pipeline demo assets/sample_utt.wav \
    --outdir data/probe/pipeline --device npu:0
cat data/probe/pipeline/summary.json
```

**交付**：
- 把 summary.json **原文**（不改字段）替换进三处：
  `docs/M7_华为云部署验收报告.md` §3.3、`deploy/README.md` §2.5、`docs/M7_华为云部署复现.md` §2.4；
- 加一行注记：`archive_segments=1 系 single 模式（整句单一映射，见 src/acoustic/warping.py protect 默认 mode）`；
- 若 `protect` 耗时与 35.2s 差异大，同步更新并在注记说明首次加载 vs 常驻差异；
- 顺手把 `data/probe/pipeline/summary.json` 拉回本地存档一份（scp 反向或 ssh cat）。

**耗时**：开机 + 执行 + 改文档 ≈ 30 分钟（文档部分可本地做）。

---

## 2. 🔴 长期公网端点（10 分部署分的核心要件）

现状只有：SSH 本地转发（仅本人）+ Pinggy 临时链接（60 分钟过期、URL 随机）。
"可公网访问"的达标路径三选一，**先做决策 D0，再投入**：

**D0（先问助教/老师，零成本）**：验收形式是"线上屏幕共享演示"——确认
**「验收通话期间现场拉起临时公网链接并演示」是否满足 10 分口径**。
若满足 → 只做方案 B（成本≈0）；若要求"提前发放固定 URL 随时可访问" → 上方案 A。

### 方案 A：ModelArts 在线服务（真·长期公网，工作量大）

目标形态：自定义镜像 → SWR 仓库 → 在线服务（昇腾规格）→ 公网服务地址。

步骤骨架（控制台细节以实际界面为准，预计 0.5~1 天含排障）：
1. **制作推理镜像**：以华为云官方昇腾推理基础镜像（ModelArts 支持列表里的
   `pytorch-2.1-ascend-*` 推理版，注意与 Notebook 训练镜像区分）为基座；
   镜像内装 `deploy/requirements-cloud.txt`（§5 产物）+ 代码；
   Gradio 不适合在线服务的 REST 网关（SSE/WebSocket 可能不通），
   **需在 `src/pipeline.py` 外套一层最小 FastAPI**（`POST /protect /publish /recover`
   三个端点 + `/health`），前端演示仍走 Notebook——在线服务只承担"公网可访问的系统接口"；
2. **模型权重上 OBS**（顺带完成 §4 方案 B 的一半）：镜像不打包权重，
   启动脚本用 `moxing`/obsutil 从 OBS 拉 `data/hf-cache` 到本地；
3. **推 SWR** → 控制台创建在线服务（选昇腾规格、绑定 OBS 路径）→ 测试公网 URL；
4. **成本**：在线服务按调用/包周期计费，演示窗口短的话用**按需**并演示后立即下线；
   预算上限先设 ¥100，超了退回方案 B。

### 方案 B：Notebook + Pinggy 自动续期（低成本兜底，验收日可用）

在现有 `~/start_tunnel.sh` 外面套循环，断线/过期自动重连并把最新 URL 落盘：
```bash
cat > ~/tunnel_loop.sh <<'SH'
#!/bin/bash
while true; do
  bash ~/start_tunnel.sh > ~/pinggy.log 2>&1
  sleep 30   # 免费隧道 60 分钟过期后 ssh 退出，30s 后重连拿新 URL
done
SH
chmod +x ~/tunnel_loop.sh
setsid nohup bash ~/tunnel_loop.sh >/dev/null 2>&1 </dev/null &
# 随时取当前可用 URL：
sed 's/\r/\n/g' ~/pinggy.log | grep -o 'https://[^ ]*' | tail -1
```
局限：URL 每小时变，**不能提前印在材料里**，只适合"验收通话现场报新链接"。
演示前用 `curl -s -o /dev/null -w '%{http_code}' <URL>` 自检 200。

### 方案 C：ModelArts 控制台 `/proxy/7860/`（备用）

需华为云登录，非匿名——作为"助教有华为云账号时"的备用入口，写进验收材料即可。

**决策建议**：D0 问清口径 → 默认按 B 准备 + 彩排；只有明确要求固定 URL 才做 A。

---

## 3. 🟡 四模型 NPU 状态表（补计划 M7 任务 2 的缺口）

**问题**：执行计划要求"四个深度模型逐个 `to('npu')` 全量迁移验证，记录不支持的算子及
CPU fallback"。M7 实测只覆盖 wav2vec2 + BERT（pipeline 链路）；ECAPA/Conformer 的
NPU 证据停留在 M1（旧实例），M7 文档也未引用。

**步骤**（开机后执行，权重若不在则先按 `docs/M1_华为云NPU验收报告.md` §4.4 的 curl 清单
下载 `pretrained_models/ecapa_manual` 与 `pretrained_models/conformer`）：

```bash
cd /home/ma-user/work/SpeechGuard-Replica
export HF_HOME=$PWD/data/hf-cache
# ECAPA（声纹嵌入）
python - <<'PY'
from src.device import register_npu; register_npu()
import torch
from speechbrain.inference.speaker import EncoderClassifier
clf = EncoderClassifier.from_hparams(source="pretrained_models/ecapa_manual",
                                     run_opts={"device": "npu"})
emb = clf.encode_batch(torch.randn(1, 16000 * 3))
print("ECAPA npu:", tuple(emb.shape), emb.device)
PY
# Conformer（ASR）
python - <<'PY'
from src.device import register_npu; register_npu()
import torch, torchaudio
from speechbrain.inference.ASR import EncoderDecoderASR
asr = EncoderDecoderASR.from_hparams(source="pretrained_models/conformer",
                                     run_opts={"device": "npu"})
wav, sr = torchaudio.load("assets/sample_utt.wav")
if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
print("Conformer npu:", asr.transcribe_batch(wav.mean(0, keepdim=True),
      torch.tensor([1.0]))[0])
PY
```

**交付**：在 `docs/M7_华为云部署验收报告.md` 加一节"四模型 NPU 状态表"：

| 模型 | 用途 | NPU 验证 | 证据 | 算子回退 |
|---|---|---|---|---|
| wav2vec2-large | 转写+词时间戳 | ✅ 本次 | pipeline demo | 无 |
| BERT-NER | 敏感实体 | ✅ 本次 | pipeline demo | 无 |
| ECAPA-TDNN | ASV 评估 | 本次重验 | 输出粘贴 | 记录或写"未观察到" |
| Conformer | ASR 评估 | 本次重验 | 输出粘贴 | 同上 |

**耗时**：约 1 小时（含下载权重的话 +1 小时）。可与 §1 同一次开机合并做。

---

## 4. 🟡 OBS 归档 或 口径修正

deploy/README 第 9 行口径声明含"OBS 存储"，实际权重在 EVS——二选一消除不一致：

- **方案 A（零成本，推荐）**：把口径行改为
  "OBS 存储为规划项；当前权重/环境保留于实例 EVS（`/home/ma-user/work`）"。
- **方案 B（做方案 A 公网服务时的前置）**：装 obsutil 上传：
  ```bash
  # 控制台创建 OBS 桶（同区域 cn-southwest-2）→ 获取 AK/SK（注意：只存云端环境变量，勿入库）
  curl -sL https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_aarch64.tar.gz | tar xz
  ./obsutil config -i=<AK> -k=<SK> -e=obs.cn-southwest-2.myhuaweicloud.com
  ./obsutil mkdir obs://<bucket>/speechguard
  ./obsutil cp -r data/hf-cache obs://<bucket>/speechguard/hf-cache/ -flat
  # 可选：LibriSpeech/VCTK 若做 §7 也走这里
  ```
  流量费低（同区域内网免），存储 50GB 约 ¥6/月。**AK/SK 严禁提交进 git**（写入
  `~/.obsutilconfig`，该文件不在仓库内）。

---

## 5. 🟢 requirements-cloud.txt 版本固化

**本地写文件**（不需要开机），内容 = M7 验收报告 §2.2 实测矩阵：
```text
# 华为云 pytorch_2_1_ascend 镜像专用；勿用根目录 environment.yml 覆盖
torchaudio==2.1.0          # --no-deps
speechbrain==1.0.3         # --no-deps
gradio==4.44.1
huggingface-hub==0.29.1
pydantic==2.9.2
urllib3==1.26.20
imageio-ffmpeg==0.8.0
hyperpyyaml joblib sentencepiece tqdm pydub jiwer pyeer soundfile librosa matplotlib
```
并在 deploy/README §2.2 把逐条 pip 命令改为引用该文件。同时把 `imageio-ffmpeg`
在根 `requirements.txt` 标注为"云端兜底可选"。

---

## 6. 🟢 文档卫生（三处顺手改）

1. `deploy/README.md` §2.1/§2.4/§2.7 的 `30905` 端口与域名旁加注：
   "实例每次重启端口可能变化，以控制台 SSH 命令为准"；
2. `A104_SpeechGuard_大作业执行计划.md` 进度勾选：`M7 ☐` → `M7 ◐（部署链路 ✅ / 公网端点、加分项 ☐）`；
3. README.md「当前进度」M7 行末尾补一句指向本文档：`剩余事项见 deploy/TODO_华为云剩余工作.md`。

---

## 7. ⚪ 云端重跑 Table 5/6/7（默认不做）

本地（CUDA）已有全集权威数字（`experiments/results/main_table.md`），云端重跑只为
"昇腾算力上的复现一致性"叙事。**默认不做**；若老师明确要求：
- 上传 LibriSpeech test-clean+dev-clean、VCTK（约 13GB，走 §4 方案 B 的 OBS 中转最快）；
- 跑 `experiments/exp3_main_table.py`（注意云端 speechbrain 1.0.3 API 差异，
  开工前先在 30 句子集上冒烟）；
- 预算：上传约 1~2 小时 + 评估数小时 × ¥18.689/h ≈ **¥150-300**，量力而行。

---

## 8. 验收日演示 runbook（提前彩排一次）

**T-7 天**：定 §2 方案；若走 A，镜像与服务就绪。
**T-1 天彩排**（全流程按下面走一遍并计时）：
1. 开机 → §0 环境验证（2 分钟）；
2. `bash ~/start_app.sh` 起前端（约 1 分钟，含模型加载）→ `curl 127.0.0.1:7860` 自检；
3. 拉起公网入口（§2 方案 B：`tunnel_loop.sh`，取 URL 并 curl 自检 200）；
4. 演示脚本（每人轮讲时顺手演示）：
   - 主人 Tab：载入示例 → ① 保护+检测（看频谱/时间轴/敏感项表）→ ② 增删一项 → ③ 发布（三态 MP3）；
   - 听众 Tab：管理员=完整恢复 / 合作者=仅声纹 / 公开=拒绝，各播 10 秒；
5. **兜底预案**（按优先级）：公网链接 → SSH 转发（本机 127.0.0.1:7860）→
   控制台 `/proxy/7860/` → **本地 Gradio 录屏**（T-1 天录好一份放 assets/，最后保险）。
6. 演示结束 → 停隧道 → **停实例**（务必，¥18.689/h）。

**已知风险**：Pinggy 断连（保活参数已配，重连 30s 内恢复）；模型冷启动慢（彩排时
确认"首次 protect 含加载约 35s"，演示前先空跑一次让模型常驻）；910B 规格售罄
（提前一天确认实例能启动，启动不了立即提工单，见 M1 报告 §5.8）。

---

## 9. 成本管理 SOP

| 动作 | 计费影响 |
|---|---|
| 控制台「停止」实例 | 计算停止计费；EVS 保留（约 ¥25/月量级） |
| 删除实例 | 归零；下次需重下 3.7GB 模型（约 1 小时） |
| 忘停实例一天 | ≈ ¥448 —— **每次开机第一件事：设手机倒计时（自动停止已配 1 小时，但手动确认）** |

预算建议：剩余工作（§1+§3+彩排+验收日）合计开机 6~8 小时 ≈ **¥110-150**；
若做 §2 方案 A 另加在线服务费用（上限自设 ¥100）。
