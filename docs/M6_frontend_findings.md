# M6 结论：双角色前端 + 案例可视化，浏览器端到端验证通过

> 启动：`E:/Miniconda/envs/speechguard/python.exe -m app.main --port 7860`
> （默认 device=cuda:0，可用 SG_DEVICE/--device 切；云端部署见 M7）

## 1. 前端（app/main.py，Gradio 6.30）

**音频主人 Tab**：录音/上传（+ 内置示例语音按钮，免上传直接走流程）→
保护强度 dist 三档（标准=论文工作点 0.5–0.6）→ ① 保护+敏感项自动检测
（展示：保护后试听、频谱对比图、敏感词时间轴、CTC 转写、敏感项表格）→
② 人工增补（起始/结束词下拉）/删除敏感项（软删除，表格状态列实时更新）→
③ 发布（加密分发版 MP3 + 无权限静音分发版 MP3 + CP-ABE 权限保险库）。

**听众 Tab**：选择权限组（管理员/合作者/公开）→ 在线试听（自动切对应版本）→
按权限恢复并下载（管理员=内容解密+声纹逆扭曲；合作者=仅声纹逆扭曲、敏感词
保持加密噪声；公开=拒绝恢复，只拿到静音分发版）→ 发布态 vs 恢复态频谱对比。

## 2. 浏览器端到端验证记录（IAB 真实交互）

- 载入示例（LibriSpeech 3.1s 句）→ ① 保护：敏感项表格渲染 `POLITICS(vocab:politics)
  / RELIGION(vocab:religion)` 两条自动命中，频谱图+时间轴正常出图；
- ③ 发布：加密对象 2 组，加密分发版/静音分发版播放器出现；
- 听众 Tab：管理员恢复 →"内容解密 + 声纹逆扭曲"；合作者恢复 →"仅声纹逆扭曲，
  敏感词保持加密"；公开听众点恢复 → 正确拒绝（无资产权限）；
- 产物电平核验：original peak=0.52 无损入库；recovered_full peak=0.39 无削波
  （恢复有损为方法本性，论文 Table 5 亦承认，见 M3/M4 findings）；
- 过程中修复两个真 bug（浏览器验证的价值所在）：
  1. Gradio 音频回传 int16 域幅度 → `pipeline.protect` 入口做幅度归一兼容；
  2. warping OLA 输出超 [-1,1] → int16 量化饱和方波 → protect 端按峰值 0.98
     归一并记录 gain、recover 端还原（tests/test_pipeline.py 固化接线）。

### 2.1 P0-4 补充记录（2026-10-26 审查修复）：人工增删链路验证

真实鼠标指针事件在本机 IAB 会话无法送达（`playwright.click()` 与坐标级 `cua.click`
均超时/未达，属受控浏览器环境限制而非应用缺陷——面板可见性已开、元素命中测试
elementFromPoint 正常）。因此改用**与浏览器完全相同的 gradio 队列协议**
（gradio_client，同 session 保持 gr.State）走通全链路，并新增
`tests/test_app_handlers.py` 直达 handler 断言：

| 步骤 | 结果 |
|---|---|
| ① 保护（真实音频+模型，非替身） | 检出 POLITICS(vocab:politics)+RELIGION(vocab:religion)，1 个相邻合并项 |
| ② 增补 "3: DISPUTE" | 表格变 2 项，含 manual 行 ✓ |
| ② 删除 "w3-3/w1-1: DISPUTE" | 状态列="已删除"（软删除留痕），有效项回 1 ✓ |
| ③ 发布 | 敏感项/加密对象计数随删除联动 ✓ |
| 恢复 | 管理员=解密+逆扭曲；合作者=仅逆扭曲；公开=试听静音版 ✓ |
| 报错文案 | 空选/无 bundle 分支已区分（"请先在①执行保护+检测" vs "请先…选择起始词"），单测覆盖 ✓ |

同时修复了验证过程中暴露的真实 bug：gradio Dropdown 可选输入（结束词）未选时必须传
`null`，前端空字符串会被 gradio 校验层拒绝——UI 上无影响（前端自己发 null），但
说明"下拉值未提交"类问题此前确实在协议层存在，现在链路两端都验证过。

## 3. 可视化素材（报告直接可用）

| 文件 | 内容 |
|---|---|
| assets/fig_m6_spectrogram.png | 原始/保护/reducing攻击/恢复 四联频谱（Fig.6 风格） |
| assets/fig_m6_timeline.png | 敏感词时间轴 |
| assets/fig_m6_warp.png | 多参数扭曲曲线 + 逆映射曲线（Fig.5 风格） |
| assets/pipeline_demo/ | 端到端全部产物（三态 MP3/恢复 WAV/vault/sk/summary.json） |
| experiments/results/fig_exp1_fig7_single.png | Fig.7 选参曲线（M3） |
| experiments/results/fig_exp2_fig8.png | Fig.8 攻击 ECDF（M3） |
| experiments/results/fig_exp4_fig10.png | Fig.10 基线攻击成功率对比（M5） |
| assets/fig_m2_warping_spectrogram.png | 扭曲前后频谱（M2） |

## 4. DoD 对照

- [x] 浏览器走完"主人发布 → 三种听众分别听到不同版本 → 管理员完整恢复"
- [x] 每步中间产物可视化（频谱图、敏感词时间轴、表格）
- [x] 案例分析配图成稿（上表），文字结论已随 M3/M4/M5 findings 沉淀
- [ ] 案例分析文案 3–5 句/图 —— 留给 M8 报告阶段统一撰写（素材已齐）
