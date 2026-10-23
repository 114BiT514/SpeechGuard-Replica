# 华为云 ModelArts + 昇腾部署（M7 填写）

> M1 已完成云端环境验收（910B4 + CANN 8.0.0 + torch_npu 2.1.0），创建实例、依赖安装、权重下载与 NPU 推理的完整步骤见 [M1 华为云 NPU 验收报告](../docs/M1_华为云NPU验收报告.md)。

目标：整套系统（pipeline + Gradio 前端）部署到华为云 ModelArts 昇腾 NPU 实例，公网可访问。

口径：华为云 ModelArts + 昇腾 NPU（910B/310P）+ torch_npu（PyTorch 昇腾适配层）+ OBS 存储。
若课程硬性要求 MindSpore 框架：仅将 BERT NER 等单点换为 MindSpore Hub 等价实现，主链路不变。

## 部署步骤占位（M7 完成）

1. [ ] 权重与数据上传 OBS
2. [ ] `src/device.py` 确认 NPU 可用，四个模型（ECAPA / Conformer / WavLM / BERT）逐一 `to('npu')` 验证
3. [ ] 打包：ModelArts 在线服务 或 昇腾实例 + Docker Compose（Gradio）
4. [ ] 安全组放行端口，公网 URL 验收演示
5. [ ] 本 README 写清从零复现部署的完整步骤

## 环境

- ModelArts 预装镜像已含 CANN + torch_npu，**勿在云端执行根目录 environment.yml 覆盖环境**
- 设备选择：`src/device.py` 自动 `npu > cuda > cpu`
