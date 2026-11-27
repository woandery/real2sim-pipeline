# 导师审阅入口

项目要解决的是：把已有视频场景重建实验整理成可安装、可复查的研究代码，并通过官方 OpenAI API 代替本地 Codex/Astra CLI。它当前重建场景与关键状态，不包含机器人策略训练或手部重建。

建议阅读顺序：

1. `README.md`：输入、输出、安装和主流程。
2. `docs/architecture.zh.md`：模型与程序各自负责的部分，审阅及证据约束。
3. `src/video_real2sim/models.py` 与 `config.example.toml`：可以生成什么、如何配置模型和运行预算。
4. `src/video_real2sim/pipeline.py`、`provider.py`：完整视频采样、多轮理解/修复、连续审阅和官方 API 调用。
5. `docs/operator-guide.zh.md`：每种物理探针能验证什么，以及如何处理失败。
6. `docs/validation.zh.md`、`docs/code-review.zh.md`：实际做过的检查、发现过的问题和未验证项。

## 最小 CPU 审阅

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
real2sim validate examples/box-proposal.json
pytest -q
```

无需 API 密钥或 GPU。`examples/historical/` 包含原有 10 个场景的几何契约样例；不包含原始数据集视频或历史批准。

## 实际使用前需填写的内容

模型名、可用的官方 API 凭据、Isaac Python 解释器路径，以及实际要处理的视频。密钥按项目所有者要求先留空；当前没有真实 API 生成质量或端到端费用/耗时数据。

## 评价边界

几何来自视觉理解和参数化建模，尺寸/质量/摩擦等仍可能依赖先验。多轮模型审阅可减少遗漏，但不等于标定或独立人工测量。物理诊断通过仅覆盖声明的条件；检查视频用关键状态摆姿展示，不能作为策略完成任务的证据。全自动路径也不能把缺失证据或未支持的物理要求转换成“成功”。
