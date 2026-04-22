# 迁移计划：ReAcTable 从 OpenAI API 迁移到本地 Qwen2.5-3B-Instruct (via vLLM)

## Context

用户希望在本地复现 ReAcTable 在 WikiTableQuestions 上的实验结果，不使用 OpenAI API，使用本地 Qwen2.5-3B-Instruct 模型。

**约束条件**：
- 所有新代码放在 `local_inference/` 新文件夹下
- **不修改任何原有文件**
- 使用 conda 创建 Python 3.10 环境进行所有操作（vLLM 最新版要求 3.10+）
- vLLM OpenAI 兼容服务器模式，GPU >= 8GB VRAM，保留 5 个 few-shot 示例

---

## 核心策略

**不修改原文件**的前提下，通过以下方式实现迁移：

1. **环境变量方案**：OpenAI SDK v1.x 在 `api_key=None` 时会自动读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 环境变量。原代码中模块级 `client = OpenAI(api_key=openai.api_key)` 在 import 时 `openai.api_key` 为 None，因此 SDK 会回退到环境变量。
2. **Monkey-patching**：在 import 原模块后，替换 `GptCompletion` 函数和响应解析逻辑，增加鲁棒性处理。
3. **子类覆盖**：创建 `CodexAnswerCOTExecutor_HighTemperaturMajorityVote` 的子类，覆盖关键方法。

---

## 文件结构

```
local_inference/
├── PLAN.md                  # 本计划文件
├── setup_env.sh             # conda 环境创建 + vLLM 服务器启动脚本
├── config.py                # 配置：模型名、vLLM 地址等
├── patch.py                 # Monkey-patch：替换 GptCompletion，修复响应解析
├── run_wikitq.py            # 主运行脚本（从 notebook 转化）
└── README.md                # 使用说明
```

---

## 修改步骤

### 第 1 步：创建 conda 环境

```bash
conda create -n reactable-qwen python=3.10 -y
conda activate reactable-qwen
cd d:/Code/TableQA/ReAcTable
pip install -e .
pip install vllm
```

### 第 2 步：`local_inference/config.py` — 配置文件

集中管理所有配置参数：
- `VLLM_BASE_URL = "http://localhost:8000/v1"`
- `VLLM_API_KEY = "EMPTY"`
- `MODEL_NAME = "qwen2.5-3b"`（对应 vLLM 的 `--served-model-name`）
- `MAX_TOKENS = 512`（原代码硬编码 128，对 3B 模型太小）
- `REPEAT_TIMES = 5`
- `MAX_DEMO = 5`
- `N_THREADS = 1`（vLLM 内部已做连续批处理，多线程无意义）
- `LINE_LIMIT = 10`

### 第 3 步：`local_inference/patch.py` — 核心适配层

**3.1 环境变量注入**

在 import 原模块之前设置环境变量，使 OpenAI SDK 自动指向 vLLM：
```python
import os
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
```

**3.2 Monkey-patch `GptCompletion`**

替换 `tabqa.GptConnector.GptCompletion`，修复以下问题：
- 统一为 chat completions 路径（原代码有不工作的 non-chat 分支）
- 修复 `max_tokens=128` 硬编码（第 152 行）
- 提升 `max_tokens` 默认值到 512
- 增加实际重试逻辑（原代码 `max_retry=1` 无重试）
- 为 response choice 设置 `.text` 属性以兼容下游

关键：通过 `tabqa.GptConnector.GptCompletion = patched_gpt_completion` 替换原函数，这样所有通过 `GptCompletion` 调用的地方都会使用新版本。

**3.3 添加鲁棒响应解析函数 `parse_llm_response()`**

兼容 3B 模型的格式偏差：
- 标准化类型名（SQL/sql → SQL, python/Python → Python）
- 灵活提取反引号内代码
- Fallback：无反引号时取冒号后内容

### 第 4 步：`local_inference/run_wikitq.py` — 主运行脚本

从原 notebook 转化为 Python 脚本。关键改动：
1. **首先 import patch.py**（设置环境变量 + monkey-patch）
2. **然后 import 原有模块**（此时环境变量已生效，client 自动指向 vLLM）
3. 创建 `CodexAnswerCOTExecutor_HighTemperaturMajorityVote` 的**子类**，覆盖 `_get_gpt_prediction`：
   - 使用 `parse_llm_response()` 替代原始的 `.split(":")[0]` 解析
   - 统一使用 `original_output.choices[0].text` 访问响应
4. 覆盖 `_get_gpt_prediction_majority_vote`：
   - 增加单次投票的 try/except 异常处理
   - 过滤空预测后进行多数投票
5. 使用 `joblib.Parallel(n_jobs=1)` 单线程运行

脚本流程：
```
加载配置 → import patch → import 原模块 → 读取数据集 →
对每条数据：创建子类实例 → 设置参数 → 生成 prompt → 多数投票 → 记录结果
→ 保存 JSON → 可选评估
```

### 第 5 步：`local_inference/setup_env.sh` — 环境脚本

自动化脚本，包含：
1. conda 环境创建命令
2. 依赖安装命令
3. vLLM 服务器启动命令（后台运行）
4. 连通性测试

### 第 6 步：vLLM 服务器启动

```bash
conda activate reactable-qwen
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen2.5-3b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype auto
```

参数说明：
- `--served-model-name qwen2.5-3b`：在 API 中使用的模型名（简短易用）
- `--max-model-len 8192`：5 个 demo + 当前表 + 中间表可能很长，8192 平衡内存和需求
- `--gpu-memory-utilization 0.85`：8GB 显卡留出余量
- Qwen2.5-3B FP16 约 6GB，8GB 显卡无需量化

---

## 技术要点说明

### 为什么环境变量方案可行

原代码 `GptConnector.py` 第 2-4 行和 `GptPrompter.py` 第 4-7 行：
```python
from openai import OpenAI
import openai
client = OpenAI(api_key=openai.api_key)
```

当 `openai.api_key` 为 None 时，OpenAI SDK v1.x 行为：
1. `api_key=None` → 读取 `OPENAI_API_KEY` 环境变量
2. `base_url` 未传 → 读取 `OPENAI_BASE_URL` 环境变量

因此在 import 前设置这两个环境变量，所有 module 级 client 都会自动指向 vLLM。

### 为什么需要 monkey-patch GptCompletion

原 `GptCompletion` 函数存在以下问题（不修改原文件的情况下只能通过 monkey-patch 修复）：
1. 第 152 行 `max_tokens=128` 硬编码，不使用参数值
2. chat 分支无 return 语句（gpt 函数内），外层直接 `return output` 依赖闭包
3. `max_retry=1` 无实际重试
4. non-chat 分支调用了错误的 API（`client.chat.completions.create` 而非 `client.completions.create`）

### 为什么需要子类覆盖 _get_gpt_prediction

`GptCOTPrompter.py` 中的 `CodexAnswerCOTExecutor_template._get_gpt_prediction()` 存在响应访问不一致：
- 第 459 行：`original_output.choices[0].message.content`
- 第 476 行：`original_output['choices'][0]['text']`（dict 风格，会崩溃）
- 第 508/535 行：`original_output.choices[0].message.content`

同时原始解析 `original_result.split(":")[0]` 对 3B 模型格式偏差不够鲁棒。

---

## 验证计划

1. **环境验证**：启动 vLLM 后运行简单测试脚本确认 API 可用
2. **单样本测试**：`maxLimit = 1`，检查完整 ReAct 循环
3. **小批量测试**：`maxLimit = 10, repeat_times = 3`，验证多数投票
4. **全量运行**：`maxLimit = 全部, repeat_times = 5`

---

## 风险与应对

| 风险 | 应对策略 |
|------|----------|
| 3B 模型格式遵循率低 | `parse_llm_response()` 鲁棒解析 + 5 次多数投票平滑错误 |
| SQL/Python 生成质量低 | ReAct 循环自带错误恢复（历史 DataFrame 回退）；降低预期准确率 |
| 显存不足 (OOM) | 降低 `--max-model-len` 或使用 `--quantization awq` |
| 推理速度慢 | 3B 模型在 8GB GPU 上推理速度尚可；可适当降低 `repeat_times` |
| monkey-patch 失效 | 如果 import 顺序不对，环境变量未及时设置会导致 client 指向 OpenAI |


## 实现顺序

1. `config.py` → 2. `patch.py` → 3. `run_wikitq.py` → 4. `setup_env.sh` → 5. 验证
