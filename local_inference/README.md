# ReAcTable 本地推理使用说明书

## 目录

- [1. 概述](#1-概述)
- [2. 前置条件](#2-前置条件)
- [3. 架构说明](#3-架构说明)
- [4. 环境搭建](#4-环境搭建)
- [5. 启动 vLLM 服务器](#5-启动-vllm-服务器)
- [6. 运行实验](#6-运行实验)
- [7. 结果评估](#7-结果评估)
- [8. 配置参数参考](#8-配置参数参考)
- [9. 文件结构说明](#9-文件结构说明)
- [10. 工作原理详解](#10-工作原理详解)
- [11. 常见问题与排查](#11-常见问题与排查)
- [12. 预期结果与局限性](#12-预期结果与局限性)

---

## 1. 概述

### 1.1 项目背景

ReAcTable 是 VLDB 2024 发表的表格问答（Table Question Answering）框架，核心思想是让大语言模型通过 **ReAct（Reasoning + Acting）** 循环逐步生成 SQL/Python 代码来处理表格数据，最终回答自然语言问题。

原始实现基于 OpenAI GPT-4 / Codex API。本工具包将其适配为**本地运行**，使用 **Qwen2.5-3B-Instruct** 小模型替代云端 API，在 WikiTableQuestions（WikiTQ）数据集上进行推理和评估。

### 1.2 核心特点

- **不修改原始代码**：所有适配代码放在 `local_inference/` 文件夹中，原项目文件完全不变
- **vLLM 加速推理**：通过 vLLM 的 OpenAI 兼容 API 模拟远程调用，本地 GPU 加速
- **鲁棒性增强**：针对 3B 小模型的输出格式偏差，增加了响应解析的容错处理
- **一键脚本**：提供环境搭建、服务器启动、连通测试的自动化脚本

### 1.3 WikiTQ 数据集

WikiTableQuestions 是斯坦福发布的表格问答基准数据集，包含从 Wikipedia 表格上收集的自然语言问答对。本工具使用其 `pristine-unseen-tables` 分割（测试集），共 434 条测试样例。

每条样例包含：
| 字段 | 说明 | 示例 |
|------|------|------|
| `id` | 样例编号 | `nu-0` |
| `utterance` | 自然语言问题 | `which country had the most cyclists finish within the top 10?` |
| `context` | 表格 CSV 路径 | `csv/203-csv/733.csv` |
| `targetValue` | 正确答案 | `Italy` |

---

## 2. 前置条件

### 2.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU，显存 >= 8GB | NVIDIA RTX 3090 / 4070 及以上 |
| 内存 | >= 16GB | >= 32GB |
| 硬盘 | >= 20GB 可用空间 | SSD |

> Qwen2.5-3B-Instruct 在 FP16 精度下约需 6GB 显存，加上 vLLM 的 KV Cache 开销，8GB 显存是最低要求。

### 2.2 软件要求

| 软件 | 版本要求 |
|------|----------|
| 操作系统 | Linux / Windows (WSL2) / Windows Native |
| CUDA | >= 11.8 |
| Conda | Anaconda 或 Miniconda |
| Git | 用于克隆项目 |

### 2.3 显存不足的替代方案

如果显存不足 8GB，可以选择以下方案：

| 显存 | 方案 |
|------|------|
| 4-6GB | 使用 AWQ 量化版模型 `Qwen2.5-3B-Instruct-AWQ`，启动时添加 `--quantization awq` |
| < 4GB | 使用 llama.cpp + GGUF 格式模型（需修改推理框架，不在本说明范围内） |

---

## 3. 架构说明

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      run_wikitq.py                          │
│  (主脚本：加载数据 → 构建 prompt → 调用推理 → 多数投票)      │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │   config.py     │    │    patch.py                  │   │
│  │  (参数配置)      │    │  ① 设置环境变量              │   │
│  └─────────────────┘    │  ② Monkey-patch GptCompletion│   │
│                         │  ③ 提供鲁棒响应解析            │   │
│                         └──────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        RobustCOTExecutor (子类)                       │   │
│  │  继承自原项目的 CodexAnswerCOTExecutor_               │   │
│  │  HighTemperaturMajorityVote                           │   │
│  │                                                      │   │
│  │  覆盖:                                               │   │
│  │  · _get_gpt_prediction()      → 鲁棒响应解析         │   │
│  │  · _get_gpt_prediction_majority_vote() → 异常处理     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ OpenAI SDK (HTTP)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  vLLM 服务器 (localhost:8000)                 │
│                                                              │
│  加载 Qwen2.5-3B-Instruct 到 GPU                             │
│  提供 OpenAI 兼容 API: /v1/chat/completions                  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 为什么不需要修改原文件

关键机制有两层：

**第一层：环境变量注入**

原项目代码在模块加载时创建 OpenAI client：
```python
client = OpenAI(api_key=openai.api_key)
```
当 `openai.api_key` 为 `None`（即未设置 API key）时，OpenAI SDK v1.x 会自动回退读取环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。`patch.py` 在 import 原模块之前设置了这两个环境变量，使所有 client 自动指向本地 vLLM。

**第二层：Monkey-patch + 子类覆盖**

- `patch.py` 用 `patched_gpt_completion` 替换了原 `GptCompletion` 函数，修复了原代码中 `max_tokens` 硬编码、无重试等问题
- `run_wikitq.py` 中的 `RobustCOTExecutor` 子类覆盖了 ReAct 循环方法，增加了鲁棒的响应解析

---

## 4. 环境搭建

### 4.1 获取项目代码

如果还没有克隆项目：
```bash
git clone https://github.com/YOUR_REPO/ReAcTable.git
cd ReAcTable
```

### 4.2 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
# 注：原项目推荐 Python 3.9，但 vLLM 最新版要求 Python 3.10+。
# Python 版本不影响实验结果，详见 4.6 节说明。
conda create -n reactable-qwen python=3.10 -y

# 激活环境
conda activate reactable-qwen
```

### 4.3 安装依赖

```bash
# 进入项目根目录
cd /path/to/ReAcTable

# 安装原项目依赖（以可编辑模式安装 tabqa 包）
pip install -e . --no-build-isolation

# 安装 vLLM
pip install vllm
```

> **注意**：vLLM 需要 CUDA 支持。如果安装失败，请确认 CUDA 版本与 vLLM 兼容。参考 [vLLM 官方安装指南](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

### 4.4 一键搭建（可选）

也可以使用提供的脚本一键完成环境搭建：

```bash
bash local_inference/setup_env.sh
```

该脚本会自动执行：创建 conda 环境 → 安装项目依赖 → 安装 vLLM。

### 4.5 验证环境

```bash
conda activate reactable-qwen
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

预期输出（示例）：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
VRAM: 12.0 GB
```

### 4.6 关于 Python 版本的说明

原项目推荐 Python 3.9，但 vLLM 最新版要求 Python 3.10+。本工具包使用 Python 3.10 环境，**不影响实验结果**：

- Python 版本只影响运行时环境和依赖库的加载方式
- 决定实验结果的要素（模型权重、prompt 模板、few-shot 示例、采样参数）均与 Python 版本无关
- `tabqa` 包的所有核心逻辑（字符串处理、pandas 操作、SQL 执行）在 Python 3.9 和 3.10 下行为完全一致

如果需要严格保持与原论文一致的 Python 3.9 环境，可以为 vLLM 单独创建一个 Python 3.10 环境运行推理服务器，两者通过 HTTP 通信互不干扰。

---

## 5. 启动 vLLM 服务器

### 5.1 基本启动

vLLM 服务器需要在一个**独立的终端**中运行，保持运行状态直到实验结束。

```bash
conda activate reactable-qwen

python -m vllm.entrypoints.openai.api_server \
    --model ./Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen2.5-3b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --dtype auto
```

**参数说明：**

| 参数 | 值 | 说明 |
|------|------|------|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace 模型 ID，首次运行会自动下载（约 6GB） |
| `--served-model-name` | `qwen2.5-3b` | API 中使用的模型名称，需与 `config.py` 中 `MODEL_NAME` 一致 |
| `--host` | `0.0.0.0` | 监听所有网络接口 |
| `--port` | `8000` | 服务端口 |
| `--gpu-memory-utilization` | `0.85` | GPU 显存使用比例，留 15% 给系统 |
| `--max-model-len` | `8192` | 最大上下文长度，平衡显存与需求 |
| `--dtype` | `auto` | 自动选择数据类型（通常为 FP16） |

### 5.2 首次启动

首次运行时，vLLM 会从 HuggingFace 下载模型权重（约 6GB）。下载完成后，模型会被缓存，后续启动只需数秒。

启动成功后会看到类似输出：
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5.3 验证服务器

在**另一个终端**中测试连通性：

```bash
# 方法一：使用脚本测试
bash local_inference/setup_env.sh test

# 方法二：手动测试
curl http://localhost:8000/v1/models
```

预期输出包含模型信息：
```json
{
  "data": [
    {
      "id": "qwen2.5-3b",
      "object": "model",
      ...
    }
  ]
}
```

也可以用 Python 做更完整的测试：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Hello, respond with 'OK'."}],
    max_tokens=10,
)
print(response.choices[0].message.content)
# 预期输出: OK
```

### 5.4 使用脚本启动（可选）

```bash
bash local_inference/setup_env.sh server
```

### 5.5 显存不足时的替代启动方式

```bash
# 使用 AWQ 4-bit 量化（需要先下载量化模型）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct-AWQ \
    --served-model-name qwen2.5-3b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --quantization awq
```

---

## 6. 运行实验

确保 vLLM 服务器已在另一个终端中运行。

### 6.1 快速测试（推荐首次运行）

使用少量数据验证整个流程是否正常：

```bash
conda activate reactable-qwen
python local_inference/run_wikitq.py --limit 5
```

预期输出：
```
[patch] GptCompletion patched -> vLLM at http://localhost:8000/v1
[patch] parse_llm_response registered
Dataset: 434 questions, running 5
Model: qwen2.5-3b, demos: 5, votes: 5, threads: 1
  0%|          | 0/5 [00:00<?, ?it/s]
...
Results saved to: ../dataset/WikiTableQuestions/results/RobustCOTExecutor_....json
Completed: 5/5 (0 errors)
```

> 首次运行较慢（模型加载、warm-up），后续推理会加快。每条样例约需 30-60 秒（5 次投票）。

### 6.2 小规模实验

```bash
python local_inference/run_wikitq.py --limit 50 --repeat 3
```

### 6.3 完整实验

```bash
python local_inference/run_wikitq.py --limit 434
```

> 完整 434 条样例 × 5 次多数投票，预计耗时 **数小时**（取决于 GPU 性能）。

### 6.4 命令行参数

```
python local_inference/run_wikitq.py [选项]

选项:
  --limit INT      处理的最大样例数（默认: 5）
  --repeat INT     多数投票重复次数（默认: 5）
  --threads INT    并行线程数（默认: 1，不建议修改）
  --demo INT       Few-shot 示例数量（默认: 5）
```

**使用示例：**

```bash
# 运行全部数据，3次投票（更快）
python local_inference/run_wikitq.py --limit 434 --repeat 3

# 运行100条，减少为3个few-shot示例（降低prompt长度）
python local_inference/run_wikitq.py --limit 100 --demo 3

# 快速验证（1条数据，1次投票）
python local_inference/run_wikitq.py --limit 1 --repeat 1
```

### 6.5 结果文件

结果保存在 `dataset/WikiTableQuestions/results/` 目录下，文件名格式：

```
RobustCOTExecutor_original-sql-py-no-intermediate_sql-py_limit{N}_modelqwen2.5-3b_votes{V}_demo{D}.json
```

每条结果包含：
```json
{
    "id": "nu-0",
    "utterance": "which country had the most cyclists finish within the top 10?",
    "source_csv": "csv/203-csv/733.csv",
    "target_value": "Italy",
    "predicted_value": "Italy",
    "prompt": "...(完整的 ReAct 推理过程)...",
    "execution_match": null,
    "gpt_error": null,
    "execution_err": null,
    "predicted_sql": null,
    "df_reformat_sql": null,
    "gpt_original_output": ["SQL: ```SELECT ...```", "Answer: ```Italy```"],
    "all_predictions": ["Italy", "Italy", "Italy", "Spain", "Italy"],
    "training_demo_ids": null
}
```

---

## 7. 结果评估

### 7.1 使用 WikiTQ 官方评估器

WikiTQ 提供了 Python 2 的评估脚本。如果系统有 Python 2：

```bash
cd dataset/WikiTableQuestions/
python2 evaluator.py ./results/<你的结果文件名>.json
```

### 7.2 手动计算准确率

如果无法使用 Python 2，可以用 Python 3 手动计算：

```python
import json

result_file = "dataset/WikiTableQuestions/results/<你的结果文件名>.json"
results = json.load(open(result_file))

total = len(results)
correct = 0
errors = 0

for r in results:
    if "uncaught_err" in r:
        errors += 1
        continue
    pred = str(r.get("predicted_value", "")).strip().lower()
    target = str(r.get("target_value", "")).strip().lower()
    if pred == target:
        correct += 1

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Errors: {errors}")
print(f"Accuracy: {correct / (total - errors) * 100:.1f}%" if total > errors else "N/A")
```

### 7.3 分析错误样例

```python
import json

results = json.load(open("path/to/results.json"))

# 查看所有投票结果
for r in results[:10]:
    if "uncaught_err" in r:
        print(f"[ERROR] {r['id']}: {r['uncaught_err']}")
    else:
        print(f"{r['id']}: pred={r['predicted_value']}, target={r['target_value']}, "
              f"votes={r.get('all_predictions', [])}, "
              f"match={str(r['predicted_value']).strip().lower() == str(r['target_value']).strip().lower()}")
```

---

## 8. 配置参数参考

所有配置集中在 `local_inference/config.py` 中：

### 8.1 服务器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM 服务器地址 |
| `VLLM_API_KEY` | `EMPTY` | API Key（本地服务器不需要真实 key） |
| `MODEL_NAME` | `qwen2.5-3b` | 模型名称，需与 vLLM 的 `--served-model-name` 一致 |

### 8.2 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_TOKENS` | `512` | 每次生成的最大 token 数（原项目为 128，对 3B 模型偏小） |
| `TEMPERATURE` | `0.6` | 多数投票时的采样温度 |
| `MAX_RETRY` | `3` | API 调用失败时的重试次数 |

### 8.3 实验参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REPEAT_TIMES` | `5` | 多数投票重复次数（越多越稳定，但越慢） |
| `MAX_DEMO` | `5` | Few-shot 示例数量 |
| `N_THREADS` | `1` | 并行线程数 |
| `LINE_LIMIT` | `10` | 每个表格在 prompt 中的最大行数 |

### 8.4 路径配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BASE_PATH` | `../dataset/WikiTableQuestions/` | 数据集根目录 |
| `DATA_FILE` | `...pristine-unseen-tables.tsv` | 测试集数据文件 |
| `TEMPLATE` | `original-sql-py-no-intermediate` | Prompt 模板名称 |
| `PROGRAM` | `sql-py` | 代码生成模式（SQL + Python） |
| `RESULTS_DIR` | `...results/` | 结果输出目录 |

### 8.5 调参建议

| 场景 | 建议调整 |
|------|----------|
| 推理速度太慢 | `REPEAT_TIMES` 降至 3，`MAX_TOKENS` 降至 256 |
| 准确率太低 | `REPEAT_TIMES` 升至 7，`MAX_DEMO` 降至 3（缩短 prompt，提高 3B 模型关注度） |
| 显存不足 | vLLM 启动时降低 `--max-model-len` 至 4096 |
| 格式遵循率低 | `MAX_DEMO` 降至 2-3，减少干扰 |

---

## 9. 文件结构说明

```
ReAcTable/
├── local_inference/               ← 本工具包（所有新增代码）
│   ├── README.md                  ← 本使用说明书
│   ├── PLAN.md                    ← 迁移计划详细文档
│   ├── config.py                  ← 集中配置参数
│   ├── patch.py                   ← 核心适配层
│   ├── run_wikitq.py              ← 主运行脚本
│   └── setup_env.sh               ← 环境搭建脚本
│
├── tabqa/                         ← 原项目核心代码（未修改）
│   ├── GptConnector.py            ← API 调用封装
│   ├── GptPrompter.py             ← 基础类和表格格式化
│   ├── GptCOTPrompter.py          ← ReAct 循环核心逻辑
│   ├── GptCOTPrompter_BeamSeach.py ← 多数投票 / Beam Search 类
│   └── ...
│
├── dataset/
│   └── WikiTableQuestions/        ← 数据集
│       ├── data/
│       │   ├── pristine-unseen-tables.tsv  ← 测试集（434条）
│       │   └── csv/                        ← 表格 CSV 文件
│       ├── prompt_template/                 ← Prompt 模板 JSON
│       │   └── original-sql-py-no-intermediate.json
│       ├── few-shot-demo/                   ← Few-shot 示例 JSON
│       │   └── WikiTQ-sql-py.json           ← 17 个标注好的推理示例
│       ├── results/                         ← 结果输出目录（运行时创建）
│       └── evaluator.py                     ← 官方评估脚本（Python 2）
│
├── notebooks/                     ← 原项目 Notebook（未修改）
├── setup.py                       ← 原项目安装配置（未修改）
└── requirements.txt               ← 原项目依赖（未修改）
```

---

## 10. 工作原理详解

### 10.1 ReAct 循环

ReAcTable 的核心是 ReAct 推理循环。对于每个问题，系统会反复执行以下步骤：

```
┌─────────────────────────────────────────────────┐
│ Step 1: 构建 Prompt                              │
│   [5 个 Few-shot 示例] + [当前表格 + 问题]        │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Step 2: 调用 LLM 生成响应                        │
│   模型输出: "SQL: ```SELECT ... FROM DF...```"   │
│   或: "Python: ```DF = DF[...]```"              │
│   或: "Answer: ```Italy```"                      │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│ Step 3: 解析响应                                 │
│   · answer_type = SQL / Python / Answer         │
│   · 提取代码或答案                               │
└──────────────────┬──────────────────────────────┘
                   ▼
         ┌──── answer_type? ────┐
         │                      │
    SQL / Python              Answer
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Step 4: 执行代码 │    │ Step 5: 返回答案 │
│ pandasql / exec │    │ 输出最终答案      │
│ 得到中间表格     │    └─────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Step 6: 将中间表格附加到 Prompt，回到 Step 2      │
│   Prompt += LLM响应 + "Intermediate table:\n..." │
│   最多循环 5 轮                                   │
└─────────────────────────────────────────────────┘
```

### 10.2 多数投票机制

为提高准确率，系统对每个问题独立运行 N 次（默认 5 次）完整的 ReAct 循环，收集所有最终答案，取出现次数最多的答案（众数投票）：

```
运行 1 → "Italy"
运行 2 → "Italy"        ┐
运行 3 → "Spain"        │ 多数: "Italy" (3票)
运行 4 → "Italy"        ┘
运行 5 → "Italy"

最终答案: "Italy"
```

温度设为 0.6 以引入采样多样性，使不同运行可能产生不同推理路径。

### 10.3 Prompt 构造

每个 Prompt 由以下部分拼接而成：

```
┌──────────────────────────────────────────┐
│ Few-shot Demo 1                           │
│   [表格] + [问题]                          │
│   SQL: ```SELECT ...```.                  │
│   [中间表格]                               │
│   Answer: ```答案```.                      │
│                                           │
│ Few-shot Demo 2                           │
│   ...                                     │
│                                           │
│ Few-shot Demo 5                           │
│   ...                                     │
│                                           │
│ 当前问题                                   │
│   [当前表格] + [当前问题]                    │
└──────────────────────────────────────────┘
```

表格格式：
```
[HEAD]: col1|col2|col3
---
[ROW] 1: val1|val2|val3
[ROW] 2: val1|val2|val3
...
```

### 10.4 代码执行

生成的 SQL 和 Python 代码由本地执行器运行：

- **SQL**：通过 `pandasql` 在内存中执行（将 DataFrame 注册为 SQLite 表），或对复杂 SQL 创建临时 SQLite 数据库
- **Python**：通过 `exec()` 执行，DataFrame 以变量 `DF` 传入
- **错误恢复**：如果代码执行失败，会回退到之前的历史 DataFrame 重试；如果全部失败，强制模型直接给出答案

### 10.5 与原项目的差异

| 方面 | 原项目 (GPT-4) | 本工具 (Qwen2.5-3B) |
|------|-----------------|---------------------|
| 推理引擎 | OpenAI API | 本地 vLLM + Qwen2.5-3B |
| max_tokens | 128（硬编码） | 512（可配置） |
| 响应解析 | `.split(":")[0]` | `parse_llm_response()` 鲁棒解析 |
| 错误处理 | 无（单次失败即崩溃） | try/except 包裹每次投票 |
| 并行线程 | 3 | 1（vLLM 内部批处理） |
| 原文件改动 | — | 零改动 |

---

## 11. 常见问题与排查

### 11.1 连接相关

**问题：`ConnectionError: Connection refused`**

原因：vLLM 服务器未启动或未就绪。

排查：
```bash
# 检查服务器是否运行
curl http://localhost:8000/v1/models

# 如果无响应，重启服务器
conda activate reactable-qwen
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen2.5-3b \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
```

等待看到 `Uvicorn running on http://0.0.0.0:8000` 后再运行实验。

**问题：`vLLM connection failed after 3 retries`**

原因：vLLM 服务器繁忙或 OOM。

排查：
- 检查 vLLM 终端是否有 OOM 错误信息
- 降低 `--max-model-len`
- 降低 `--gpu-memory-utilization`

### 11.2 模型相关

**问题：模型下载失败或速度慢**

解决：手动下载模型到本地，然后使用本地路径启动：

```bash
# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# 使用本地路径启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Qwen2.5-3B-Instruct \
    --served-model-name qwen2.5-3b \
    ...
```

**问题：`CUDA out of memory`**

解决：
1. 降低 `--max-model-len`（如改为 4096）
2. 降低 `--gpu-memory-utilization`（如改为 0.70）
3. 使用量化模型（见 5.5 节）

### 11.3 运行相关

**问题：所有预测结果都为空字符串**

原因：模型未能遵循 `SQL: \`\`\`...\`\`\`` 的输出格式。

排查：
1. 检查 vLLM 服务器是否正确加载了 Instruct 版本的模型
2. 查看结果文件中 `gpt_original_output` 字段了解模型的实际输出
3. 尝试减少 few-shot 示例数量（`--demo 2`），降低 prompt 复杂度

**问题：大量 `uncaught_err` 错误**

排查：
```python
import json
results = json.load(open("path/to/results.json"))
errors = [r for r in results if "uncaught_err" in r]
for e in errors[:5]:
    print(f"{e['id']}: {e['uncaught_err'][:200]}")
```

常见错误类型：
- `context length` → prompt 太长，减少 `--demo` 或在 config.py 中降低 `LINE_LIMIT`
- `Cannot execute SQL` → 模型生成了无效 SQL（属于正常现象，多数投票会平滑此类错误）

**问题：运行速度极慢**

优化建议：
1. 降低 `--repeat` 至 3（减少多数投票次数）
2. 在 `config.py` 中降低 `MAX_TOKENS` 至 256
3. 如果 GPU 支持，使用 FP8 精度：添加 `--dtype float8` 启动参数

### 11.4 Windows 特有问题

**问题：`setup_env.sh` 在 Windows 上无法运行**

解决：Windows 上可以直接在 PowerShell / CMD 中执行对应命令：

```powershell
# 创建环境
conda create -n reactable-qwen python=3.10 -y
conda activate reactable-qwen

# 安装依赖
cd D:\Code\TableQA\ReAcTable
pip install -e .
pip install vllm

# 启动服务器
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --served-model-name qwen2.5-3b --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192 --dtype auto
```

**问题：vLLM 在 Windows 上安装失败**

vLLM 对 Windows 的支持有限。可选方案：
1. 使用 WSL2 运行
2. 使用 Docker 运行 vLLM 服务器
3. 使用 `llama-cpp-python` 替代 vLLM（需修改 `config.py` 中的连接方式）

---

## 12. 预期结果与局限性

### 12.1 预期性能

| 模型 | WikiTQ 准确率（参考） |
|------|----------------------|
| GPT-4（原始论文） | ~54% |
| Codex（原始论文） | ~52% |
| Qwen2.5-3B-Instruct（预估） | ~15-25% |

3B 模型与 GPT-4 存在显著能力差距，主要体现在：
- SQL 生成错误率更高
- 输出格式遵循率更低
- 多步推理能力有限

### 12.2 已知局限性

1. **格式遵循**：3B 模型可能不严格遵循 `SQL: \`\`\`...\`\`\`` 格式，已通过 `parse_llm_response()` 缓解但无法完全消除
2. **SQL 质量**：复杂查询（JOIN、子查询、聚合）的错误率显著高于 GPT-4
3. **长上下文衰减**：5 个 few-shot 示例 + 表格数据的 prompt 很长，小模型在长上下文下注意力衰减明显
4. **Python 代码**：3B 模型生成的 Python 代码质量通常不如 SQL，出错率更高

### 12.3 改进方向

如果希望提升准确率，可考虑：

| 方向 | 具体操作 |
|------|----------|
| 更换更大模型 | 使用 Qwen2.5-7B-Instruct 或 14B（需更多显存） |
| 减少示例 | `--demo 2` 缩短 prompt，让模型聚焦当前问题 |
| 调整模板 | 修改 prompt template 添加更强的格式指令 |
| 增加投票 | `--repeat 7` 或更高，用更多样本平滑错误 |
| 量化蒸馏 | 在 WikiTQ 训练集上微调模型 |
