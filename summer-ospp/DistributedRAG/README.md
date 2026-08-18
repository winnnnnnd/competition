# DistributedRAG - 分布式多源异构文档检索增强生成系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-green.svg)](https://www.mindspore.cn/)
[![Ray](https://img.shields.io/badge/Ray-2.9.3-orange.svg)](https://ray.io/)

## 📖 项目简介

DistributedRAG 是一个基于 Ray、MinIO、PostgreSQL、Milvus 和 MindSpore 构建的分布式 RAG 系统。系统将文件存储、文档解析、OCR/ASR、Chunk、Embedding、向量写入、检索重排和答案引用拆分为独立阶段，并通过确定性 ID、版本发布和任务状态实现重复提交保护与失败恢复。

系统提供两套兼容入口：

1. `main_app1`：MindNLP CPU 模型配置；
2. `main_app2`：MindSpore Qwen3-Embedding、Qwen3-Reranker 和 Qwen2.5 生成模型配置。

两套入口共享 `rag_core` 中的摄取、存储、检索、引用、容错和可观测实现。

## ✨ 核心能力

- 支持 PDF、Word、PowerPoint、Excel、CSV、Markdown、文本、HTML、图片和音频；
- PDF 按页保留页码和文本偏移，扫描页通过 OCR Actor 处理；
- Word 保留标题路径和段落编号；
- PowerPoint 保留 Slide 编号和文本框坐标；
- Excel 保留 Sheet 名称和单元格范围；
- OCR 保留页码和 bbox，ASR 保留开始与结束时间戳；
- 原始文件及 Elements、Chunks 等中间产物使用确定性路径写入 MinIO；
- Ray Remote Task 负责无状态解析和 Chunk，Actor Pool 复用 OCR、ASR、Embedding 和 Reranker 模型；
- 使用有界 ObjectRef、`ray.wait`、Token 微批和批量 upsert 控制内存与写入压力；
- PostgreSQL 持久化 Job、Stage 和文档版本，本地可使用 SQLite；
- checksum、`document_version`、`chunk_id` 和 Milvus 主键均为确定性生成；
- 文档版本全部写入成功后再发布，检索仅查询已发布版本；
- 原 Query 与 HyDE 双路召回，通过 RRF 融合，低置信度时才执行 Rewrite；
- 支持 IVF_FLAT、FLAT、HNSW，索引参数、召回数和阈值全部可配置；
- 生成前建立 `S1/S2/...` 到真实 Chunk 和原始位置的服务端映射；
- 结构化解析答案与引用，执行引用白名单、版本、事实支持和引用覆盖检查；
- 提供任务查询、取消、重试、健康检查和 Prometheus 指标接口。

## 🏗️ 系统架构

```text
Streamlit / REST API
        │
        ├── 文件上传 ──> MinIO 原始文件
        │                    │
        │                    v
        │              PostgreSQL Job/Stage
        │                    │
        │                    v
        │           Ray 分布式摄取流水线
        │         ┌──────────┼───────────┐
        │         v          v           v
        │    Parser Task  OCR/ASR Pool  Chunk Task
        │         └──────────┬───────────┘
        │                    v
        │             Embedding Actor Pool
        │                    │ 微批
        │                    v
        │            Milvus Writer Actor
        │                    │ 批量 upsert
        │                    v
        │               发布文档版本
        │
        └── 用户问题
              ├── 原 Query Dense Recall
              ├── HyDE Dense Recall
              └── 低置信度 Rewrite Recall
                         │
                    RRF + 去重
                         │
                  Reranker + MMR
                         │
                  S1/S2 引用白名单
                         │
                结构化答案与来源定位
```

摄取过程中 Ray 任务只接收 MinIO URI、checksum、文档 ID 和配置等小型数据。页级 Elements 和 Chunk 批次通过有界 ObjectRef 流转，完整大文件不会长期保存在 Ray Object Store 中。

## 📁 项目结构

```text
DistributedRAG/
├── rag_core/
│   ├── config.py             # 环境配置与资源参数
│   ├── models.py             # Document、Element、Chunk、Citation 协议
│   ├── ids.py                # checksum 和确定性 ID
│   ├── storage.py            # MinIO 对象存储适配器
│   ├── state.py              # PostgreSQL/SQLite Job 与版本状态
│   ├── parsers.py            # 多源解析与来源定位
│   ├── chunking.py           # 保留定位信息的 Token Chunk
│   ├── actors.py             # OCR/ASR/Embedding/Reranker/Writer Actor
│   ├── ray_tasks.py          # Parser 与 Chunk Remote Task
│   ├── pipeline.py           # 流式摄取、背压、取消和发布
│   ├── vector_store.py       # Milvus Schema、批量 upsert 和检索
│   ├── retrieval.py          # Query/HyDE/Rewrite、RRF、重排和 MMR
│   ├── citations.py          # 短 ID 引用、校验与 repair
│   ├── observability.py      # 结构化日志和 Prometheus 指标
│   ├── service.py            # 统一业务服务
│   ├── api.py                # REST API
│   └── ui.py                 # Streamlit UI
├── main_app1/                # CPU 入口
├── main_app2/                # MindSpore Qwen 入口和模型实现
├── tests/                    # 单元、集成和故障注入测试
├── evaluation/               # 固定数据集格式与离线评测脚本
├── Dockerfiles/              # 应用、Ray Worker 和系统依赖镜像
├── docker-compose1.yml       # CPU 编排
├── docker-compose2.yml       # 加速设备编排
├── requirements.txt          # 固定版本运行依赖
└── requirements-dev.txt      # 测试依赖
```

## 🚀 快速开始

### 1. 准备配置

```bash
cp .env.example .env
```

修改 `.env` 中的 PostgreSQL 和 MinIO 凭据，并按机器资源调整 Ray Actor 数量、CPU 数量、批次大小和检索参数。

### 2. 准备模型目录

```bash
mkdir -p rag_models_cache
```

Qwen 配置默认使用以下目录：

```text
rag_models_cache/Qwen3-Embedding
rag_models_cache/Qwen3-Reranker
rag_models_cache/Qwen2_5-1_5B-Instruct
```

也可以通过 `EMBEDDING_MODEL`、`RERANKER_MODEL` 和 `LLM_MODEL` 指定其他路径。

### 3. 启动 CPU 配置

```bash
docker compose --env-file .env -f docker-compose1.yml up -d --build
```

### 4. 启动 Qwen/MindSpore 配置

```bash
docker compose --env-file .env -f docker-compose2.yml up -d --build
```

### 5. 访问服务

- Streamlit：`http://localhost:7860`
- REST API：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- Ray Dashboard：`http://localhost:8265`
- MinIO Console：`http://localhost:9001`
- Prometheus Metrics：`http://localhost:8000/metrics`

## 🔧 关键配置

### 分布式资源

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RAY_HEAD_CPUS` | `2` | Ray Head CPU 数量 |
| `RAY_WORKER_CPUS` | `8` | 每个 Worker 的 CPU 数量 |
| `RAY_MODEL_GPUS` | `0.25` | 每个模型 Actor 的 GPU 调度份额 |
| `RAY_MODEL_NPUS` | `0` | 每个模型 Actor 的 NPU 调度份额 |
| `RAY_CUSTOM_RESOURCES` | 空 | Worker 自定义 Ray 资源 JSON |
| `RAY_MAX_IN_FLIGHT` | `8` | 最大在途任务数量 |
| `INGESTION_CONCURRENCY` | `2` | API 后台摄取协调线程数 |
| `TASK_TIMEOUT_SECONDS` | `600` | 单个 Ray 操作超时时间 |
| `OCR_ACTOR_COUNT` | `1` | OCR Actor 数量 |
| `ASR_ACTOR_COUNT` | `1` | ASR Actor 数量 |
| `EMBEDDING_ACTOR_COUNT` | `1` | Embedding Actor 数量 |
| `RERANKER_ACTOR_COUNT` | `1` | Reranker Actor 数量 |
| `OCR_FAILURE_MODE` | `fail` | OCR 失败时选择 `fail` 或 `skip` |
| `ASR_FAILURE_MODE` | `fail` | ASR 失败时选择 `fail` 或 `skip` |
| `ALLOW_CPU_MODEL_FALLBACK` | `true` | 加速模型初始化失败时允许切换 CPU |

NPU Worker 注册示例：

```bash
export RAG_PROFILE=npu
export RAY_MODEL_GPUS=0
export RAY_MODEL_NPUS=0.25
export RAY_CUSTOM_RESOURCES='{"NPU":1}'
export NPU_VISIBLE_DEVICES=0
```

直接加入远程 Ray Head：

```bash
RAY_HEAD_ADDRESS=10.0.0.10:6379 \
RAY_WORKER_CPUS=16 \
RAY_CUSTOM_RESOURCES='{"NPU":1}' \
NPU_VISIBLE_DEVICES=0 \
./Dockerfiles/start-ray-worker.sh
```

### Chunk 与批处理

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `ELEMENT_BATCH_SIZE` | `16` | Parser 每次产出的 Element 数量 |
| `CHUNK_SIZE` | `600` | Chunk Token 数 |
| `CHUNK_OVERLAP` | `120` | 相邻 Chunk 重叠 Token 数 |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding 最大文本条数 |
| `EMBEDDING_MAX_TOKENS` | `8192` | Embedding 微批最大 Token 数 |
| `MILVUS_WRITE_BATCH_SIZE` | `128` | Milvus 单次 upsert 数量 |

### 检索与索引

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MILVUS_INDEX_TYPE` | `IVF_FLAT` | 可选 `IVF_FLAT`、`FLAT`、`HNSW` |
| `MILVUS_NLIST` | `1024` | IVF 聚类中心数量 |
| `MILVUS_NPROBE` | `16` | IVF 查询探测数量 |
| `DENSE_TOP_N` | `30` | 每路 Dense Recall 数量 |
| `RERANKER_TOP_N` | `20` | 进入 Reranker 的候选数 |
| `FINAL_TOP_K` | `5` | 最终证据数量 |
| `RRF_K` | `60` | RRF 融合常数 |
| `MMR_LAMBDA` | `0.7` | 相关性与多样性权重 |
| `CONFIDENCE_TOPK_MEAN_MAX` | `0.8` | Top-K 平均距离阈值 |
| `CONFIDENCE_MARGIN_MIN` | `0.03` | 结果间隔阈值 |

使用 MRL 时设置 `MRL_DIMENSION`。Embedding 会先裁剪到目标维度，再进行 L2 归一化；Milvus Collection 的维度必须与该值一致。

## 🔌 REST API

### 上传并摄取文档

```bash
curl -X POST http://localhost:8000/v1/documents \
  -F 'file=@./example.pdf'
```

### 查询任务

```bash
curl http://localhost:8000/v1/jobs/job_xxx
```

### 取消或重试任务

```bash
curl -X POST http://localhost:8000/v1/jobs/job_xxx/cancel
curl -X POST http://localhost:8000/v1/jobs/job_xxx/retry
```

### 提问

```bash
curl -X POST http://localhost:8000/v1/query \
  -F 'question=文档的主要结论是什么？' \
  -F 'document_ids=doc_xxx' \
  -F 'use_hyde=true'
```

返回结果将答案和引用分离：

```json
{
  "answer": "……[S1]",
  "citations": [
    {
      "source_id": "S1",
      "chunk_id": "ch_xxx",
      "document_id": "doc_xxx",
      "document_version": "dv_xxx",
      "source_locator": {"page_number": 3},
      "claim": "该页支持的事实",
      "source_name": "example.pdf",
      "source_url": null
    }
  ],
  "trace_id": "trace_xxx",
  "evidence_sufficient": true
}
```

客户端不能提交真实 `chunk_id` 作为模型引用。`S1` 到 Chunk、文档版本和原始定位的对应关系仅由服务端为当前请求建立，并在返回前重新校验。

## ♻️ 幂等与恢复机制

1. 原始内容计算 SHA-256 checksum；
2. checksum 与 Parser、Chunk、Embedding 版本共同生成 `document_version`；
3. `document_version + source_locator + chunking_version + chunk_index + text` 生成 `chunk_id`；
4. MinIO 原文件和中间产物使用确定性对象路径；
5. PostgreSQL 对文档处理版本建立唯一约束；
6. Milvus 使用 `chunk_id` 作为非自动主键并执行批量 upsert；
7. Ray Worker 或 Actor 失败后可按相同 ID 重试；
8. 只有完成 Milvus flush 的版本才进入 PostgreSQL 已发布集合；
9. 查询时使用已发布版本列表过滤 Milvus，未完成版本不会参与检索；
10. 取消请求写入 Job 状态，流水线在批次边界取消仍在执行的 ObjectRef。

## 📊 可观测性

所有流水线日志均为 JSON，并使用 `trace_id`、`job_id` 和 `document_id` 串联。Prometheus 指标包括：

- 各 Stage 成功/失败耗时；
- 文档、Element 和 Chunk 数量；
- Embedding 与 Milvus 批次大小；
- MinIO 读写、Milvus 写入和检索耗时；
- 重试、超时与降级次数；
- Ray CPU/GPU/NPU 逻辑资源使用率；
- Ray Object Store 容量与使用量；
- GPU 设备利用率；
- 检索耗时和引用合法性。

## 🧪 测试与评测

安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

单元测试：

```bash
pytest -m 'not integration and not failure_injection'
```

集成测试：

```bash
RUN_RAG_INTEGRATION_TESTS=1 pytest -m integration
```

故障注入测试：

```bash
RUN_RAG_FAILURE_TESTS=1 pytest -m failure_injection
```

离线评测数据使用 JSONL，每条记录包含 Query、限定文档、相关 Chunk、答案关键词和期望定位字段。运行：

```bash
python evaluation/run_evaluation.py evaluation/dataset.jsonl \
  --profile accelerated \
  --output evaluation-results.json
```

摄取吞吐与延迟：

```bash
python evaluation/benchmark_ingestion.py ./evaluation/documents \
  --profile accelerated \
  --concurrency 2 \
  --output ingestion-benchmark.json
```

评测脚本输出 Recall@5、nDCG@5、MRR、答案关键词正确率、Faithfulness、引用合法率、引用正确率、引用覆盖率以及端到端 P50/P95 延迟。索引对照实验可分别设置：

```bash
MILVUS_INDEX_TYPE=FLAT
MILVUS_INDEX_TYPE=IVF_FLAT MILVUS_NLIST=1024 MILVUS_NPROBE=16
MILVUS_INDEX_TYPE=HNSW
```

## 🔍 代码能力对应关系

| 能力 | 代码位置 |
|---|---|
| 多源文档解析与原始定位 | `rag_core/parsers.py`、`rag_core/models.py` |
| MinIO 持久化与确定性路径 | `rag_core/storage.py`、`rag_core/ids.py` |
| Ray Task、Actor Pool 与背压 | `rag_core/ray_tasks.py`、`rag_core/runtime.py`、`rag_core/pipeline.py` |
| Job、Stage、版本发布与恢复 | `rag_core/state.py`、`rag_core/pipeline.py` |
| Embedding 归一化和批量 upsert | `rag_core/actors.py`、`rag_core/vector_store.py` |
| Query/HyDE/Rewrite 与 RRF | `rag_core/retrieval.py` |
| 短 ID 引用与服务端校验 | `rag_core/citations.py` |
| 健康检查、任务接口与指标 | `rag_core/api.py`、`rag_core/observability.py` |
| 单元、集成和故障注入用例 | `tests/` |
| 可重复离线评测 | `evaluation/run_evaluation.py` |

## 🙏 致谢

- [Ray](https://ray.io/)
- [MindSpore](https://www.mindspore.cn/)
- [Qwen](https://github.com/QwenLM/Qwen)
- [Milvus](https://milvus.io/)
- [MinIO](https://min.io/)
