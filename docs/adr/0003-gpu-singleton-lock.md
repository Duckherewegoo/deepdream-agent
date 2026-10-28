---
id: 0003-gpu-singleton-lock
aliases: []
tags: []
---

# ADR-0003: GPU 推理的全局锁与单例约束

- Status: Accepted
- Date: 2026-10-24（commit `02230b9` 重构落地）
- Related:
  - 代码：`tools/dream_gen.py`、`deepdream/generator.py`、`node/chat_service.py`
  - 痛点：多路径引发的显存抢占、重复加载与双写

## Context

DeepDream 算法极其吃资源：单张 800x800 图片单次迭代约 120 秒，模型（InceptionV3）体积约 100MB，显存占用 2-4GB。

由于 Agent 架构允许多条路径触发图片生成（意图直达、LLM 工具调用、历史复现），同时 Gradio 作为 Web 服务天然并发请求。在 2026-10-24 的重构前，系统出现了严重的资源冲突：
1. **双份模型**：多个模块私自实例化 `DreamGenerator`，导致内存中出现两份 InceptionV3，占满显存。
2. **显存抢占**：并发请求同时进入 GPU 推理，直接抛出 `CUDA out of memory`。
3. **状态双写**：因为多处设置 `output_dir`，生成的文件有时落在 `dream_outputs/`，有时落在 `deepdream_outputs/`。

## Decision

引入**跨模块单例（Singleton）**与**全局推理锁（Inference Lock）**：

1. 所有模型实例化必须通过 `get_dream_generator()` 获取单例。
2. 所有的 GPU 推理路径（无论从哪里发起），必须包裹在同一把 `threading.Lock` 中（`inference_lock`）。推理结束、显存清理（`torch.cuda.empty_cache()`）必须在锁内完成。
3. `set_output_dir` 统一在 Agent 初始化时调用一次，作为全局基准。

## Alternatives Considered

- **A. 不管并发，假设单用户使用**：不可取。Web UI 很容易被多次点击或并发访问，不加锁必定导致程序崩溃。
- **B. 用消息队列（Redis / Celery）处理生成任务**：**未采纳**。对于单机、单进程的个人项目，引入外部中间件维护成本过高。一把互斥锁就够了。
- **C. 依赖 Gradio 的全局队列（`queue()`）**：**未采纳**。Gradio 的队列只能控制进入 UI 回调的并发，但无法控制 Agent 内部图执行（LangGraph）在不同请求之间对全局模型的并发争抢。

## Consequences

**正面**

- **稳定性达标**：彻底解决了并发触发时的 `CUDA OOM`，以及模型重复加载的问题。
- **一致性**：所有生成文件统一落到一个目录，消除了双写。

**负面（接受的代价）**

- **并发性能归零**：由于使用全局互斥锁，同一时刻只能处理一个生成请求。后续用户必须等待（阻塞）。
- **锁粒度的权衡**：如果锁的范围放得太宽（把 LLM 调用也包进去），会导致整个应用卡死。目前的锁仅保护“模型推理+清显存”这一段。

## 备注
这是单体 Python 应用连接重型计算资源时的经典妥协：**在没有任务队列的前提下，单例 + 互斥锁是防止显存雪崩的最后防线。**
