# 文档切分评测：数据集与评估方法说明

本文说明我们为什么选这三个测试集、怎么评估、做了哪些质量控制，以及结论的适用边界。所有数字都可以用仓库里的脚本复现（见文末"复现"一节）。

## 1. 要回答的问题

1. 自研实现的切分效果，是否不低于成熟开源库（LangChain、chonkie）？
2. 四种策略（fixed_size、layout_aware、semantic、context_enriched）在不同类型的文档上各有多大差别？
3. 哪些参数（chunk 大小、overlap）对效果影响最大？

切分本身没有"标准答案"，所以我们评估的是它对下游检索的影响：**切完之后，问题的标准答案能不能被检索回来，以及顺带检索回来多少无关内容。**

## 2. 为什么选这三个数据集

我们的选择标准是：

- **公开、有论文、被同行使用**，别人可以独立复核；
- **标准答案精确到原文片段**，而不只是"哪篇文档相关"，这样才能衡量切分的好坏；
- **覆盖 Document Studio 的真实输入形态**：纯文本、带标题结构的 Markdown、OCR 输出（含表格、公式、中英文）。

| 数据集 | 出处 | 规模（原始） | 标准答案怎么来的 | 对应 Document Studio 的哪类输入 | 许可 |
|---|---|---|---|---|---|
| Chroma chunking evaluation | Smith & Troynikov, *Evaluating Chunking Strategies for Retrieval*, Chroma Technical Report, 2024 | 5 个语料，约 32.8 万 token，472 个问题 | LLM 生成问题和引用片段，只保留片段能在原文中**逐字匹配**的问题，再按 embedding 相似度去重、去弱相关 | 纯文本（演讲、维基、聊天记录、财报、PubMed） | 仓库 MIT（语料来自各公开来源） |
| Qasper | Dasigi et al., *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*, NAACL 2021 | 1,585 篇论文，5,049 个问题 | NLP 从业者只看标题和摘要写问题，**另一组人**读全文作答并标出证据段落 | 有章节层级的长文档（转成 `#/##/###` Markdown） | CC-BY-4.0 |
| OHR-Bench | Zhang et al., *OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation*, ICCV 2025 | 8,500+ 页 PDF，7 个领域，8,498 个问答 | 每页 PDF 配有**人工校验**的结构化文本；问答的证据来自正文、表格、公式等 | OCR 后的文档：法律、财报、手册、论文、教材等，含 LaTeX 表格，中英文混合 | CC-BY-4.0，但数据卡注明**仅限研究用途、不可商用** |

### 2.1 Chroma：业内专门为"切分"设计的评测

大多数检索基准（MTEB、BEIR）只判断"整篇文档是否相关"，用 nDCG 之类的排序指标，看不出切分的好坏。Chroma 的报告指出：RAG 场景里，相关内容可能分散在多个片段，chunk 里会夹带干扰内容，而 LLM 对相关内容在上下文中的位置不太敏感。所以他们改用**按 token 计算的召回率、精确率和 IoU**，直接衡量"该取回的内容取回了多少、多带了多少无关内容"。这正是我们关心的问题，所以我们沿用了他们的数据和指标定义。

他们的报告也提供了外部参照：在他们的实验里，LangChain 的 RecursiveCharacterTextSplitter 在 chunk 较小时表现稳定，而 Kamradt 式语义切分默认设置下召回率略低于平均。我们的结果与此一致（见第 5 节）。

### 2.2 Qasper：人工标注、有章节结构

Chroma 的问题是 LLM 生成的，语料也基本没有标题结构，测不出 layout_aware 的价值。Qasper 补上这两点：

- 问题和证据都由人标注，而且**提问的人看不到正文**，避免了"照着原文改写问题"导致检索过于容易；
- 论文有天然的章节层级，我们抽样的 80 篇论文平均每篇 16 个标题。

### 2.3 OHR-Bench：最接近我们的生产输入

Document Studio 的文档大多先经过 OCR。OHR-Bench 专门研究 OCR 质量对 RAG 的影响，提供人工校验的 OCR 结果（相当于"理想 OCR 输出"），覆盖 7 个领域，证据类型包括正文、表格和公式。这让我们能在接近真实输入的文本上比较切分方法，而且不受某个 OCR 引擎好坏的干扰。

## 3. 评估方法

### 3.1 指标

对每个问题，用同一个 embedding 模型把问题和所有 chunk 向量化，按余弦相似度排序，然后按字符区间计算：

- **召回率** = 检索结果覆盖的标准答案字符数 ÷ 标准答案总字符数
- **精确率** = 检索结果中属于标准答案的字符数 ÷ 检索结果总字符数
- **IoU** = 重叠部分 ÷（检索结果 ∪ 标准答案）

定义与 Chroma 报告一致。他们按 token 计，我们按字符计，因为不同实现用的分词器不同，字符区间对所有实现都是同一把尺子。

**主指标是"2,000 token 预算下的召回率"（R@2k tok）：** 按相似度从高到低取 chunk，直到累计 2,000 token。如果按固定 top-k 比，切得越大的实现，每次取回的内容越多，会占便宜；按 token 预算比，各实现给 LLM 的上下文一样多，比较才公平。报告里同时给出 R@5、P@5、IoU@5 和 1,000 token 预算的召回率作为参考。

### 3.2 控制变量

| 变量 | 做法 |
|---|---|
| embedding 模型 | 三个方案都用 BAAI/bge-m3（多语言，支持 100 多种语言，最长 8,192 token；Chen et al., 2024），保证中英文文档都能处理 |
| 生成上下文的 LLM | 三个方案都用同一个模型（本地 Qwen3-4B-Instruct），同一份 prompt（Anthropic, *Contextual Retrieval*, 2024） |
| chunk 大小 | 统一设为 400 token、overlap 50；另外在 200 和 300 两个大小上重跑，横轴改成实际平均 token 数，在同样的 chunk 粒度下比较 |
| 库的参数 | 除了 chunk 大小，都用库的默认值；只有 chonkie 的 semantic 阈值校准到与自研相近的平均 chunk 大小（`bench/calibrate_chonkie.py`），否则它切出的 chunk 只有约 70 token，没法比 |
| 检索范围 | Chroma 和 OHR-Bench 在整个数据集的所有 chunk 里检索。Qasper 的问题不写是哪篇论文（例如"这个数据集有多大？"），所以只在对应论文内检索，这也是 Qasper 本身的单文档问答设定 |

### 3.3 显著性检验

每个数据集上，我们对"自研 − LangChain"和"自研 − chonkie"做了**配对 bootstrap**（同一批问题，有放回抽样 5,000 次），给出 95% 置信区间（`bench/bootstrap.py`，结果在 `results/bootstrap.json`）。区间不跨 0 才算显著差异。

## 4. 质量控制：我们怎么确认测量是对的

| 检查 | 结果 |
|---|---|
| 每个 chunk 的位置信息都正确 | 评测脚本对每个 chunk 断言 `text == 原文[start:end]`，三个方案、所有数据集全部通过。LangChain 自带的 `start_index` 在有 overlap 时会错位，我们发现后改为自己重新定位 |
| Chroma 标准答案能对上原文 | 790 个引用片段中 790 个与原文位置逐字一致，472 个问题全部保留 |
| Qasper 标准答案能对上原文 | 抽样 80 篇论文、261 个问题，其中 226 个至少有一段证据能在正文里找到（其余是"无法回答"或证据在图表里），保留这 226 个 |
| OHR-Bench 标准答案能对上原文 | 每个领域抽 10 篇文档（共 70 篇），超长文档截到 5 万字符。这些文档里共 2,116 个问答：731 个保留；727 个的证据在截断之后；658 个的证据文本与 OCR 文本不逐字一致（例如表格格式不同），一律丢弃，不做模糊匹配，宁可少也不引入错误标注 |
| 检索范围设定合理 | 初版在 80 篇论文的混合池里检索 Qasper，召回只有 23%，主要是在猜问题属于哪篇论文。改为论文内检索后升到 68%，与 Qasper 的单文档设定一致 |
| 切分是否切断标准答案 | 统计"标准答案是否完整落在一个 chunk 里"（`bench/span_integrity.py`）。早期自研版本在 Qasper 上只有 85.6%，据此定位到"不看段落边界"的问题；现在的自研版本是 99.5%，与 LangChain 相同 |
| 同一算法结果一致 | LangChain 方案的 semantic 是自研算法套 LangChain 接口，两者结果逐位相同，说明评测流水线本身没有随机性 |

## 5. 主要结论及其证据强度

chunk_size=400 时的配对 bootstrap 结果（R@2k tok，单位为百分点，* 表示显著）：

| 数据集 | 策略 | 自研 − LangChain | 自研 − chonkie |
|---|---|---|---|
| Chroma | fixed_size | +0.6 [−0.7, +1.9] | **+2.2 [+0.1, +4.4] \*** |
| Chroma | layout_aware | +1.2 [−0.6, +3.0] | −0.7 [−2.0, +0.5] |
| Chroma | semantic | 0（同一算法） | −1.0 [−3.4, +1.4] |
| Qasper | fixed_size | 0（切出的 chunk 相同） | **+9.2 [+4.7, +13.9] \*** |
| Qasper | layout_aware | −1.4 [−4.6, +1.8] | +1.9 [−1.8, +5.4] |
| Qasper | semantic | 0（同一算法） | **+5.2 [+0.4, +10.2] \*** |
| OHR-Bench | fixed_size | 0.0 [−0.4, +0.4] | +2.0 [−0.6, +4.7] |
| OHR-Bench | layout_aware | +0.4 [−1.9, +2.8] | −0.3 [−1.2, +0.5] |
| OHR-Bench | semantic | 0（同一算法） | +0.5 [−2.2, +3.1] |

- **18 组比较里，自研没有一组显著差于两个库。** 与 LangChain 的差异都在噪声范围内；对 chonkie 有 3 组显著更好。
- **置信区间宽度也告诉我们什么叫"持平"：** Chroma 和 OHR-Bench 上约 ±2 个点，Qasper（问题较少）约 ±3–5 个点。报告里小于这个幅度的差异，都不应解读为谁更好。

## 6. 局限：哪些结论不能外推

1. **这是检索指标，不是端到端问答效果。** chunk 越小，检索越准，但交给 LLM 回答时上下文可能不够。chunk_size 最终要结合问答效果来定。
2. **模型是本地开源模型，不是公司的服务。** 换成公司的 embedding 和 LLM 后，绝对数值会变，方案之间的相对差距更有参考价值。context enrich 尤其依赖 LLM 质量：本地 4B 模型约 7% 的输出是照抄原文。
3. **中文覆盖有限。** OHR-Bench 抽样里中文为主的文档只有 4 篇，其余是英文。如果生产文档以中文为主，需要补充中文评测。
4. **Chroma 的问题是 LLM 生成的**，作者自己也提到数据集偏小、偏合成；Qasper 和 OHR-Bench 的人工标注弥补了这一点。
5. **context enrich 只在子集上测了**（每个 chunk 要调一次 LLM）：Qasper 子集 44 个问题，OHR 子集 186 个，单个数字波动较大。
6. **OHR-Bench 是抽样加截断的**，且约三分之一的问答因证据无法逐字定位被丢弃，表格类问题的占比因此偏低（保留的 731 个里正文 578、表格 130、公式 23）。

**建议的下一步：** 从公司真实文档里抽 50–100 篇，人工标 200 个左右的问题和证据片段，用同一套脚本跑一遍（数据加载器只需输出"文档 + 问题 + 证据字符区间"）。这是判断"在我们自己的文档上谁更好"最直接的证据。

## 7. 复现

三个数据集已经提交在仓库的 `data/` 目录（来源、版本和许可见 `data/README.md`）；`bench/download_data.py` 可以按固定版本重新下载，并校验 SHA-256。

```bash
uv sync --all-extras --all-groups
uv run pytest
uv run python bench/run_bench.py --strategies fixed_size layout_aware semantic
uv run python bench/run_bench.py --strategies context_enriched --out results/results_ctx.jsonl
bench/run_sweep.sh
uv run python bench/bootstrap.py
uv run python bench/make_report.py            # -> results/report.md
```

**换成公司的 embedding 和 LLM：** 设置 `EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_API_KEY` 和 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 后运行 `bench/run_company.sh`。它会把所有阶段的结果和报告写到 `results/company/`，再用 `bench/compare_runs.py results results/company` 与本地模型的结果逐项对比。在非 Apple Silicon 的机器上，本地 LLM 不可用，context enrich 只能用公司的接口。

## 参考文献

- Smith, B., & Troynikov, A. (2024). *Evaluating Chunking Strategies for Retrieval*. Chroma Technical Report. https://www.trychroma.com/research/evaluating-chunking
- Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021). *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*. NAACL 2021. https://arxiv.org/abs/2105.03011
- Zhang, J., Zhang, Q., Wang, B., et al. (2025). *OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation*. ICCV 2025. https://arxiv.org/abs/2412.02592
- Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. https://arxiv.org/abs/2402.03216
- Anthropic (2024). *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
