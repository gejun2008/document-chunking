# 用 Copilot 整理旧版切分设计的 prompt

用途：在公司电脑上，让 Copilot 读取以前基于 LlamaIndex 的切分实现（父子块 + 父块摘要），按固定格式整理出设计说明，用来对齐本仓库第 4 种策略（context_enriched）的实现和数据结构。

用法：在 VS Code 中打开以前的项目，把下面整段发给 Copilot Chat。拿到结果后，先确认里面没有不能带出公司的内容（内部系统名、地址、客户信息等），再拿来分析。

```text
请阅读这个项目中与文档切分（chunking）相关的代码，整理一份设计说明。只描述设计和参数，不要输出 API key、内部地址、客户数据或大段源码；必要时可以引用不超过 10 行的关键代码片段。请严格按下面的结构输出 Markdown：

## 1. 入口与调用链
- 切分功能的入口函数或类（文件路径 + 函数名）
- 从"输入文档"到"输出 chunk"的完整调用顺序，每一步对应哪个函数

## 2. 使用的 LlamaIndex 组件
- 列出用到的所有 LlamaIndex 类，例如 TokenTextSplitter、SentenceSplitter、HierarchicalNodeParser、SummaryExtractor、DocumentContextExtractor、IngestionPipeline、AutoMergingRetriever 等
- 每个类的初始化参数（chunk_size、chunk_overlap、separator、tokenizer 等，写出实际数值）
- llama-index 相关依赖包的名称和版本（来自 requirements / pyproject / lock 文件）

## 3. 父块和子块
- 父块怎么切：用什么切分器、大小、overlap
- 子块怎么切：在父块内部切还是在整篇文档上切；大小和 overlap
- 是否有更多层级（例如三层）
- 表格、标题、页码、换行这些是否有特殊处理

## 4. 摘要生成
- 对谁生成摘要（父块、整篇文档还是每个子块）
- 调用的模型名称或配置项名称，以及 temperature、max_tokens 等参数
- 完整的摘要 prompt 原文
- 摘要是怎么加到子块上的：拼进正文、放在元数据里，还是两者都有；拼接格式是什么
- 有没有缓存、重试、并发、失败时的处理

## 5. 数据结构与 ID
- 每个 chunk 输出的字段：名称、类型、含义（例如 id、doc_id、parent_id、text、embed_text、start/end 位置、page、metadata 等）
- chunk ID 怎么生成（随机 UUID、哈希还是序号），重复处理同一份文档时 ID 是否保持不变
- 父子映射怎么存：节点之间的 relationships、单独的映射表，还是字段
- 子块原文是否保持不变（是否做过空白或格式的规整）；有没有记录在原文中的字符位置

## 6. 下游怎么使用
- 哪些内容会被 embedding，哪些内容存入向量库
- 服务层如何用父子映射"还原"（例如检索到子块后取父块原文交给 LLM 做字段抽取）
- 字段抽取环节拿到的输入是什么

## 7. 配置与测试
- 哪些参数可以配置，默认值是多少，在哪个配置文件里
- 有没有相关的单元测试或评估脚本，测的是什么

## 8. 已知问题
- 代码注释或 TODO 里提到的问题、限制
- 依赖中有哪些安全漏洞（CVE）相关的说明，如果能看到的话

如果某一项在代码里找不到，就写"未找到"，不要猜测。
```

拿到设计说明之后的计划：

1. 对照旧设计，确定第 4 种策略的父子层级、大小、摘要 prompt 和拼接方式。
2. 让输出的数据结构（ID、父子映射、原文和位置）尽量与旧版一致，服务层不用改。
3. 把旧设计实现出来，在同样三个测试集上与当前做法对比效果和 LLM 调用成本。
