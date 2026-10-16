"""Hand-written conclusions for the HTML report (kept next to the data they summarize)."""
import json
from pathlib import Path

VERDICT = """
<ol>
  <li><strong>方案 1（自研）和方案 3（LangChain + 自研）都可以选，chonkie 不推荐。</strong>1 和 3 的效果差距在统计噪声范围内，选哪个主要看团队情况：想少引入依赖（10 个对 34 个）、自己掌控升级，选 1；langchain-core 已在白名单、团队本来就用 LangChain，选 3。</li>
  <li><strong>默认用 layout_aware，chunk_size 建议 200–300，可以按文档类型调整。</strong>有标题的文档上它比 fixed_size 好，没有标题时两者持平。semantic 更慢，效果也没有更好，一般不需要。</li>
  <li><strong>context enrich 先做成可选项。</strong>本地小模型测下来最多提升 5 个点，是否默认开启，等用公司的 LLM 测过效果和成本再定。</li>
</ol>
"""

CTX_NOTE = """
<p class="note">自研的基线是递归切分的 fixed_size。chroma-lite 上三个方案都跑了，结果一致，变化都在 1 个点以内。Qasper 和 OHR 子集只跑了自研，因为 context enrich 在三个方案里是同一份代码。OHR 子集只有 186 个问题、Qasper 子集只有 44 个，单个数字的波动不小。抽查发现，本地 4B 模型生成的上下文约 7% 是直接照抄原文；换成更强的生产模型后，效果预计会更好。</p>
"""

EXTRA = """
<li>粒度对照里，chunk_size 200 / 300 对应的 overlap 是 25 / 37 token（chunk_size 的 1/8），400 时是 50。召回率是检索指标；chunk 太小虽然检索更准，但交给 LLM 回答时上下文可能不够，chunk_size 最终要结合端到端问答效果来定。</li>
<li>所有代码、数据下载方式和原始结果都在 <code>document_chunking/</code> 目录，<code>uv run pytest</code> 可以跑 27 个单元测试。早期两版自研 fixed_size 的结果归档在 <code>results/archive_v1v2/</code>。</li>
"""

out = Path(__file__).resolve().parent.parent / "results" / "report_fragments.json"
out.write_text(json.dumps({"VERDICT": VERDICT, "CTX_NOTE": CTX_NOTE, "EXTRA_NOTES": EXTRA}, ensure_ascii=False))
