# 扩展实验与报告更新交接清单

> 项目：基于预训练语言模型的简历岗位匹配度评估与面试辅助系统  
> 工作目录：`C:\Users\25138\project\RJMN\resume_jd_matcher`  
> 目标：补齐证据检索、生成结构有效性、Qwen/模板人工盲评和生成案例实验，并用真实结果更新 `docs/overleaf/main.tex` 第五章。

## 1. 给接手 AI 的总指令

请先阅读完整仓库，再执行本清单。你需要完成代码、数据、实验、图片、LaTeX 和验证，不要只给方案。

必须遵守以下原则：

- 不修改 `core/` 中的匹配与评分算法；若为记录实验阶段增加诊断字段，只能增加元数据，不得改变原有生成文本和评分结果。
- 不得虚构实验样本、人工评分、成功率或图表数值。
- 所有表格、正文数字和图片必须来自同一份结构化实验结果文件。
- 原始 Qwen 输出、修复输出、模板降级和最终规范化结果必须分开记录。
- TF-IDF 与 BGE 只能在相同候选证据、相同人工标注和相同指标下比较。
- 不得把最终字段完整率写成“Qwen 准确率”。
- 不得把六组构造案例的等级一致性写成真实招聘准确率或泛化能力。
- 保留工作区中已有的用户修改，不得重置或删除无关文件。
- 修改前先查看 `git status --short`；只处理本任务涉及的文件。

## 2. 开始前必须阅读的文件

- `README.md`
- `config.py`
- `data/eval_cases.json`
- `core/preprocess.py`
- `core/evidence_matcher.py`
- `core/embedding_model.py`
- `core/llm_generator.py`
- `core/matcher.py`
- `scripts/run_eval_cases.py`
- `scripts/check_full_pipeline.py`
- `scripts/visualize.py`
- `docs/overleaf/main.tex`

重点定位以下 LaTeX 标签，后续用于替换占位图：

- `fig:retrieval-metrics`
- `fig:generation-validity`
- `fig:generation-human-rating`
- `fig:generation-case`

## 3. 环境与模型预检

在 PowerShell 中执行：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
git status --short
.\myenv\Scripts\python.exe scripts\check_models.py
.\myenv\Scripts\python.exe scripts\run_eval_cases.py
.\myenv\Scripts\python.exe scripts\check_full_pipeline.py
```

若项目实际使用的不是 `myenv`，先执行：

```powershell
Get-Command python
python -c "import sys; print(sys.executable)"
```

预检通过标准：

- BGE 模式必须是 `pretrained_embedding`，不能是 `tfidf_fallback`。
- Qwen 生成模式必须确认模型真实加载；如果进入模板降级，要先记录原因，不能伪装成 Qwen 结果。
- 当前配置应指向：
  - `models/bge-small-zh-v1.5`
  - `models/Qwen2.5-1.5B-Instruct`
- 至少保存 Python、Torch、Transformers、Sentence-Transformers 版本和运行设备。

建议新增：

```text
output/experiments/environment.json
```

至少记录：时间、Git 提交、Python 版本、模型路径、模型名称、设备、阈值、Top-K 和随机种子。

## 4. 任务 A：扩充整体匹配评价集

现有 `data/eval_cases.json` 的 6 组案例保留为“探索性机制验证集”，不要覆盖。另建：

```text
data/eval_cases_extended.json
```

建议总量为 24 组，即原有 6 组加 18 组人工复核的新案例。最低不得少于 18 组。四个等级尽量平衡：

| 等级 | 建议数量 | 典型情况 |
| --- | ---: | --- |
| 高度匹配 | 6 | 技能、项目证据和岗位职责充分对应 |
| 较高匹配 | 6 | 核心技能覆盖，但存在少量缺口 |
| 一般匹配 | 6 | 相邻领域、项目证据不足或关键词堆叠 |
| 匹配度较低 | 6 | 明显跨领域或缺少核心技术证据 |

数据要求：

- 不得包含真实姓名、电话、邮箱、学校编号等个人信息。
- 每份 JD 至少包含 4 条独立要求。
- 每份简历至少包含技能、项目或实习等 6 条可检索证据。
- 必须覆盖同义表达、词面相同但证据不足、跨领域迁移和边界分数案例。
- AI 可以起草案例，但必须由三名组员独立检查等级；最终标签采用多数票。
- 预期等级必须在运行系统前写入，禁止看完系统输出后反向修改。

建议保存三名标注者的原始结果：

```text
data/annotations/overall_labels_rater1.csv
data/annotations/overall_labels_rater2.csv
data/annotations/overall_labels_rater3.csv
data/annotations/overall_labels_consensus.csv
```

需要计算并保存：

- 精确等级准确率；
- Macro-F1；
- 相邻等级准确率；
- 综合分与人工序数等级的 Spearman 相关系数；
- 4×4 混淆矩阵。

结果保存为：

```text
output/experiments/overall_match_results.json
output/experiments/overall_match_predictions.csv
```

如果来不及完成扩展集，必须保留正文“探索性机制验证”的限定，不得编写准确率和 Macro-F1。

## 5. 任务 B：建立同口径证据检索评价集

不能直接用当前六组全文分数计算 TF-IDF 与 BGE 的性能提升。当前案例中部分 JD 只会切出一条长要求，证据候选也很少，`Recall@3` 可能失去区分度。

请新建结构化文件：

```text
data/retrieval_eval_cases.json
```

推荐结构：

```json
[
  {
    "case_id": "R01",
    "requirement_id": "R01-Q01",
    "requirement": "能够使用 Python 完成文本数据清洗",
    "candidates": [
      {"evidence_id": "R01-E01", "section": "项目经历", "text": "..."},
      {"evidence_id": "R01-E02", "section": "专业技能", "text": "..."}
    ],
    "relevant_evidence_ids": ["R01-E01"]
  }
]
```

数据规模与质量要求：

- 至少 36 条独立岗位要求，推荐 48 条以上。
- 每条要求至少 6 条候选证据，必须包含干扰项。
- 允许一条要求对应多个相关证据。
- 三名组员独立标注相关证据，分歧通过讨论形成共识。
- 不能让 AI 的相似度排名直接充当人工真值。

请新增：

```text
scripts/evaluate_retrieval.py
```

该脚本必须：

1. 读取同一份 `retrieval_eval_cases.json`。
2. 对每条要求和候选证据计算字符 2--4 gram TF-IDF 余弦相似度。
3. 对完全相同的文本计算 BGE 余弦相似度。
4. 分别排序，并计算 `Recall@1`、`Recall@3`、MRR。
5. 确认 BGE 状态为 `pretrained_embedding`，否则实验失败。
6. 保存每条要求的完整排名，便于误差分析。

结果保存为：

```text
output/experiments/retrieval_metrics.json
output/experiments/retrieval_rankings.csv
```

`retrieval_metrics.json` 至少包含：

```json
{
  "num_requirements": 48,
  "tfidf": {"recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0},
  "bge": {"recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0},
  "embedding_mode": "pretrained_embedding"
}
```

数字必须由程序计算，示例中的 `0.0` 不是实验结果。

## 6. 任务 C：记录 Qwen 结构化生成阶段

当前 `generate_analysis()` 返回最终结果，但没有完整暴露“首次解析、修复、规范化、模板降级”阶段。为了可靠统计，可以在不改变生成内容的前提下增加诊断元数据。

允许在 `core/llm_generator.py` 返回对象中增加：

```text
output_source                 direct_qwen / repaired_qwen / template_fallback
direct_parse_success          true / false
repair_attempted              true / false
repair_success                true / false
raw_required_fields_present   0..5
normalization_used            true / false
```

注意：

- 不得修改 Prompt、评分逻辑、采样参数和模板内容来追求更漂亮的指标。
- 首次解析失败后修复成功，应标记为 `repaired_qwen`。
- 修复失败并使用模板，应标记为 `template_fallback`。
- 错误提示必须准确，不能把模板降级写成“结构化修复成功”。
- `generation_mode` 表示模型加载方式，`output_source` 表示最终内容来源，两者不要混用。

请新增：

```text
scripts/evaluate_generation.py
```

脚本对至少 6 组案例运行；推荐使用扩展集中的 12 组代表性案例。由于当前 `do_sample=False`，无需通过重复生成制造样本量。

必须保存：

```text
output/experiments/generation_outputs.json
output/experiments/generation_metrics.json
```

每条输出至少包含：案例 ID、综合分、匹配等级、生成模式、输出来源、原始文本、解析状态、修复状态、五个字段、模型名称、设备和错误信息。

生成指标至少包括：

- 原始 JSON 直接解析率；
- 原始五字段完整率；
- JSON 修复尝试次数和成功率；
- 最终结构化成功率；
- 模板降级比例；
- 各状态对应的案例数量。

原始字段完整率必须在 `_normalize_analysis_result()` 补全之前统计。

## 7. 任务 D：制作 Qwen 与模板的盲评材料

在相同结构化输入下，为每个案例分别准备：

- Qwen 最终输出；
- `template_fallback_analysis()` 输出。

随机打乱并生成两个文件：

```text
output/experiments/blind_rating_items.csv
output/experiments/blind_rating_key.csv
```

`blind_rating_items.csv` 不能包含模型来源；`blind_rating_key.csv` 由汇总人员保管，评阅结束前不能交给评分者。

每位评阅者填写独立文件：

```text
data/annotations/generation_rating_rater1.csv
data/annotations/generation_rating_rater2.csv
data/annotations/generation_rating_rater3.csv
```

评分维度均为 1--5 分：

- 证据一致性；
- 岗位相关性；
- 建议可执行性；
- 语言清晰度。

评分文件还应包含 `notes`，用于记录虚构经历、套话、重复建议、问题偏题等错误。

请新增汇总脚本：

```text
scripts/aggregate_human_ratings.py
```

输出：

```text
output/experiments/human_rating_summary.json
output/experiments/human_rating_details.csv
```

每个模型、每个维度必须报告均值、标准差和有效评分数。不得用 AI 自评代替三名人工评阅者。

## 8. 任务 E：生成四张正式实验图片

请新增统一绘图脚本：

```text
scripts/plot_extended_experiments.py
```

统一要求：

- 图片读取 `output/experiments/*.json` 或 CSV，不得在脚本中手写实验数值。
- 白色背景，300 DPI，无 3D、无渐变、无装饰性元素。
- 中文字体优先使用 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。
- 图中文字在 A4 页面宽度下可读，不要依赖放大查看。
- 同时保存 PNG；有条件可额外保存 PDF。
- 柱顶标注数值，坐标轴写清范围和单位。

### 图 1：证据检索性能

保存为：

```text
docs/overleaf/figures/figure-12.png
```

要求：

- 横轴：Recall@1、Recall@3、MRR。
- 两组柱：TF-IDF、BGE。
- 纵轴：0--1。
- 标题：`TF-IDF 与 BGE 证据检索性能对比`。

### 图 2：生成结构有效性

保存为：

```text
docs/overleaf/figures/figure-13.png
```

要求：

- 展示直接解析成功、修复后成功、规范化补全、模板降级的案例数。
- 图内注明总案例数、原始字段完整率和最终结构化成功率。
- 不得把最终成功率标成 Qwen 准确率。

### 图 3：人工盲评

保存为：

```text
docs/overleaf/figures/figure-14.png
```

要求：

- 横轴：四个人工评价维度。
- 两组柱：Qwen、规则模板。
- 纵轴：1--5。
- 误差线：三名评阅者评分的标准差。
- 标题：`Qwen 与规则模板人工盲评结果`。

### 图 4：真实生成案例

保存为：

```text
docs/overleaf/figures/figure-15.png
```

优先选择“意图识别项目简历—NLP 语义匹配岗位”，展示：

- 命中技能与缺失技能；
- 两条优势；
- 两条短板；
- 两条优化建议；
- 基础技术、项目深挖、岗位匹配三类问题及回答思路；
- 实际 `output_source` 和模型名称。

必须来自 `generation_outputs.json` 的真实输出。可以使用 Streamlit 页面截图，也可以将 JSON 制作为排版清晰的结果图，但不能人工改写后声称是模型原始输出。

## 9. 任务 F：替换 LaTeX 占位符并更新文字

文件：

```text
docs/overleaf/main.tex
```

按标签查找，不依赖容易变化的行号。

将 `fig:retrieval-metrics` 前的占位框替换为：

```latex
\includegraphics[width=0.88\textwidth]{figures/figure-12.png}
```

将 `fig:generation-validity` 前的占位框替换为：

```latex
\includegraphics[width=0.88\textwidth]{figures/figure-13.png}
```

将 `fig:generation-human-rating` 前的占位框替换为：

```latex
\includegraphics[width=0.88\textwidth]{figures/figure-14.png}
```

将 `fig:generation-case` 前的占位框替换为：

```latex
\includegraphics[width=0.92\textwidth]{figures/figure-15.png}
```

然后根据真实结果更新第五章对应段落，必须写出：

- 检索要求数，以及 TF-IDF/BGE 的 Recall@1、Recall@3、MRR；
- 哪类案例中 BGE 有优势，哪类案例仍然失败；
- Qwen 直接解析率、修复成功率、最终结构化成功率、模板降级比例；
- Qwen 与模板四项人工评分的均值和标准差；
- 至少一个生成错误或不足，不能只写优点；
- 六组构造案例或扩展数据集的适用边界。

禁止使用以下表述，除非有独立数据严格支持：

- “BGE 准确率提升了 XX\%”；
- “Qwen 准确率为 100\%”；
- “系统可以准确判断候选人能力”；
- “实验充分证明系统具有真实招聘价值”；
- “所有生成内容均无幻觉”。

如果扩展整体评价集完成，还要同步更新：

- 摘要中的实验描述；
- 第五章综合匹配结果；
- 第六章工作总结；
- 第六章现有不足。

摘要只保留最重要的 1--2 个结果，不要堆叠全部数字。

## 10. 验证命令

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
.\myenv\Scripts\python.exe -m py_compile scripts\evaluate_retrieval.py
.\myenv\Scripts\python.exe -m py_compile scripts\evaluate_generation.py
.\myenv\Scripts\python.exe -m py_compile scripts\aggregate_human_ratings.py
.\myenv\Scripts\python.exe -m py_compile scripts\plot_extended_experiments.py
.\myenv\Scripts\python.exe scripts\evaluate_retrieval.py
.\myenv\Scripts\python.exe scripts\evaluate_generation.py
.\myenv\Scripts\python.exe scripts\aggregate_human_ratings.py
.\myenv\Scripts\python.exe scripts\plot_extended_experiments.py
```

人工评分未完成前，`aggregate_human_ratings.py` 应明确提示缺少文件并退出，不能自动生成虚假评分。

LaTeX 检查：

- Overleaf 编译器选择 XeLaTeX。
- 确认四张图片都能显示且没有越界。
- 检查图号、正文引用和目录页码。
- 检查不存在“待填写”“待实测”“请生成图片”等占位文字。
- 检查表格和正文中的数值与 JSON 完全一致。
- 检查图中文字在 100\% 页面缩放下可读。

## 11. 最终交付物

完成后必须交付：

```text
data/eval_cases_extended.json                         # 若完成扩展集
data/retrieval_eval_cases.json
data/annotations/*.csv
scripts/evaluate_retrieval.py
scripts/evaluate_generation.py
scripts/aggregate_human_ratings.py
scripts/plot_extended_experiments.py
output/experiments/environment.json
output/experiments/overall_match_results.json         # 若完成扩展集
output/experiments/retrieval_metrics.json
output/experiments/retrieval_rankings.csv
output/experiments/generation_outputs.json
output/experiments/generation_metrics.json
output/experiments/human_rating_summary.json
docs/overleaf/figures/figure-12.png
docs/overleaf/figures/figure-13.png
docs/overleaf/figures/figure-14.png
docs/overleaf/figures/figure-15.png
docs/overleaf/main.tex
```

## 12. 完成判定

只有同时满足以下条件才算完成：

- BGE 实验确认使用预训练模型而非 TF-IDF 降级。
- TF-IDF 与 BGE 使用完全相同的检索任务和人工真值。
- 生成阶段能够区分直接解析、修复成功和模板降级。
- 人工评分确实来自三名人员，而不是 AI 自评。
- 四张图均由结构化结果文件自动生成。
- LaTeX 已替换全部实验占位符，并按真实数字重写分析。
- 报告同时陈述有效结果和失败案例。
- 未出现无法由原始实验文件复核的结论。

完成后，请向项目负责人提交：修改文件清单、运行命令、关键结果摘要、四张图片、人工评分原始文件，以及仍然存在的限制。
