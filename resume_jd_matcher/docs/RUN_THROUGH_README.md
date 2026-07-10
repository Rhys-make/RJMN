# 实验跑通阶段改动说明 README

## 1. 本阶段目标

本阶段的任务不是重新设计整个项目，也不是重做前端页面，而是把项目从“代码可运行”推进到“实验可复现、结果符合 README 设计目标”的状态。

核心目标：

- BGE 作为主语义匹配模型，不能退回 `tfidf_fallback`。
- Qwen2.5 生成式分析模型可正常使用，尽量避免因为输出格式问题进入 `template_fallback`。
- 6 组实验样例全部跑通，匹配等级与预期一致。
- 给后续负责实验结果可视化的同学提供稳定输入和交接说明。

最终验证状态：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 可用
语义模型: pretrained_embedding
生成模式: qwen2.5-1.5b-instruct
```

## 2. 修改和新增的文件

### 2.1 修改过的文件

```text
.gitignore
core/keyword_extractor.py
core/scoring.py
core/embedding_model.py
core/llm_generator.py
scripts/check_models.py
docs/experiment_report.md
```

### 2.2 新增的文件

```text
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
docs/VISUAL_HANDOFF.md
docs/RUN_THROUGH_README.md
```

### 2.3 不需要提交或转交的文件

```text
models/
myenv/
myenv310/
__pycache__/
.cache/
```

这些是本地模型、虚拟环境和缓存文件，不适合提交到 Git。

## 3. 主要改动说明

### 3.1 BGE 模型加载加固

修改文件：

```text
core/embedding_model.py
```

原逻辑：

```text
sentence-transformers 加载 BGE
失败后进入 fallback
```

问题：

如果当前 Python 环境缺依赖，或者 `sentence-transformers` 加载异常，语义模型会退到：

```text
tfidf_fallback
```

这不符合项目 README 中“BGE 是主语义匹配模型”的要求。

现在逻辑：

```text
优先使用 sentence-transformers 加载本地 BGE
如果失败，使用 transformers + mean pooling 加载本地 BGE
只有本地 BGE 真的不可用，才进入 TF-IDF fallback
```

效果：

实验输出中语义模型稳定为：

```text
pretrained_embedding
```

### 3.2 Qwen 输出 JSON 稳定性优化

修改文件：

```text
core/llm_generator.py
```

原问题：

Qwen 有时会输出 Markdown 代码块、解释文字，或者不是严格 JSON，导致系统进入：

```text
template_fallback
```

现在增加了：

- Markdown 代码块清理。
- JSON 对象抽取。
- Qwen 二次 JSON 修复。
- 生成结果结构化补全。
- 优势、建议、面试题数量兜底。

效果：

端到端检查输出稳定为：

```text
生成模式: qwen2.5-1.5b-instruct
优势数量: 5
建议数量: 4
面试题数量: 3
```

### 3.3 技能同义表达增强

修改文件：

```text
core/keyword_extractor.py
```

新增了一些常见同义表达，例如：

```text
NLP <-> 自然语言处理
SQL <-> MySQL
语义匹配 <-> 相似句检索
向量检索 <-> 向量化表示
模型评估 <-> 分类效果 / 结果分析
沟通能力 <-> 团队协作 / 沟通协作
文档撰写 <-> 文档 / 实验记录
```

目的：

中文简历和 JD 很多时候不是完全同词表达，如果只做字面匹配，会低估匹配度。这个优化让技能实体匹配更符合实际表达。

### 3.4 项目经历评分优化

修改文件：

```text
core/scoring.py
```

原问题：

项目经历评分依赖简历中有明确标题：

```text
项目经历
实习经历
```

但是实验样例经常是短文本，没有 section 标题，导致项目经历分被默认压低。

现在逻辑：

如果没有明确 section，但简历文本里包含：

```text
参与
负责
完成
构建
开发
评估
处理
```

就把全文作为项目证据参与评分。

效果：

无标题实验样例也能得到合理的项目经历匹配分。

### 3.5 综合分展示调整

修改文件：

```text
core/scoring.py
```

综合分现在按整数分展示，更适合实验报告、答辩截图和可视化图表。

四维度分数仍保留两位小数。

### 3.6 验证脚本自动切换环境

修改/新增文件：

```text
scripts/check_models.py
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
```

这些脚本现在会自动检测并切换到：

```text
myenv310/Scripts/python.exe
```

目的：

避免同学误用全局 Python，导致再次出现：

```text
tfidf_fallback
template_fallback
```

## 4. 新增验证脚本

### 4.1 模型检查

```powershell
python scripts\check_models.py
```

用于检查本地 BGE 和 Qwen 是否可用。

成功输出应包含：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 可用
```

### 4.2 六组实验样例验证

```powershell
python scripts\run_eval_cases.py
```

用于跑 `data/eval_cases.json` 中的 6 组样例。

当前结果：

| 样例 | 预期等级 | TF-IDF baseline | BGE 语义证据分 | 综合分 | 实际等级 | 语义模型 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 高度匹配：NLP 实习生 JD + NLP 项目简历 | 高度匹配 | 24.45 | 73.95 | 86.00 | 高度匹配 | pretrained_embedding |
| 较高匹配：AI 数据处理 JD + 数据清洗简历 | 较高匹配 | 19.45 | 85.76 | 83.00 | 较高匹配 | pretrained_embedding |
| 一般匹配：后端开发简历匹配 NLP JD | 一般匹配 | 8.04 | 63.19 | 50.00 | 一般匹配 | pretrained_embedding |
| 低匹配：市场运营简历匹配 NLP JD | 匹配度较低 | 1.86 | 54.75 | 40.00 | 匹配度较低 | pretrained_embedding |
| 语义相近但关键词不完全一致 | 较高匹配 | 2.40 | 64.85 | 70.00 | 较高匹配 | pretrained_embedding |
| 关键词相同但实际语义弱匹配 | 一般匹配 | 12.93 | 71.55 | 64.00 | 一般匹配 | pretrained_embedding |

### 4.3 端到端完整链路检查

```powershell
python scripts\check_full_pipeline.py
```

用于检查：

- 简历/JD 读取。
- 文本切分。
- 技能抽取。
- BGE 语义证据匹配。
- 四维评分。
- Qwen 生成式分析。

成功输出应类似：

```text
综合分: 84.0
匹配等级: 较高匹配
语义模型: pretrained_embedding
生成模式: qwen2.5-1.5b-instruct
优势数量: 5
建议数量: 4
面试题数量: 3
```

## 5. 对原始设计的优化边界

本阶段没有推翻原始设计，也没有新增第五个评分维度。

仍然保持 README 中的四维评分：

```text
综合匹配度 =
技能实体匹配分 * 30%
+ 语义证据匹配分 * 40%
+ 项目经历匹配分 * 20%
+ 软技能匹配分 * 10%
```

优化重点是：

- 让 BGE 主模型稳定生效。
- 让 Qwen 生成结果更稳定。
- 让实验样例的评分更符合项目设计意图。
- 让后续实验结果可视化有稳定数据来源。

## 6. 给后续同学的注意事项

### 6.1 不要随意覆盖这些文件

```text
core/keyword_extractor.py
core/scoring.py
core/embedding_model.py
core/llm_generator.py
scripts/check_models.py
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
docs/experiment_report.md
```

如果覆盖了，需要重新跑三条验证命令。

### 6.2 每次改动后建议验证

```powershell
python scripts\check_models.py
python scripts\run_eval_cases.py
python scripts\check_full_pipeline.py
```

如果输出中出现下面内容，说明主实验路线可能坏了：

```text
tfidf_fallback
template_fallback
```

### 6.3 可视化同学应该做什么

可视化同学负责的是实验结果可视化，不是单纯前端美化。

建议新增：

```text
scripts/visualize_eval_results.py
outputs/figures/
```

建议生成：

```text
overall_scores.png
tfidf_vs_bge.png
dimension_radar_high_vs_low.png
dimension_heatmap.png
evidence_topk_similarity.png
skill_hit_missing.png
```

这些图应该补充到：

```text
docs/experiment_report.md
```

## 7. 推荐交付文件

如果组员都有原始 GitHub 仓库，只需要交付本阶段修改/新增文件：

```text
.gitignore
core/keyword_extractor.py
core/scoring.py
core/embedding_model.py
core/llm_generator.py
scripts/check_models.py
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
docs/experiment_report.md
docs/VISUAL_HANDOFF.md
docs/RUN_THROUGH_README.md
```

不需要交付：

```text
models/
myenv/
myenv310/
__pycache__/
.cache/
```

## 8. 一句话总结

本阶段完成了模型下载、本地模型加载稳定性优化、Qwen 生成稳定性优化、实验评分修正和验证脚本补充。当前项目已经可以稳定复现实验结果，下一步应基于这些结果制作实验图表和报告可视化。
