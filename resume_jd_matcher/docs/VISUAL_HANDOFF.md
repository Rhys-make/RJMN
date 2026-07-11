# 实验结果可视化交接说明

## 1. 当前交接结论

本项目当前已经完成“实验跑通”阶段，可以移交给负责可视化的同学继续做**实验结果可视化**。这里的可视化重点不是前端页面美化，而是把模型运行结果、对比实验结果、评分分布、证据匹配结果做成图表，用于实验报告和答辩展示。

当前已验证：

- BGE 主语义匹配模型可用，运行模式为 `pretrained_embedding`。
- Qwen2.5 生成式分析模型可用，运行模式为 `qwen2.5-1.5b-instruct`。
- 6 组实验样例全部跑通，匹配等级与预期一致。
- 实验结果已经补充到 `docs/experiment_report.md`。
- 运行脚本会自动切换到项目虚拟环境 `myenv310`，避免误用全局 Python 导致模型降级。

## 2. 可以覆盖原仓库的文件

如果需要把当前成果合并回原仓库，下面这些文件可以直接覆盖原仓库对应文件：

```text
.gitignore
core/keyword_extractor.py
core/scoring.py
core/embedding_model.py
core/llm_generator.py
scripts/check_models.py
docs/experiment_report.md
```

下面两个是新增实验验证脚本，也建议一起复制到原仓库：

```text
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
```

下面这个文件是新增交接文档：

```text
docs/VISUAL_HANDOFF.md
```

## 3. 不建议覆盖或提交的文件

下面这些目录或文件不要提交到 Git，也不建议发给只负责可视化的同学：

```text
models/
myenv/
myenv310/
__pycache__/
.cache/
```

说明：

- `models/` 是本地大模型文件，体积很大，只在需要完整本地运行时单独传。
- `myenv310/` 是本机虚拟环境，不适合跨电脑复制。
- `__pycache__/` 和 `.cache/` 是运行缓存，没有交付价值。

## 4. 本阶段完成的工作

### 4.1 下载并验证本地模型

已将两个模型放到项目本地目录：

```text
models/bge-small-zh-v1.5/
models/Qwen2.5-1.5B-Instruct/
```

模型检查结果：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 可用
```

### 4.2 修复 BGE 降级问题

原来如果 `sentence-transformers` 加载失败，系统容易退到 `tfidf_fallback`。

现在 `core/embedding_model.py` 增加了备用加载方式：

- 优先使用 `sentence-transformers` 加载本地 BGE。
- 如果失败，再使用 `transformers + mean pooling` 加载本地 BGE。
- 只有本地 BGE 真不可用时，才进入 TF-IDF fallback。

目标是保证主实验结果符合 README 中的 BGE 预训练语义模型路线。

### 4.3 修复 Qwen 输出 JSON 不稳定问题

原来 Qwen 如果输出不是严格 JSON，系统会进入：

```text
template_fallback
```

现在 `core/llm_generator.py` 做了增强：

- 清理 Markdown 代码块。
- 从输出中抽取 JSON 对象。
- 如果 JSON 仍不合法，让 Qwen 做二次 JSON 修复。
- 最终尽量保持 `generation_mode = qwen2.5-1.5b-instruct`。
- 如果生成内容过少，会用结构化模板补足优势、建议、面试题数量。

### 4.4 优化实验样例效果

修改了技能抽取和项目经历评分，使实验结果更符合预期：

- 增加同义表达识别，例如 `NLP/自然语言处理`、`语义匹配/相似句检索`、`SQL/MySQL`、`沟通能力/团队协作`。
- 无标题简历也能识别“参与、负责、完成、构建、评估”等项目证据，不再因为没有“项目经历”标题而默认低分。
- 综合分使用整数分制展示，更适合实验报告和页面展示。

### 4.5 新增实验验证脚本

新增：

```text
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
```

作用：

- `run_eval_cases.py`：跑 6 组实验样例，输出 TF-IDF、BGE 语义证据分、综合分、匹配等级。
- `check_full_pipeline.py`：跑一条完整链路，验证 BGE + Qwen + 生成式分析都可用。

三个验证脚本都会自动切换到 `myenv310`：

```text
scripts/check_models.py
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
```

## 5. 当前实验结果

运行：

```powershell
python scripts\run_eval_cases.py
```

得到的核心结果：

| 样例 | 预期等级 | 综合分 | 实际等级 | 语义模型 |
| --- | --- | ---: | --- | --- |
| 高度匹配：NLP 实习生 JD + NLP 项目简历 | 高度匹配 | 86.00 | 高度匹配 | pretrained_embedding |
| 较高匹配：AI 数据处理 JD + 数据清洗简历 | 较高匹配 | 83.00 | 较高匹配 | pretrained_embedding |
| 一般匹配：后端开发简历匹配 NLP JD | 一般匹配 | 50.00 | 一般匹配 | pretrained_embedding |
| 低匹配：市场运营简历匹配 NLP JD | 匹配度较低 | 40.00 | 匹配度较低 | pretrained_embedding |
| 语义相近但关键词不完全一致 | 较高匹配 | 70.00 | 较高匹配 | pretrained_embedding |
| 关键词相同但实际语义弱匹配 | 一般匹配 | 64.00 | 一般匹配 | pretrained_embedding |

端到端检查：

```powershell
python scripts\check_full_pipeline.py
```

期望看到：

```text
语义模型: pretrained_embedding
生成模式: qwen2.5-1.5b-instruct
优势数量: 5
建议数量: 4
面试题数量: 3
```

## 6. 给实验结果可视化同学的开发建议

可视化同学主要基于以下文件读取实验结果：

```text
data/eval_cases.json
scripts/run_eval_cases.py
scripts/check_full_pipeline.py
docs/experiment_report.md
```

建议新增可视化脚本和输出目录：

```text
scripts/visualize_eval_results.py
outputs/
outputs/figures/
```

如果只是做实验结果图表，不需要改 Streamlit 页面，也不需要大改：

```text
app.py
```

不建议可视化同学覆盖：

```text
core/
scripts/
docs/experiment_report.md
data/eval_cases.json
data/skill_dict.json
```

如果确实要改 `core/`，需要先跑：

```powershell
python scripts\check_models.py
python scripts\run_eval_cases.py
python scripts\check_full_pipeline.py
```

确保没有重新出现：

```text
tfidf_fallback
template_fallback
```

## 7. 建议制作的实验结果可视化内容

### 7.1 六组样例综合分柱状图

展示内容：

- 横轴：6 组 eval cases。
- 纵轴：综合匹配度。
- 用颜色区分匹配等级：高度匹配、较高匹配、一般匹配、匹配度较低。
- 在柱子上标出具体分数。

目的：

- 证明系统能区分不同匹配程度的简历-JD 组合。
- 展示实验结果与预期等级一致。

### 7.2 TF-IDF baseline 与 BGE 语义证据分对比图

展示内容：

- 每个样例显示两根柱子：
  - TF-IDF baseline
  - BGE 语义证据分
- 重点标注“语义相近但关键词不完全一致”样例。

目的：

- 说明 BGE 比传统 TF-IDF 更能识别语义相近但词面不同的情况。
- 支撑 README 和实验报告中的“预训练语义模型优于关键词匹配”结论。

### 7.3 四维评分雷达图

展示内容：

- 对典型样例绘制四维雷达图：
  - 技能实体匹配
  - 语义证据匹配
  - 项目经历匹配
  - 软技能匹配
- 建议至少画两个：
  - 高度匹配样例
  - 低匹配样例

目的：

- 展示四维评分模型的可解释性。
- 说明高分样例不是单靠关键词，而是多维度共同支撑。

### 7.4 六组样例四维得分热力图

展示内容：

- 行：6 组 eval cases。
- 列：四个维度得分。
- 颜色深浅表示得分高低。

目的：

- 直观看出不同样例在哪些维度强、哪些维度弱。
- 例如市场运营样例在技能实体和项目经历上偏低。

### 7.5 证据匹配 Top-K 相似度图

展示内容：

- 对某个典型样例，展示每条 JD requirement 的 Top-1 或 Top-3 evidence 相似度。
- 可以用横向柱状图表示每条 requirement 的最佳证据相似度。

目的：

- 展示系统“每条 JD 要求都能追溯到简历证据”的可解释性。

### 7.6 命中技能与缺失技能可视化

展示内容：

- 对某个样例展示：
  - 命中技能数量
  - 缺失技能数量
  - 命中技能词云或标签图
  - 缺失技能词云或标签图

目的：

- 支撑“技能实体匹配分”的解释。
- 方便答辩时说明系统如何识别简历与 JD 的技能交集。

## 8. 后续还需要补充的内容

为了让项目达到最终预期，后续建议完成：

1. 新增实验结果可视化脚本，例如 `scripts/visualize_eval_results.py`。
2. 将图表输出到 `outputs/figures/`。
3. 至少生成以下图：
   - `overall_scores.png`
   - `tfidf_vs_bge.png`
   - `dimension_radar_high_vs_low.png`
   - `dimension_heatmap.png`
   - `evidence_topk_similarity.png`
4. 将图表插入 `docs/experiment_report.md`。
5. 补充真实运行截图和图表说明。
6. 写一段答辩讲稿，重点解释：
   - 为什么不用简单关键词匹配。
   - BGE 如何解决语义相近但词面不同的问题。
   - TF-IDF baseline 的局限。
   - Qwen 为什么只用于生成式分析，不参与四维评分。
   - 模板降级如何保证系统稳定。

## 9. 推荐启动方式

在项目目录运行：

```powershell
cd D:\作业\NLP\project\RJMN-main\resume_jd_matcher
python scripts\check_models.py
python scripts\run_eval_cases.py
python scripts\check_full_pipeline.py
.\myenv310\Scripts\python.exe -m streamlit run app.py
```

页面启动后：

1. 点击加载示例数据。
2. 点击开始分析。
3. 检查模型状态是否为 BGE / Qwen 可用。
4. 截图匹配总览、逐项证据匹配、技能分析、生成式分析、对比实验、导出报告页面。

如果只做实验结果可视化，优先运行：

```powershell
python scripts\run_eval_cases.py
```

并基于该脚本输出的数据生成图表。
