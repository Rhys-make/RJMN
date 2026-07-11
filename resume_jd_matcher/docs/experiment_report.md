# 实验报告：简历岗位匹配度评估系统

## 1. 实验目的

验证 BGE 预训练语义模型在简历岗位匹配任务中的效果，并与 TF-IDF baseline 进行对比，同时观察 Qwen2.5 生成式分析对结果解释的增强作用。

## 2. 测试样例说明

实验数据位于 `data/eval_cases.json`，包含 6 组样例：高度匹配、较高匹配、一般匹配、低匹配、关键词不完全一致但语义相近、关键词相同但实际语义弱匹配。

## 3. TF-IDF baseline 方法

TF-IDF baseline 使用 `TfidfVectorizer` 将简历和 JD 转换为字符 n-gram 向量，并使用余弦相似度计算整体相似度。该方法作为传统词面匹配基线。

## 4. BGE 预训练语义模型方法

BGE 方法将 JD requirement 和简历 evidence 分别编码为语义向量。系统对每条 requirement 检索 Top-3 evidence，并根据最佳相似度判断“高度匹配、部分匹配、匹配不足”。

## 5. 不同样例下的匹配结果

| 样例 | 预期等级 | 观察重点 |
| --- | --- | --- |
| NLP 实习生 JD + NLP 项目简历 | 高度匹配 | 技能和项目 evidence 均较强 |
| AI 数据处理 JD + 数据清洗简历 | 较高匹配 | 数据处理能力强，NLP 深度略不足 |
| 后端开发简历匹配 NLP JD | 一般匹配 | 编程相关，但 NLP evidence 不足 |
| 市场运营简历匹配 NLP JD | 匹配度较低 | 技术技能和项目证据不足 |
| 关键词不完全一致但语义相近 | 较高匹配 | BGE 应优于 TF-IDF |
| 关键词相同但实际语义弱匹配 | 一般匹配 | evidence 匹配可降低关键词堆砌误判 |

## 6. 结果分析

TF-IDF 的优势是实现简单、速度快，但对同义表达敏感。BGE 预训练语义模型能够利用上下文语义表示，适合发现“表达不同但岗位能力相近”的证据。Qwen2.5 生成式模型将结构化分数和 evidence 转换为可读的优势、短板、建议和面试题，提升了系统的实用价值。

## 7. 结论

实验表明，BGE 语义证据匹配比单纯 TF-IDF 更适合简历岗位匹配任务。结合 Qwen2.5 生成式分析后，系统不仅能给出匹配分数，还能解释匹配原因并辅助面试准备。若生成模型不可用，模板降级机制可以保证系统稳定展示。

## 8. 复现实验与交接记录

队友接手后建议按以下顺序复现实验：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
.\myenv\Scripts\python.exe scripts\check_models.py
streamlit run app.py
```

进入页面后：

1. 点击“加载示例数据”。
2. 点击“开始分析”。
3. 记录四维得分、综合匹配度和匹配等级。
4. 截图保存“匹配总览”“逐项证据匹配”“生成式分析”“对比实验”四个页面。
5. 若 Qwen 权重未补齐，记录 `generation_mode = template_fallback`；若 Qwen 可用，记录 `generation_mode = qwen2.5-1.5b-instruct`。

当前已定位的模型状态说明：

- BGE 若可用，样例数据的 `语义证据匹配` 不应为 0。
- Qwen 加载失败最常见原因是缺少 `model.safetensors` 权重文件。
- Qwen 降级不会影响 BGE 主语义匹配和四维评分，但会影响生成式分析是否由真实预训练生成模型完成。

后续可以将 6 组 `data/eval_cases.json` 的运行结果整理成表格，补充到本报告中：

| 样例 | TF-IDF baseline | BGE 语义证据分 | 综合分 | 匹配等级 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 高度匹配样例 | 24.45 | 73.95 | 86.00 | 高度匹配 | 技能和项目证据均较强 |
| 较高匹配样例 | 19.45 | 85.76 | 83.00 | 较高匹配 | 数据处理能力较强 |
| 一般匹配样例 | 8.04 | 63.19 | 50.00 | 一般匹配 | NLP 证据不足，但编程和系统经历有一定迁移性 |
| 低匹配样例 | 1.86 | 54.75 | 40.00 | 匹配度较低 | 技术技能不足 |
| 语义相近样例 | 2.40 | 64.85 | 70.00 | 较高匹配 | TF-IDF 很低，但 BGE 能识别意图识别、向量检索等语义相近证据 |
| 关键词堆砌样例 | 12.93 | 71.55 | 64.00 | 一般匹配 | 虽然关键词重合，但项目 evidence 较弱，综合分未被误判为高匹配 |

本次验证命令：

```powershell
.\myenv310\Scripts\python.exe scripts\check_models.py
.\myenv310\Scripts\python.exe scripts\run_eval_cases.py
.\myenv310\Scripts\python.exe scripts\check_full_pipeline.py
```

验证结果：

- BGE 语义匹配模型从项目本地 `models/bge-small-zh-v1.5/` 成功加载。
- Qwen2.5-1.5B-Instruct 从项目本地 `models/Qwen2.5-1.5B-Instruct/` 成功加载。
- 端到端生成模式为 `qwen2.5-1.5b-instruct`，不是 `template_fallback`。
- `scripts/check_models.py`、`scripts/run_eval_cases.py`、`scripts/check_full_pipeline.py` 会自动切换到项目虚拟环境 `myenv310`，避免误用全局 Python 导致模型降级。
- BGE 加载已增加 `transformers-mean-pooling` 备用后端；如果 `sentence-transformers` 后端异常，仍会优先使用本地 BGE 权重，而不是直接退回 TF-IDF。
- Qwen 输出已增加 JSON 抽取、代码块清理和二次 JSON 修复，减少因格式问题进入 `template_fallback` 的概率。

## 9. 高价值实验分析补充

为了进一步证明系统不是简单关键词匹配，本项目新增了三类高价值实验分析，详见：

```text
docs/high_value_analysis.md
```

生成命令：

```powershell
python scripts\generate_high_value_analysis.py
```

该脚本不会改动核心算法代码，只基于当前模型输出生成报告材料。

### 9.1 Top-K Evidence 可解释性分析

系统将 JD 拆解为多条 requirement，并为每条 requirement 检索简历中的 Top-3 evidence。该分析可以直接回答：

- 每条岗位要求是否能在简历中找到证据。
- 最相关的简历证据是什么。
- 相似度是多少。
- 匹配状态是高度匹配、部分匹配还是匹配不足。

这体现了本系统相较于普通关键词匹配系统的核心优势：不仅输出综合分，还能追溯每一条分数背后的文本证据。

### 9.2 四维评分消融实验

新增消融分析用于验证四维评分设计的必要性。实验分别去掉：

- 技能实体匹配
- 语义证据匹配
- 项目经历匹配
- 软技能匹配

并将剩余维度按原权重比例重新归一化计算分数。消融实验说明，完整四维模型能够同时利用技能覆盖、语义证据、项目经历和软技能信息，避免单一指标造成误判。

### 9.3 典型案例分析

新增两个边界案例分析：

| 案例 | 观察重点 | 实验结论 |
| --- | --- | --- |
| 语义相近但关键词不完全一致 | TF-IDF 分数很低，但 BGE 语义证据分较高 | BGE 能识别词面不同但语义相近的能力表达 |
| 关键词相同但实际语义弱匹配 | 简历出现关键词，但缺少真实项目 evidence | 四维评分限制其综合分，避免关键词堆砌误判 |

这两个案例分别说明：

- BGE 可以弥补传统词面匹配的不足。
- evidence 约束和四维评分可以降低关键词堆砌带来的误判风险。
