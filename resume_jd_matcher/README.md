# 基于预训练语言模型的简历岗位匹配度评估与面试辅助系统

本项目是一个自然语言处理课程设计系统，面向“简历与岗位 JD 匹配评估”场景。用户输入或上传简历和岗位 JD 后，系统会自动完成文本读取、结构化切分、技能实体抽取、语义证据匹配、四维度评分、生成式分析和报告导出。

项目当前已经实现了一个可运行的 Streamlit 演示系统，适合课程答辩展示，也方便后续队友继续补充实验、截图、模型文件和报告内容。

## 1. 当前项目做了什么

系统主要完成以下任务：

1. 支持用户直接粘贴简历和岗位 JD 文本。
2. 支持上传 `txt`、`pdf`、`docx` 文件并自动读取文本。
3. 支持一键加载示例简历和示例 JD。
4. 将 JD 拆分成多条岗位要求 `requirement`。
5. 将简历拆分成教育背景、专业技能、项目经历、实习经历等证据片段 `evidence`。
6. 使用技能词典抽取简历和 JD 中的技能实体。
7. 使用 `BAAI/bge-small-zh-v1.5` 对 JD requirement 和简历 evidence 做语义匹配。
8. 对每条 JD requirement 检索 Top-3 简历证据，并展示相似度和匹配状态。
9. 使用四维度评分模型输出综合匹配度和匹配等级。
10. 使用 TF-IDF baseline 与 BGE 语义匹配结果做对比实验。
11. 使用 `Qwen/Qwen2.5-1.5B-Instruct` 生成优势、短板、优化建议和面试问题。
12. 当 Qwen 不可用时，自动进入 `template_fallback`，保证系统仍然可演示。
13. 支持导出 JSON 和 Markdown 分析报告。

系统输出内容包括：

- 综合匹配度
- 匹配等级
- 四个维度得分
- 命中关键词
- 缺失关键词
- JD requirement 与简历 evidence 的逐项匹配
- 优势分析
- 短板分析
- 简历优化建议
- 面试问题和回答思路
- TF-IDF baseline 对比结果
- JSON / Markdown 报告

## 2. 设计思路

本项目不是简单关键词匹配系统，而是围绕“可解释的语义证据匹配”设计。

传统关键词匹配容易出现两个问题：

- 简历和 JD 语义相近但关键词不同，系统可能误判为不匹配。
- 简历堆砌关键词但缺少真实项目证据，系统可能误判为高度匹配。

因此本项目采用以下思路：

1. 先对 JD 和简历做结构化切分。
2. 再对技能实体做词典匹配，得到命中技能和缺失技能。
3. 使用 BGE 预训练语义模型计算 JD requirement 与简历 evidence 的相似度。
4. 用 evidence 匹配结果约束最终评分，避免只看关键词。
5. 保留 TF-IDF baseline，用于说明预训练语义模型相对传统方法的优势。
6. 使用 Qwen2.5 将结构化匹配结果转成自然语言分析，辅助面试和简历优化。

课程设计答辩时可以强调：

- BGE 是主语义匹配模型。
- TF-IDF 只是 baseline。
- Qwen 是生成式分析模块，不参与四维主评分。
- 系统可解释性来自“每条 JD 要求都能追溯到简历证据”。

## 3. 整体框架

系统采用 Streamlit 前端加 Python 核心模块的结构。

```text
用户输入 / 文件上传
        |
        v
文本读取 file_reader.py
        |
        v
文本清洗与结构化切分 preprocess.py
        |
        +----------------------------+
        |                            |
        v                            v
技能实体抽取 skill_extractor.py     JD requirement / 简历 evidence
        |                            |
        v                            v
命中技能 / 缺失技能                  BGE 语义向量编码 embedding_model.py
        |                            |
        +-------------+--------------+
                      |
                      v
        证据匹配 evidence_matcher.py
                      |
                      v
        四维评分 scoring.py
                      |
        +-------------+--------------+
        |                            |
        v                            v
TF-IDF baseline 对比                 Qwen 生成式分析 llm_generator.py
        |                            |
        +-------------+--------------+
                      |
                      v
        Streamlit 页面展示 app.py
                      |
                      v
        JSON / Markdown 报告导出
```

## 4. 核心评分公式

综合匹配度采用四维度加权：

```text
综合匹配度 =
技能实体匹配分 × 30%
+ 语义证据匹配分 × 40%
+ 项目经历匹配分 × 20%
+ 软技能匹配分 × 10%
```

四个维度含义：

| 维度 | 权重 | 含义 |
| --- | --- | --- |
| 技能实体匹配 | 30% | JD 中技能实体被简历覆盖的比例 |
| 语义证据匹配 | 40% | 每条 JD 要求与简历证据的 BGE 语义相似度 |
| 项目经历匹配 | 20% | 项目或实习经历与岗位职责的相关性 |
| 软技能匹配 | 10% | 沟通协作、学习能力、文档撰写等软技能覆盖情况 |

匹配等级：

| 分数区间 | 等级 |
| --- | --- |
| 85-100 | 高度匹配 |
| 70-84 | 较高匹配 |
| 50-69 | 一般匹配 |
| 0-49 | 匹配度较低 |

注意：不要新增“大模型一致性评分”作为第五维，当前课程设计要求是四维评分。

## 5. 模型说明

当前项目使用两个 Hugging Face 模型：

| 用途 | 模型 | 说明 |
| --- | --- | --- |
| 语义匹配 | `BAAI/bge-small-zh-v1.5` | 主模型，用于 JD requirement 与简历 evidence 的向量相似度计算 |
| 生成式分析 | `Qwen/Qwen2.5-1.5B-Instruct` | 根据结构化结果生成优势、短板、建议和面试题 |

明确说明：

- 不使用 Ollama。
- 不需要执行 `ollama pull`。
- 不调用付费 API。
- 不从零训练模型。
- 不做模型微调。
- 如果 Qwen 不可用，系统会显示 `generation_mode = template_fallback`。

## 6. 当前模型状态

当前已经定位过两个关键问题：

### 6.1 BGE 语义匹配为 0 的原因

语义匹配为 0 通常不是评分公式错误，而是 BGE 没有成功加载。

已做修复：

- `core/embedding_model.py` 已改为优先读取项目本地模型目录。
- 如果本地目录不存在，再尝试 Hugging Face 本地缓存。
- 如果缓存也没有，才尝试联网下载。

验证结果：

- BGE 已验证可以从 Hugging Face 本地缓存加载。
- 示例数据中 `语义证据匹配` 可恢复到约 `78.00`。

### 6.2 Qwen 加载失败的原因

Qwen 模型名是正确的，config 和 tokenizer 也能识别。常见失败原因是本地缺少真正的权重文件：

```text
model.safetensors
```

如果报错类似：

```text
does not appear to have a file named pytorch_model.bin or model.safetensors
```

说明 Qwen 目录里只有配置和 tokenizer，没有权重文件。解决方法是把完整 Qwen 模型文件下载到本地 `models/Qwen2.5-1.5B-Instruct/`。

## 7. 队友接下来要做什么

接手项目后，建议按这个顺序做：

1. 进入项目目录。
2. 检查本地模型是否完整。
3. 如果 Qwen 缺少权重，手动下载模型文件到 `models/`。
4. 启动 Streamlit 页面。
5. 加载示例数据并截图。
6. 补充实验报告中的真实运行结果。
7. 根据答辩需要 polish 页面和文档。

推荐先读：

- `docs/AI_HANDOFF.md`
- `docs/design_report.md`
- `docs/experiment_report.md`

最短接管命令：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
.\myenv\Scripts\python.exe scripts\check_models.py
streamlit run app.py
```

如果没有激活虚拟环境，也可以显式运行：

```powershell
.\myenv\Scripts\python.exe -m streamlit run app.py
```

## 8. 本地模型下载与放置

项目支持把模型手动下载到本地目录，避免每次运行访问 Hugging Face。

目标目录：

```text
resume_jd_matcher/
└── models/
    ├── bge-small-zh-v1.5/
    └── Qwen2.5-1.5B-Instruct/
```

BGE 目录至少需要：

```text
models/bge-small-zh-v1.5/
├── config.json
├── modules.json
├── tokenizer.json 或 vocab.txt
└── model.safetensors
```

Qwen 目录至少需要：

```text
models/Qwen2.5-1.5B-Instruct/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── model.safetensors
```

模型检查命令：

```powershell
.\myenv\Scripts\python.exe scripts\check_models.py
```

理想结果：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 可用
```

如果只显示：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 不可用
```

说明主匹配评分可以正常工作，但生成式分析仍在模板降级模式，需要补齐 Qwen 权重。

## 9. 安装依赖

推荐使用虚拟环境：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
.\myenv\Scripts\activate
pip install -r requirements.txt
```

如果是新机器，可以新建环境：

```powershell
python -m venv myenv
.\myenv\Scripts\activate
pip install -r requirements.txt
```

主要依赖：

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `jieba`
- `sentence-transformers`
- `transformers`
- `torch`
- `pdfplumber`
- `python-docx`

## 10. 运行方法

进入项目目录：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
```

启动：

```powershell
streamlit run app.py
```

或：

```powershell
.\myenv\Scripts\python.exe -m streamlit run app.py
```

页面打开后：

1. 点击“加载示例数据”。
2. 点击“开始分析”。
3. 查看“模型状态”。
4. 查看“匹配总览”“逐项证据匹配”“技能分析”“生成式分析”“对比实验”“导出报告”。

## 11. 项目结构

```text
resume_jd_matcher/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── scripts/
│   └── check_models.py
├── core/
│   ├── baseline_tfidf.py
│   ├── embedding_model.py
│   ├── evidence_matcher.py
│   ├── file_reader.py
│   ├── keyword_extractor.py
│   ├── llm_generator.py
│   ├── matcher.py
│   ├── preprocess.py
│   ├── question_generator.py
│   ├── report_generator.py
│   ├── scoring.py
│   └── skill_extractor.py
├── data/
│   ├── eval_cases.json
│   ├── sample_jd.txt
│   ├── sample_resume.txt
│   └── skill_dict.json
├── docs/
│   ├── AI_HANDOFF.md
│   ├── design_report.md
│   └── experiment_report.md
└── models/
    ├── bge-small-zh-v1.5/
    └── Qwen2.5-1.5B-Instruct/
```

`models/` 不提交到 Git，已经由根目录 `.gitignore` 忽略。

## 12. 核心模块说明

| 文件 | 作用 |
| --- | --- |
| `app.py` | Streamlit 页面和主流程入口 |
| `config.py` | 模型名称、本地模型路径、阈值和生成参数 |
| `core/file_reader.py` | 读取 txt、pdf、docx 上传文件 |
| `core/preprocess.py` | 文本清洗、分词、简历 section 和 JD requirement 切分 |
| `core/skill_extractor.py` | 调用技能词典抽取技能实体 |
| `core/keyword_extractor.py` | 技能词典匹配和 TF-IDF 关键词提取 |
| `core/embedding_model.py` | 加载 BGE 并生成文本向量 |
| `core/evidence_matcher.py` | 计算 requirement 与 evidence 的语义相似度 |
| `core/scoring.py` | 四维评分公式 |
| `core/baseline_tfidf.py` | TF-IDF baseline |
| `core/llm_generator.py` | Qwen 生成式分析和模板降级 |
| `core/report_generator.py` | JSON / Markdown 报告导出 |
| `scripts/check_models.py` | 检查本地模型是否可用 |

## 13. 验证命令

编译检查：

```powershell
.\myenv\Scripts\python.exe -m py_compile app.py config.py core\baseline_tfidf.py core\embedding_model.py core\evidence_matcher.py core\file_reader.py core\keyword_extractor.py core\llm_generator.py core\matcher.py core\preprocess.py core\question_generator.py core\report_generator.py core\scoring.py core\skill_extractor.py
```

模型检查：

```powershell
.\myenv\Scripts\python.exe scripts\check_models.py
```

Git 状态检查：

```powershell
git status --short
```

注意：`myenv/`、`venv/`、`models/`、`__pycache__/` 不应进入 Git。

## 14. 实验与答辩建议

队友后续建议补充：

1. 使用 `data/eval_cases.json` 跑 6 组样例。
2. 记录每组的 TF-IDF baseline、BGE 语义证据分、综合分和匹配等级。
3. 截图保存以下页面：
   - 匹配总览
   - 逐项证据匹配
   - 技能分析
   - 生成式分析
   - 对比实验
4. 在 `docs/experiment_report.md` 中补真实分数表。
5. 在 `docs/design_report.md` 中补系统截图。

答辩时建议重点讲：

- 为什么不用简单关键词匹配。
- 为什么要把 JD 和简历拆成 requirement / evidence。
- BGE 如何解决语义相近但词面不同的问题。
- TF-IDF baseline 的局限。
- Qwen 生成式分析如何辅助面试，但不直接参与四维评分。
- 模板降级如何保证系统稳定性。

## 15. 常见问题

### Q1：还需要 Ollama 吗？

不需要。本项目已经改为 Hugging Face / Sentence-Transformers 路线，不使用 Ollama。

### Q2：Qwen 加载失败是不是模型名错了？

通常不是。已验证 `Qwen/Qwen2.5-1.5B-Instruct` 的 config 和 tokenizer 可识别。最常见原因是缺少 `model.safetensors` 权重文件。

### Q3：BGE 语义匹配为 0 怎么办？

先运行：

```powershell
.\myenv\Scripts\python.exe scripts\check_models.py
```

如果 BGE 不可用，优先把模型下载到：

```text
models/bge-small-zh-v1.5/
```

### Q4：Qwen 不可用时项目还能答辩吗？

可以，但需要说明当前处于 `template_fallback`。主匹配评分依赖 BGE，不依赖 Qwen。若要体现“生成式模型”效果，最好补齐 Qwen 权重。

### Q5：为什么不从零训练模型？

课程设计周期和硬件条件有限，从零训练大模型需要大量数据和 GPU。使用预训练语言模型更符合本项目目标，也更容易完成可运行系统和实验对比。

## 16. 当前交接结论

当前项目已经完成主体功能：

- Streamlit 页面可运行。
- 示例数据可加载。
- txt / pdf / docx 文件读取已实现。
- 技能实体抽取已实现。
- BGE 语义证据匹配链路已实现。
- 四维评分已实现。
- TF-IDF baseline 对比已实现。
- Qwen 生成式分析代码已实现。
- Qwen 不可用时模板降级已实现。
- JSON / Markdown 报告导出已实现。
- 课程设计报告和实验报告已有初稿。

队友接下来最应该做的是：

1. 下载并放置本地模型，尤其是 Qwen 的 `model.safetensors`。
2. 运行 `scripts/check_models.py`。
3. 跑通 Streamlit 页面并截图。
4. 补全实验结果表。
5. 根据答辩要求继续完善文档和演示材料。
