# 基于预训练语言模型的简历岗位匹配度评估与面试辅助系统

本项目是自然语言处理课程设计系统，面向“简历与岗位 JD 匹配评估”场景。系统使用 Hugging Face / Sentence-Transformers 预训练模型完成语义表示和生成式分析，不依赖付费 API，不从零训练大模型，也不进行大模型微调。

## 队友接管入口

如果你是队友或准备让 AI 助手继续接管本项目，请先阅读：

- `docs/AI_HANDOFF.md`
- `docs/design_report.md`
- `docs/experiment_report.md`

最短接管流程：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
.\myenv\Scripts\python.exe scripts\check_models.py
.\myenv\Scripts\python.exe -m py_compile app.py config.py core\*.py
streamlit run app.py
```

当前关键状态：

- 不使用 Ollama，不需要 `ollama pull`。
- 语义主模型是 `BAAI/bge-small-zh-v1.5`。
- 生成模型是 `Qwen/Qwen2.5-1.5B-Instruct`。
- BGE 已验证可以从 Hugging Face 本地缓存加载。
- Qwen 若缺少 `model.safetensors`，会进入 `template_fallback`，这是待补齐的本地模型文件问题，不是模型名称错误。

## 技术亮点

- 主方法使用 `BAAI/bge-small-zh-v1.5` 进行语义向量表示。
- 将 JD 拆分为独立 requirement，将简历拆分为教育背景、专业技能、项目经历、实习经历等 evidence。
- 对每条 JD requirement 检索 Top-3 简历证据片段，并展示相似度和匹配状态。
- 基于技能词典抽取命中技能、缺失技能和按类别统计结果。
- 使用四维评分模型：技能实体匹配、语义证据匹配、项目经历匹配、软技能匹配。
- 生成式分析优先使用 `Qwen/Qwen2.5-1.5B-Instruct`。
- Qwen 加载失败时自动进入 `template_fallback`，页面和导出报告会明确显示。
- 保留 TF-IDF baseline，用于与 BGE 语义匹配结果进行对比实验。
- 支持 JSON 和 Markdown 报告导出。

## 模型说明

- 语义匹配模型：`BAAI/bge-small-zh-v1.5`
- 生成式分析模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 备用说明：如果本地机器性能不足，可在 `config.py` 中将生成模型改为 `Qwen/Qwen2.5-0.5B-Instruct`
- 降级模式：如果生成式模型加载失败，系统使用模板生成，并标明 `generation_mode = template_fallback`

## 为什么使用预训练模型而不是从零训练

从零训练语言模型需要大量语料、GPU 算力和训练时间，不适合短周期课程设计。本项目使用预训练语言模型进行语义表示和生成式分析，可以在有限时间内完成高质量 NLP 应用系统，同时保留模型状态、证据匹配和 baseline 对比，便于答辩说明。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果网络环境可以稳定访问 Hugging Face，第一次运行会自动下载模型。若网络不稳定，推荐手动下载模型并放入本项目的 `models/` 目录：

```text
resume_jd_matcher/
└── models/
    ├── bge-small-zh-v1.5/
    │   ├── config.json
    │   ├── modules.json
    │   ├── tokenizer.json 或 vocab.txt
    │   └── model.safetensors
    └── Qwen2.5-1.5B-Instruct/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.json
        ├── merges.txt
        └── model.safetensors
```

模型检查命令：

```bash
python scripts/check_models.py
```

检查结果中应显示：

- `BGE 语义匹配: 可用`
- `Qwen 生成分析: 可用`

## 运行方法

```bash
streamlit run app.py
```

如果当前位于仓库根目录：

```bash
cd resume_jd_matcher
streamlit run app.py
```

## 项目结构

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
│   ├── llm_generator.py
│   ├── preprocess.py
│   ├── report_generator.py
│   ├── scoring.py
│   └── skill_extractor.py
├── data/
│   ├── eval_cases.json
│   ├── sample_resume.txt
│   ├── sample_jd.txt
│   └── skill_dict.json
└── docs/
    ├── AI_HANDOFF.md
    ├── design_report.md
    └── experiment_report.md
```

## 示例输入输出

点击“加载示例数据”后再点击“开始分析”，系统会输出：

- 综合匹配度和匹配等级
- 四个维度得分与柱状图
- JD requirement 与简历 evidence 的逐项匹配表
- 命中技能和缺失技能
- Qwen 生成式分析或模板降级分析
- TF-IDF baseline 与 BGE 语义模型分数对比
- JSON 和 Markdown 报告下载按钮

## 常见问题

1. BGE 模型下载失败怎么办？
   - 页面会显示模型加载错误，系统不会崩溃，并会使用 baseline fallback 保证页面可展示。

2. Qwen 模型下载或加载失败怎么办？
   - 系统会显示 `template_fallback`，使用基于命中技能、缺失技能和 evidence 的动态模板生成分析。

3. 是否训练或微调了大模型？
   - 没有。本项目使用预训练模型进行语义表示和生成式分析。
