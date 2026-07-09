# 项目交接说明：给队友和 AI 助手

## 1. 项目当前定位

项目名称：`resume-jd-match-nlp`

正式题目：基于预训练语言模型的简历岗位匹配度评估与面试辅助系统

这是一个自然语言处理课程设计项目，核心目标是输入简历和岗位 JD，输出：

- 综合匹配度
- 匹配等级
- 四个维度得分
- 命中技能与缺失技能
- JD requirement 与简历 evidence 的逐项语义匹配
- 优势分析
- 短板分析
- 简历优化建议
- 面试问题与回答思路
- JSON / Markdown 报告

项目强调“预训练语言模型应用”，不是从零训练模型，也不进行微调。

## 2. 当前技术路线

主方法：

- 语义匹配模型：`BAAI/bge-small-zh-v1.5`
- 生成式分析模型：`Qwen/Qwen2.5-1.5B-Instruct`
- baseline：TF-IDF 字符 n-gram 余弦相似度
- 前端：Streamlit

明确不要做：

- 不使用 Ollama。
- 不调用付费 API。
- 不把 TF-IDF 当主方法。
- 不增加“大模型一致性评分”作为第五维。
- 不随意改动四维评分公式。

四维评分公式：

```text
综合匹配度 =
技能实体匹配分 × 30%
+ 语义证据匹配分 × 40%
+ 项目经历匹配分 × 20%
+ 软技能匹配分 × 10%
```

匹配等级：

- 85-100：高度匹配
- 70-84：较高匹配
- 50-69：一般匹配
- 0-49：匹配度较低

## 3. 重要目录和文件

```text
resume_jd_matcher/
├── app.py                         # Streamlit 前端和主流程入口
├── config.py                      # 模型名称、本地模型路径、阈值配置
├── requirements.txt               # Python 依赖
├── README.md                      # 项目说明和运行方法
├── scripts/
│   └── check_models.py            # 本地模型检查脚本
├── core/
│   ├── preprocess.py              # 文本清洗、简历 section / JD requirement 切分
│   ├── file_reader.py             # txt / pdf / docx 文件读取
│   ├── skill_extractor.py         # 技能实体抽取
│   ├── keyword_extractor.py       # 技能词典与 TF-IDF 关键词
│   ├── embedding_model.py         # BGE 模型加载和编码
│   ├── evidence_matcher.py        # JD requirement 到简历 evidence 的 Top-K 匹配
│   ├── scoring.py                 # 四维评分公式
│   ├── baseline_tfidf.py          # TF-IDF baseline
│   ├── llm_generator.py           # Qwen 生成式分析与模板降级
│   ├── report_generator.py        # JSON / Markdown 报告导出
│   └── matcher.py                 # 兼容入口，内部走新版流程
├── data/
│   ├── sample_resume.txt          # 示例简历
│   ├── sample_jd.txt              # 示例 JD
│   ├── skill_dict.json            # 技能词典
│   └── eval_cases.json            # 实验样例
└── docs/
    ├── AI_HANDOFF.md              # 当前交接文档
    ├── design_report.md           # 课程设计报告
    └── experiment_report.md       # 实验报告
```

## 4. 环境与运行

推荐在项目目录运行：

```powershell
cd C:\Users\25138\project\RJMN\resume_jd_matcher
```

如果已有虚拟环境：

```powershell
.\myenv\Scripts\activate
```

安装依赖：

```powershell
pip install -r requirements.txt
```

启动系统：

```powershell
streamlit run app.py
```

或显式使用当前虚拟环境：

```powershell
.\myenv\Scripts\python.exe -m streamlit run app.py
```

## 5. 模型文件放置方式

项目现在支持优先读取本地模型目录，避免每次运行时访问 Hugging Face。

本地模型目录：

```text
resume_jd_matcher/
└── models/
    ├── bge-small-zh-v1.5/
    └── Qwen2.5-1.5B-Instruct/
```

BGE 目录至少需要：

```text
config.json
modules.json
tokenizer.json 或 vocab.txt
model.safetensors
```

Qwen 目录至少需要：

```text
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
model.safetensors
```

检查模型：

```powershell
.\myenv\Scripts\python.exe scripts\check_models.py
```

期望结果：

```text
BGE 语义匹配: 可用
Qwen 生成分析: 可用
```

当前已知状态：

- BGE 已验证可以从 Hugging Face 本地缓存加载，样例语义证据匹配分约为 `78.00`。
- Qwen 的模型名、config、tokenizer 没问题。
- Qwen 如果报缺少 `model.safetensors`，说明权重文件没有下载完整，需要手动补齐。

## 6. 快速验证命令

编译检查：

```powershell
.\myenv\Scripts\python.exe -m py_compile app.py config.py core\baseline_tfidf.py core\embedding_model.py core\evidence_matcher.py core\file_reader.py core\keyword_extractor.py core\llm_generator.py core\matcher.py core\preprocess.py core\question_generator.py core\report_generator.py core\scoring.py core\skill_extractor.py
```

模型检查：

```powershell
.\myenv\Scripts\python.exe scripts\check_models.py
```

Streamlit 启动：

```powershell
streamlit run app.py
```

功能检查：

1. 点击“加载示例数据”。
2. 点击“开始分析”。
3. 检查“模型状态”区域。
4. 检查四维度得分是否出现。
5. 检查“逐项证据匹配”是否有 Top-3 evidence。
6. 检查“生成式分析”是否显示 Qwen 或 `template_fallback`。
7. 检查 JSON / Markdown 是否可以导出。

## 7. 已定位过的问题

### 7.1 语义证据匹配为 0

根因不是评分公式，而是 BGE 没有成功加载。旧现象通常是：

```text
预训练语义模型 BAAI/bge-small-zh-v1.5 加载失败
Cannot send a request, as the client has been closed.
```

处理方式：

- 优先把 BGE 手动放到 `models/bge-small-zh-v1.5/`。
- 或确保 Hugging Face 本地缓存中已有完整模型。
- 运行 `scripts/check_models.py` 验证。

修复后，样例中 `语义证据匹配` 应明显高于 0，当前验证约为 `78.00`。

### 7.2 Qwen 加载失败

如果报错类似：

```text
does not appear to have a file named pytorch_model.bin or model.safetensors
```

说明 Qwen 目录中只有 config/tokenizer，没有权重文件。需要补齐：

```text
model.safetensors
```

这不是模型名称错误。已验证：

- `Qwen/Qwen2.5-1.5B-Instruct` 的 config 可用。
- tokenizer 可用。
- 架构为 `Qwen2ForCausalLM`。

### 7.3 Git 显示很多文件

仓库根目录是：

```text
C:\Users\25138\project\RJMN
```

根目录 `.gitignore` 已忽略：

- `myenv/`
- `venv/`
- `models/`
- `__pycache__/`
- `*.safetensors`
- `*.bin`
- 日志和导出报告

`?? resume_jd_matcher/` 表示项目源码还未被 Git 跟踪，不代表虚拟环境被加入暂存区。

## 8. 给后续 AI 助手的工作原则

接手后请先做：

1. 阅读 `README.md`、`docs/AI_HANDOFF.md`、`docs/design_report.md`。
2. 运行 `git status --short`，确认未跟踪/已修改文件。
3. 运行 `scripts/check_models.py`，确认模型状态。
4. 不要使用 Ollama。
5. 不要把 `models/`、`myenv/`、`venv/` 加入 Git。
6. 不要把“加保底方案”当成主要修复。模型异常要先定位真实原因。
7. 修改核心算法前，先确认是否违反四维评分公式。

适合继续优化的方向：

- 扩展 `data/skill_dict.json`，提升技能实体覆盖。
- 增加更多 `data/eval_cases.json` 实验样例。
- 在 `docs/experiment_report.md` 中补充真实运行截图和分数表。
- 优化 `preprocess.py` 的简历/JD 结构化切分规则。
- 在 Qwen 权重补齐后，记录 Qwen 与 `template_fallback` 的输出差异。

不建议当前阶段做：

- 引入复杂前端框架。
- 引入付费 API。
- 从零训练或微调大模型。
- 删除模板降级逻辑。
- 把 TF-IDF baseline 替代 BGE 主方法。

## 9. 当前交接结论

当前项目已经具备完整课程设计系统形态：

- Streamlit 页面可运行。
- 文件上传、示例数据、四维评分、证据匹配、关键词分析、报告导出已实现。
- BGE 语义匹配链路已验证可用。
- Qwen 生成链路的代码已完成，当前主要依赖本地权重文件是否补齐。

后续接手重点不是重写项目，而是：

1. 补齐 Qwen 本地权重。
2. 运行模型检查脚本。
3. 补充实验截图和答辩材料。
4. 做少量界面和文档 polish。
