# 可视化部分说明

## 负责内容

本部分由可视化同学完成，主要任务是为课程报告和答辩 PPT 生成模型输入输出的图表（In and Out 可视化），不涉及 Streamlit 前端页面改动，也未修改 `core/` 核心代码。

## 新增文件

```text
scripts/visualize.py            # 可视化生成脚本（主交付物）
outputs/                        # 生成的图表输出目录
├── 01_wordcloud.png            # 简历 & JD 词云
├── 02_similarity_heatmap.png   # JD要求 × 简历证据 语义相似度热力图
├── 03_vector_scatter.png       # BGE 向量 t-SNE/PCA 降维散点图
├── 04_tfidf_vs_bge.png         # TF-IDF vs BGE 6组样例对比
├── 05_radar_chart.png          # 四维匹配得分雷达图
├── 06_score_comparison.png     # 6组样例综合匹配度对比
├── 07_skill_coverage.png       # 技能覆盖率饼图 + 分类柱状图
├── 08_match_status_distribution.png  # JD要求匹配状态分布
├── 09_similarity_histogram.png # 相似度分数分布直方图
└── 10_dimension_stacked.png    # 6组样例四维度加权贡献堆叠图
```

## 图表说明

| 图表 | 展示内容 | 用途 |
|------|----------|------|
| 词云 | 简历和JD的高频词分布 | 展示模型输入文本的特征 |
| 相似度热力图 | BGE 计算的 requirement-evidence 余弦相似度矩阵 | 展示语义匹配模型的核心输出 |
| 向量散点图 | BGE 编码后的高维向量降维到2D | 展示语义空间中JD要求与简历证据的分布关系 |
| TF-IDF vs BGE | 6组实验样例下两种方法的得分对比 | 论证预训练模型优于传统方法 |
| 雷达图 | 四维度（技能实体/语义证据/项目经历/软技能）得分 | 展示评分模型的多维输出 |
| 综合分对比 | 6组样例的最终匹配度和等级 | 展示系统端到端输出效果 |
| 技能覆盖率 | JD技能命中/缺失比例及各类别分布 | 展示技能实体抽取模块的输出 |
| 匹配状态分布 | 各JD要求的匹配状态（高度/部分/不足）统计 | 展示证据匹配模块的整体表现 |
| 相似度直方图 | 所有requirement×evidence对的相似度值分布 | 展示BGE模型打分的统计特征 |
| 四维堆叠图 | 6组样例的四维加权贡献分解 | 展示评分公式各维度对总分的贡献 |

## 运行方式

```powershell
cd resume_jd_matcher
python scripts/visualize.py
```

运行后图表自动保存到 `outputs/` 目录。

## 依赖

在已有 `requirements.txt` 基础上，额外需要：

```
wordcloud
matplotlib
seaborn
```

## 注意事项

- 需要本地 `models/bge-small-zh-v1.5/` 模型可用
- 中文字体使用 SimHei（Windows 系统自带）
- 未修改任何 `core/` 代码和 `app.py`
