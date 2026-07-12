"""Convert the course-design Markdown report into a standalone Overleaf file."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "design_report.md"
TARGET = ROOT / "docs" / "overleaf" / "main.tex"


PREAMBLE = r"""\documentclass[UTF8,zihao=-4,a4paper,oneside]{ctexrep}

% Overleaf 编译器请选择 XeLaTeX。
\usepackage[a4paper,left=2.8cm,right=2.6cm,top=2.6cm,bottom=2.6cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs,tabularx,array,longtable,multirow}
\usepackage{graphicx,float,caption,subcaption}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{microtype}

\definecolor{linkblue}{RGB}{32,86,148}
\definecolor{codegray}{RGB}{246,248,250}
\definecolor{framegray}{RGB}{160,168,176}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=linkblue,urlcolor=linkblue}
\urlstyle{same}
\setstretch{1.45}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\renewcommand{\thefigure}{\thechapter-\arabic{figure}}
\renewcommand{\thetable}{\thechapter-\arabic{table}}
\renewcommand{\theequation}{\thechapter-\arabic{equation}}
\renewcommand{\arraystretch}{1.35}
\captionsetup{font=small,labelsep=quad}
\titleformat{\chapter}{\centering\heiti\zihao{3}}{第\chinese{chapter}章}{1em}{}
\titleformat{\section}{\heiti\zihao{4}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\heiti\zihao{-4}}{\thesubsection}{0.8em}{}
\titlespacing*{\chapter}{0pt}{-12pt}{24pt}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\small 基于预训练语言模型的简历岗位匹配度评估与面试辅助系统}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\lstset{
  basicstyle=\ttfamily\small,
  backgroundcolor=\color{codegray},
  frame=single,
  rulecolor=\color{framegray},
  breaklines=true,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  xleftmargin=1em,
  xrightmargin=1em
}

% 封面信息：只需修改下面五项。
\newcommand{\coursename}{【待填写】}
\newcommand{\collegeclass}{【待填写】}
\newcommand{\studentinfo}{【待填写】}
\newcommand{\advisorname}{【待填写】}
\newcommand{\finishdate}{【待填写】}

% 图片尚未生成时保持可编译；上传图片后按注释替换对应占位框。
\newcommand{\placeholderbox}[1]{%
  \fbox{\begin{minipage}[c][5.8cm][c]{0.88\textwidth}
    \centering\color{framegray}\zihao{-4}#1
  \end{minipage}}%
}

\begin{document}

\begin{titlepage}
  \centering
  \vspace*{1.6cm}
  {\heiti\zihao{1} 课程设计报告\par}
  \vspace{2.2cm}
  {\heiti\zihao{2} 基于预训练语言模型的\par}
  \vspace{0.35cm}
  {\heiti\zihao{2} 简历岗位匹配度评估与面试辅助系统\par}
  \vfill
  \renewcommand{\arraystretch}{1.7}
  \begin{tabular}{>{\heiti}r p{8.5cm}}
    课程名称： & \coursename \\
    学院、专业、班级： & \collegeclass \\
    学号、姓名： & \studentinfo \\
    指导教师： & \advisorname \\
    完成日期： & \finishdate \\
  \end{tabular}
  \vfill
\end{titlepage}

\pagenumbering{Roman}
"""


POSTAMBLE = "\n\\end{document}\n"


def escape_text(value: str) -> str:
    """Escape LaTeX syntax while preserving Markdown inline constructs."""
    tokens: list[str] = []

    def hold(rendered: str) -> str:
        tokens.append(rendered)
        return f"@@TOKEN{len(tokens) - 1}@@"

    def render_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if label == url:
            return hold(r"\url{" + url + "}")
        return hold(r"\href{" + url + "}{" + escape_plain(label) + "}")

    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", render_link, value)
    value = re.sub(
        r"`([^`]+)`",
        lambda m: hold(r"\texttt{\detokenize{" + m.group(1) + "}}"),
        value,
    )
    value = re.sub(
        r"\\\((.+?)\\\)",
        lambda m: hold(r"\(" + m.group(1) + r"\)"),
        value,
    )
    value = re.sub(r"\$([^$]+)\$", lambda m: hold("$" + m.group(1) + "$"), value)
    value = re.sub(r"\*\*([^*]+)\*\*", lambda m: hold(r"\textbf{" + escape_plain(m.group(1)) + "}"), value)
    value = re.sub(r"\*([^*]+)\*", lambda m: hold(r"\emph{" + escape_plain(m.group(1)) + "}"), value)
    value = escape_plain(value)
    for index, rendered in enumerate(tokens):
        value = value.replace(f"@@TOKEN{index}@@", rendered)
    return value


def escape_plain(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def clean_heading(title: str) -> str:
    title = re.sub(r"^第[一二三四五六七八九十]+章\s*", "", title)
    title = re.sub(r"^\d+(?:\.\d+)*\s*", "", title)
    return escape_text(title.strip())


def make_figure(raw: str, note: str, index: int) -> list[str]:
    text = raw.strip().lstrip("【")
    if "】" in text:
        text = text.split("】", 1)[0]
    text = re.sub(r"占位符[：:]?", "：", text)
    text = re.sub(r"^图\d+-\d+[：:]?", "", text)
    text = re.sub(r"^系统", "系统", text)
    text = re.sub(r"[。.]?插入.*$", "", text).strip("。：: ")
    label = f"fig:placeholder-{index}"
    note_line = escape_text(note) if note else "请根据正文说明生成并插入对应图片。"
    return [
        r"\begin{figure}[H]",
        r"  \centering",
        f"  % 上传图片后，可替换下一行为：\\includegraphics[width=0.88\\textwidth]{{figures/figure-{index:02d}.png}}",
        f"  \\placeholderbox{{{escape_text(text)}\\\\[0.6em]\\footnotesize {note_line}}}",
        f"  \\caption{{{escape_text(text)}}}",
        f"  \\label{{{label}}}",
        r"\end{figure}",
        "",
    ]


def table_to_latex(rows: list[list[str]], caption: str | None) -> list[str]:
    columns = len(rows[0])
    spec = "".join([r">{\raggedright\arraybackslash}X" for _ in range(columns)])
    output = [r"\begin{table}[H]", r"  \centering"]
    if caption:
        normalized = re.sub(r"^表\d+(?:-\d+)?\s*", "", caption).strip()
        output.append(f"  \\caption{{{escape_text(normalized)}}}")
    output.extend([r"  \small", f"  \\begin{{tabularx}}{{\\textwidth}}{{{spec}}}", r"    \toprule"])
    for row_index, row in enumerate(rows):
        output.append("    " + " & ".join(escape_text(cell.strip()) for cell in row) + r" \\")
        if row_index == 0:
            output.append(r"    \midrule")
    output.extend([r"    \bottomrule", r"  \end{tabularx}", r"\end{table}", ""])
    return output


def convert(lines: list[str]) -> str:
    output = [PREAMBLE.rstrip(), ""]
    index = 0
    figure_index = 0
    pending_caption: str | None = None
    in_code = False
    in_math = False
    list_kind: str | None = None
    appendix = False
    toc_inserted = False

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.extend([f"\\end{{{list_kind}}}", ""])
            list_kind = None

    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()

        if index <= 10:
            index += 1
            continue

        if stripped.startswith("```"):
            close_list()
            if not in_code:
                language = stripped[3:].strip()
                output.append(r"\begin{lstlisting}" + (f"[language={language}]" if language in {"Python"} else ""))
                in_code = True
            else:
                output.extend([r"\end{lstlisting}", ""])
                in_code = False
            index += 1
            continue
        if in_code:
            output.append(raw)
            index += 1
            continue

        if stripped == r"\[":
            close_list()
            output.append(r"\begin{equation}")
            in_math = True
            index += 1
            continue
        if stripped == r"\]":
            output.extend([r"\end{equation}", ""])
            in_math = False
            index += 1
            continue
        if in_math:
            output.append(raw)
            index += 1
            continue

        if stripped.startswith("> 【") and "占位符" in stripped:
            close_list()
            note = ""
            if index + 1 < len(lines) and lines[index + 1].strip().startswith(">"):
                note = lines[index + 1].strip()[1:].strip()
                index += 1
            figure_index += 1
            output.extend(make_figure(stripped[1:].strip(), note, figure_index))
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            data_rows = []
            for table_line in table_lines:
                cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                data_rows.append(cells)
            output.extend(table_to_latex(data_rows, pending_caption))
            pending_caption = None
            continue

        caption_match = re.fullmatch(r"\*\*(表\d+(?:-\d+)?\s+.+)\*\*", stripped)
        if caption_match:
            pending_caption = caption_match.group(1)
            index += 1
            continue

        ordered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        desired_list = "enumerate" if ordered else "itemize" if unordered else None
        if desired_list:
            if list_kind != desired_list:
                close_list()
                list_kind = desired_list
                output.append(f"\\begin{{{list_kind}}}[leftmargin=3.5em,itemsep=0.25em]")
            item_text = ordered.group(2) if ordered else unordered.group(1)
            output.append(r"  \item " + escape_text(item_text))
            index += 1
            continue
        close_list()

        if stripped.startswith("### "):
            output.extend([f"\\subsection{{{clean_heading(stripped[4:])}}}", ""])
        elif stripped.startswith("## "):
            heading = stripped[3:].strip()
            if appendix and heading.startswith("附录"):
                heading = re.sub(r"^附录[A-Z]\s*", "", heading)
                output.extend([f"\\chapter{{{clean_heading(heading)}}}", ""])
            else:
                output.extend([f"\\section{{{clean_heading(heading)}}}", ""])
        elif stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading == "中文摘要":
                output.extend([r"\chapter*{中文摘要}", r"\addcontentsline{toc}{chapter}{中文摘要}", ""])
            elif heading == "Abstract":
                output.extend([r"\chapter*{Abstract}", r"\addcontentsline{toc}{chapter}{Abstract}", ""])
            elif heading == "参考文献":
                output.extend([r"\chapter*{参考文献}", r"\addcontentsline{toc}{chapter}{参考文献}", ""])
            elif heading == "附录":
                appendix = True
                output.extend(
                    [
                        r"\appendix",
                        r"\titleformat{\chapter}{\centering\heiti\zihao{3}}{附录\thechapter}{1em}{}",
                        "",
                    ]
                )
            else:
                if not toc_inserted:
                    output.extend(
                        [
                            r"\clearpage",
                            r"\tableofcontents",
                            r"\clearpage",
                            r"\pagenumbering{arabic}",
                            "",
                        ]
                    )
                    toc_inserted = True
                output.extend([f"\\chapter{{{clean_heading(heading)}}}", ""])
        elif stripped.startswith(">"):
            output.extend([r"\begin{quote}", escape_text(stripped[1:].strip()), r"\end{quote}", ""])
        elif stripped:
            if re.match(r"^\[\d+\]", stripped):
                output.append(r"\noindent " + escape_text(stripped) + r"\par")
            else:
                output.append(escape_text(stripped) + "\n")
        else:
            output.append("")
        index += 1

    close_list()
    output.append(POSTAMBLE.strip())
    return "\n".join(output) + "\n"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE.read_text(encoding="utf-8")
    TARGET.write_text(convert(source_text.splitlines()), encoding="utf-8")
    print(f"Generated {TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
