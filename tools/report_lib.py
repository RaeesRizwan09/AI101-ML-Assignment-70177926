#!/usr/bin/env python3
"""Shared LaTeX report plumbing.

Every report is generated from the task's results/*.json, so no number is typed by
hand and the PDF can never disagree with the notebook.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

STUDENT = {
    "name": "Rizwan Ali",
    "roll": "70177926",
    # the section is read from the environment so that one PDF can be built per section
    # (AI101_SECTION=B python B1_report.py) without touching the sources
    "section": os.environ.get("AI101_SECTION", "C").strip(),
    "github": "RaeesRizwan09",
}
REPO = f"https://github.com/{STUDENT['github']}/AI101-ML-Assignment-{STUDENT['roll']}"

PREAMBLE = r"""
\documentclass[10pt,conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\setlength{\textfloatsep}{4pt plus 1pt minus 1pt}
\setlength{\floatsep}{4pt plus 1pt minus 1pt}
\setlength{\intextsep}{5pt plus 1pt minus 1pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{0pt}
\renewcommand{\arraystretch}{0.94}
\setlength{\parskip}{0pt}
\hyphenpenalty=1000
\sloppy
"""


def header(task_id, title, abstract_one_liner):
    return rf"""
\title{{\vspace{{-1.2em}}\normalsize THE UNIVERSITY OF LAHORE $\bullet$ DEPARTMENT OF INTELLIGENT SYSTEMS $\bullet$ AI-101\\[2pt]
\Large Task {task_id}: {title}}}
\author{{\normalsize {STUDENT['name']} \quad|\quad Roll No.\ {STUDENT['roll']} \quad|\quad Section {STUDENT['section']}\\
\normalsize\url{{{REPO}}} \quad|\quad Instructor: Dr.\ Muhammad Awais, AI Architect}}
\maketitle
\vspace{{-1.4em}}
\begin{{center}}\small\itshape {abstract_one_liner}\end{{center}}
\vspace{{-0.6em}}
"""


def fig(path, caption, label, width=r"\columnwidth", wide=False):
    env = "figure*" if wide else "figure"
    return rf"""
\begin{{{env}}}[t]
\centering
\includegraphics[width={width}]{{{path}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{{env}}}
"""


def table(spec, header_row, body, caption, label, size=r"\footnotesize", wide=False):
    """header_row is given without the trailing LaTeX line break; body rows must end in '\\\\'."""
    if not header_row.rstrip().endswith(r"\\"):
        header_row = header_row.rstrip() + r" \\"
    env = "table*" if wide else "table"
    return rf"""
\begin{{{env}}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
{size}
\begin{{tabular}}{{{spec}}}
\toprule
{header_row}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{{env}}}
"""


def esc(s):
    """Escape text that came from data (titles, n-grams) for LaTeX."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def bold_if(cond, s):
    return rf"\textbf{{{s}}}" if cond else s


def compile_tex(tex_path: Path, tinytex_bin: Path) -> Path:
    """Compile twice (refs), return the pdf path."""
    env_path = f"{tinytex_bin}:/usr/local/bin:/usr/bin:/bin"
    import os
    env = dict(os.environ, PATH=env_path, TEXMFVAR=str(tex_path.parent / ".texmfvar"))
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        log = (tex_path.parent / (tex_path.stem + ".log")).read_text(errors="ignore")
        tail = "\n".join(log.splitlines()[-40:])
        raise RuntimeError(f"pdflatex failed:\n{tail}")
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.exists():
        raise RuntimeError("no pdf produced")
    return pdf


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1))


def check_and_report(pdf: Path, task_id: str, max_content_pages: int = 3):
    n = page_count(pdf)
    size_kb = pdf.stat().st_size / 1024
    print(f"  {pdf.name}: {n} pages, {size_kb:.0f} kB")
    if n > max_content_pages:
        print(f"  !! {n} pages exceeds the limit (2 content + 1 for disclosure/refs)")
    return n
