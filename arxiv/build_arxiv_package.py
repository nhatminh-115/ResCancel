#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARXIV_DIR = ROOT / "arxiv"
BUILD = ARXIV_DIR / "build"
SRC = BUILD / "source"

MAIN_MD = ROOT / "docs" / "PAPER_DRAFT.md"
SUPP_MD = ROOT / "docs" / "PAPER_SUPPLEMENTARY_DRAFT.md"
BIB = ROOT / "docs" / "PAPER_REFERENCES_DRAFT.bib"

FIGURES = [
    "figures/paper_final/main/fig01_conceptual_overview.png",
    "figures/paper_final/main/fig02_depth_emergence.png",
    "figures/paper_final/main/fig03_dense_fraction.png",
    "figures/paper_final/main/fig04_geometry_constraint.png",
    "figures/paper_final/main/fig05_diversity_constraint.png",
    "figures/paper_final/main/fig06_lowdim_direction.png",
    "figures/paper_final/supp/figS01_mask_robustness.png",
    "figures/paper_final/supp/figS02_retention_thresholds.png",
    "figures/paper_final/supp/figS03_pca_rank.png",
    "figures/paper_final/supp/figS04_pc1_amplitude.png",
    "figures/paper_final/supp/figS05_rank_propagation.png",
    "figures/paper_final/supp/figS06_carrier_equivalence.png",
    "figures/paper_final/supp/figS07_accuracy_vs_tokens.png",
    "figures/paper_final/supp/figS08_accuracy_vs_latency.png",
    "figures/paper_final/supp/figS09_geometry_bank_vs_pruning.png",
]

CITATIONS = {
    "(Dosovitskiy et al., 2021)": r"\citep{dosovitskiy2021vit}",
    "(Touvron et al., 2021)": r"\citep{touvron2021deit}",
    "(Oquab et al., 2023)": r"\citep{oquab2023dinov2}",
    "(Rao et al., 2021)": r"\citep{rao2021dynamicvit}",
    "(Bolya et al., 2023)": r"\citep{bolya2023tome}",
    "(Darcet et al., 2024)": r"\citep{darcet2024registers}",
    "(Marouani et al., 2026)": r"\citep{marouani2026clspatch}",
    "(Parodi et al., 2026)": r"\citep{parodi2026zeroablation}",
    "(Wang et al., 2026)": r"\citep{wang2026informationhorizon}",
    "(Yan et al., 2023)": r"\citep{yan2023ccvit}",
}
INLINE_CITATIONS = {
    "Darcet et al. (2024)": r"\citet{darcet2024registers}",
    "Marouani et al. (2026)": r"\citet{marouani2026clspatch}",
    "Parodi et al. (2026)": r"\citet{parodi2026zeroablation}",
    "Wang et al. (2026)": r"\citet{wang2026informationhorizon}",
    "CCViT (Yan et al., 2023)": r"CCViT \citep{yan2023ccvit}",
    "CAAP; Izadi et al., 2026": r"CAAP; \citealp{izadi2026caap}",
}

TITLE = "Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations"
AUTHOR = "Nhat Minh Nghiem"
AFFILIATION = "University of Economics Ho Chi Minh City (UEH)"
EMAIL = "minhnghiem.31231024429@st.ueh.edu.vn"

def run(cmd, cwd=None, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=check)

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

def strip_internal_preamble(md: str) -> str:
    lines = md.splitlines()
    out = []
    skipping_quote = False
    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("> "):
            continue
        if line.strip() == "":
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"

def split_main(md: str):
    m_abs = re.search(r"^## Abstract\s*$", md, flags=re.M)
    m_intro = re.search(r"^## 1\. Introduction\s*$", md, flags=re.M)
    if not m_abs or not m_intro:
        raise RuntimeError("Could not locate Abstract / Introduction in main manuscript.")
    between = md[m_abs.end():m_intro.start()]
    keyword_match = re.search(r"^\*\*Keywords:\*\*.*$", between, flags=re.M)
    abstract = (between[:keyword_match.start()] if keyword_match else between).strip()
    body = md[m_intro.start():].strip()
    return abstract, body

def prepare_supp(md: str) -> str:
    # Remove the two front headings and internal quote notes.
    md = re.sub(r"^# Supplementary Material\s*$", "", md, count=1, flags=re.M)
    md = re.sub(
        r"^## Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations\s*$",
        "",
        md,
        count=1,
        flags=re.M,
    )
    md = "\n".join(line for line in md.splitlines() if not line.startswith("> "))
    return md.strip() + "\n"

def normalize_paths(md: str) -> str:
    return md.replace("../figures/", "figures/")

def normalize_inline_math(md: str) -> str:
    # The manuscript uses TeX-style inline delimiters \(...\).
    # Pandoc's Markdown reader can treat these as escaped parentheses, so
    # normalize them to dollar-delimited inline math before conversion.
    md = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1) + "$", md, flags=re.S)
    return md

def strip_main_heading_numbers(md: str) -> str:
    out = []
    for line in md.splitlines():
        line = re.sub(r"^(#{2,4})\s+\d+(?:\.\d+)*\.?\s+", r"\1 ", line)
        out.append(line)
    return "\n".join(out)

def merge_figure_captions(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"!\[(?:Supplementary )?Figure ([0-9S]+)\.\s*(.*?)\]\(([^)]+)\)", line)
        if m:
            number, short_caption, path = m.groups()
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                detail = lines[j].strip()
                prefix = f"**Figure {number}. "
                supp_prefix = f"**Supplementary Figure {number}. "
                if detail.startswith(prefix) or detail.startswith(supp_prefix):
                    plain = detail.replace("**", "")
                    plain = re.sub(rf"^(?:Supplementary )?Figure {re.escape(number)}\.\s*", "", plain)
                    out.append(f"![{plain}]({path})")
                    i = j + 1
                    continue
            clean = short_caption.strip()
            out.append(f"![{clean}]({path})")
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)

def inject_citations(md: str) -> str:
    for old, new in INLINE_CITATIONS.items():
        md = md.replace(old, new)
    for old, new in CITATIONS.items():
        md = md.replace(old, new)
    return md

def markdown_to_latex(md_path: Path, tex_path: Path, shift_heading: bool = False):
    cmd = [
        "pandoc",
        str(md_path),
        "--from=markdown+raw_tex+tex_math_dollars",
        "--to=latex",
        "--wrap=preserve",
    ]
    if shift_heading:
        cmd.append("--shift-heading-level-by=-1")
    cmd += [
        "-o",
        str(tex_path),
    ]
    run(cmd)
    tex = tex_path.read_text(encoding="utf-8")
    tex = tex.replace(
        r"\includegraphics{",
        r"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{",
    )
    tex_path.write_text(tex, encoding="utf-8")

def copy_figures():
    for rel in FIGURES:
        src = ROOT / rel
        require(src)
        dst = SRC / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def write_main_tex():
    preamble = r"""
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array}
\usepackage{calc}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\usepackage{enumitem}
\setlist{nosep}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\real}[1]{#1}

\title{""" + TITLE + r"""}
\author{""" + AUTHOR + r"""\\
\small """ + AFFILIATION + r"""\\
\small \texttt{""" + EMAIL.replace("_", r"\_") + r"""}}
\date{}

\begin{document}
\maketitle

\begin{abstract}
\input{abstract_body.tex}
\end{abstract}

\noindent\textbf{Keywords:} Vision Transformer; patch tokens; causal intervention; activation substitution; representation geometry; token diversity

\input{main_body.tex}

\clearpage
\section*{Supplementary Material}
\addcontentsline{toc}{section}{Supplementary Material}
\setcounter{secnumdepth}{0}
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\figurename}{Supplementary Figure}
\input{supplement_body.tex}

\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
"""
    (SRC / "main.tex").write_text(preamble, encoding="utf-8")

def compile_pdf():
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("latexmk not found; source package created without PDF compile.", file=sys.stderr)
        return
    run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=SRC)
    shutil.copy2(SRC / "main.pdf", BUILD / "main.pdf")

def make_zip():
    zip_path = BUILD / "arxiv_submission.zip"
    allowed_ext = {".tex", ".bib", ".png", ".jpg", ".jpeg", ".pdf", ".sty", ".bst"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_ext:
                continue
            # Exclude generated PDF and LaTeX auxiliaries; arXiv should compile from source.
            if p.name == "main.pdf":
                continue
            z.write(p, p.relative_to(SRC))
    print(f"Created {zip_path}")

def main():
    require(MAIN_MD)
    require(SUPP_MD)
    require(BIB)
    if not shutil.which("pandoc"):
        raise RuntimeError("Pandoc is required to generate the arXiv LaTeX source.")

    if BUILD.exists():
        shutil.rmtree(BUILD)
    SRC.mkdir(parents=True)

    main_raw = strip_internal_preamble(MAIN_MD.read_text(encoding="utf-8"))
    abstract, body = split_main(main_raw)
    supp = prepare_supp(SUPP_MD.read_text(encoding="utf-8"))

    abstract = inject_citations(normalize_inline_math(normalize_paths(abstract)))
    body = inject_citations(merge_figure_captions(strip_main_heading_numbers(normalize_inline_math(normalize_paths(body)))))
    supp = inject_citations(merge_figure_captions(normalize_inline_math(normalize_paths(supp))))

    # Markdown intermediates remain only in build/, never in the arXiv ZIP.
    tmp = BUILD / "tmp"
    tmp.mkdir()
    (tmp / "abstract.md").write_text(abstract, encoding="utf-8")
    (tmp / "main.md").write_text(body, encoding="utf-8")
    (tmp / "supplement.md").write_text(supp, encoding="utf-8")

    markdown_to_latex(tmp / "abstract.md", SRC / "abstract_body.tex")
    markdown_to_latex(tmp / "main.md", SRC / "main_body.tex", shift_heading=True)
    markdown_to_latex(tmp / "supplement.md", SRC / "supplement_body.tex", shift_heading=True)

    shutil.copy2(BIB, SRC / "references.bib")
    copy_figures()
    write_main_tex()
    compile_pdf()
    make_zip()

if __name__ == "__main__":
    main()
