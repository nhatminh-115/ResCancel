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
    "figures/fungibility_v1/depth_generalization.png",
    "figures/fungibility_dense_fraction/dense_fraction_accuracy.png",
    "figures/fungibility_v1/geometry_controls_across_models.png",
    "figures/fungibility_v1/shared_vs_independent_across_models.png",
    "figures/fungibility_v1/learned_vs_random_1d.png",
    "figures/fungibility_dense_fraction/mask_seed_robustness.png",
    "figures/fungibility_dense_fraction/threshold_summary.png",
    "figures/fungibility_v0_9/natural_vs_energy_matched_rank.png",
    "figures/fungibility_v0_9/pc1_amplitude_sweep.png",
    "figures/fungibility_v0_9/rank_expansion_through_blocks.png",
    "figures/fungibility_compression_poc/equivalence_error.png",
    "figures/fungibility_compression_poc/accuracy_vs_tail_tokens.png",
    "figures/fungibility_compression_poc/accuracy_vs_latency.png",
    "figures/fungibility_geometry_bank/delta_vs_pruning.png",
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
    abstract = md[m_abs.end():m_intro.start()].strip()
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

def inject_citations(md: str) -> str:
    for old, new in INLINE_CITATIONS.items():
        md = md.replace(old, new)
    for old, new in CITATIONS.items():
        md = md.replace(old, new)
    return md

def markdown_to_latex(md_path: Path, tex_path: Path):
    cmd = [
        "pandoc",
        str(md_path),
        "--from=markdown+raw_tex+tex_math_dollars",
        "--to=latex",
        "--wrap=preserve",
        "-o",
        str(tex_path),
    ]
    run(cmd)

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
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\usepackage{enumitem}
\setlist{nosep}
\setlength{\emergencystretch}{3em}

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
\appendix
\section*{Supplementary Material}
\addcontentsline{toc}{section}{Supplementary Material}
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

    abstract = inject_citations(normalize_paths(abstract))
    body = inject_citations(normalize_paths(body))
    supp = inject_citations(normalize_paths(supp))

    # Markdown intermediates remain only in build/, never in the arXiv ZIP.
    tmp = BUILD / "tmp"
    tmp.mkdir()
    (tmp / "abstract.md").write_text(abstract, encoding="utf-8")
    (tmp / "main.md").write_text(body, encoding="utf-8")
    (tmp / "supplement.md").write_text(supp, encoding="utf-8")

    markdown_to_latex(tmp / "abstract.md", SRC / "abstract_body.tex")
    markdown_to_latex(tmp / "main.md", SRC / "main_body.tex")
    markdown_to_latex(tmp / "supplement.md", SRC / "supplement_body.tex")

    shutil.copy2(BIB, SRC / "references.bib")
    copy_figures()
    write_main_tex()
    compile_pdf()
    make_zip()

if __name__ == "__main__":
    main()
