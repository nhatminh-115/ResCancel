# arXiv submission package

This directory prepares the frozen manuscript for an arXiv v1 submission.

## Intended metadata

- **Title:** Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations
- **Author:** Nhat Minh Nghiem
- **Affiliation:** University of Economics Ho Chi Minh City (UEH)
- **Primary category:** `cs.CV`
- **Possible cross-list:** `cs.LG`
- **License:** intentionally left undecided; select this manually in the arXiv submission UI.
- **Journal reference:** leave blank for v1 unless the paper has already been accepted/published.
- **DOI:** leave blank for v1.

## Build

From the repository root:

```bash
python arxiv/build_arxiv_package.py
```

The script:

1. reads `docs/PAPER_DRAFT.md` and `docs/PAPER_SUPPLEMENTARY_DRAFT.md`;
2. converts the frozen Markdown manuscript to LaTeX with Pandoc;
3. converts the author-year prose citations to BibTeX citation commands;
4. copies only the figures used by the paper;
5. copies `docs/PAPER_REFERENCES_DRAFT.bib`;
6. builds `arxiv/build/main.pdf` when LaTeX tools are available;
7. creates `arxiv/build/arxiv_submission.zip`.

The ZIP is the source package intended for arXiv. Do not upload the whole repository.

## Before upload

Inspect the generated PDF page by page, especially tables, figure captions, equations, line breaks, references, and the transition into the Supplementary Material. Also verify the arXiv metadata against `metadata.yaml`.

The source package deliberately excludes experiment outputs, internal audit notes, historical reports, comments, and unused figures.
