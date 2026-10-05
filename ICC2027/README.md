# ICC 2027 paper source

LaTeX source and compiled PDF for "An Auditable LLM Gating Agent for
Poisoning-Resilient Federated IoT Security" (IEEE ICC 2027 submission).
The experiment code and results this paper reports on live in the
parent directory (`../fedgate`, `../scripts`, `../results`) -- see
`../README.md` for the full experimental log.

To compile:

```bash
cd ICC2027
latexmk -pdf -interaction=nonstopmode main.tex
```

Requires a standard TeX Live install (`pdflatex`, `bibtex`, `latexmk`).
