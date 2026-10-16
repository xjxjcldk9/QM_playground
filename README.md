# QM Playground

Numerical and visual solutions to quantum mechanics exercises, primarily from Griffiths' *Introduction to Quantum Mechanics*.

## Layout

```
QM_playground/
├── code/                 # Python scripts, one per problem (probX.Y.py)
├── latex/                # LaTeX write-up (originally from Overleaf)
│   ├── main.tex          # root document
│   ├── chapters/         # ch01_wave_function.tex … ch10_scattering.tex
│   ├── fig/              # figures included by the write-up
│   ├── .latexmkrc        # build config (output goes to latex/build/)
│   └── .chktexrc         # LaTeX linter config
├── .vscode/              # LaTeX Workshop + LTeX+ settings
└── pyproject.toml        # Python deps (uv)
```

## Code

Each script in `code/` corresponds to a problem number:

| File | Problem | Description |
|------|---------|-------------|
| `code/prob1.13.py` | 1.13 | Probability density — histogram of random samples vs. the arcsine distribution |
| `code/prob2.55.py` | 2.55 | ODE solver for `y'' = (t² − K)·y` with an interactive slider for K |
| `code/prob2.56.py` | 2.56 | Same ODE, three panels side-by-side exploring different eigenvalue candidates |
| `code/prob2.57.py` | 2.57 | Harmonic oscillator `y'' = −K·y`, four energy levels with sliders |
| `code/prob2.59.py` | 2.59 | Airy function Ai(x): plot of ψ₁ and ψ₁₀, integrals from the 1st and 10th zeros |
| `code/prob2.59_calc.py` | 2.59 | Expectation values ⟨x⟩, ⟨x²⟩, σ_x, σ_p, and σ_x·σ_p for ψ₁ and ψ₁₀ |

## Requirements

- Python ≥ 3.12
- `scipy`, `matplotlib`

Install and run with [uv](https://github.com/astral-sh/uv):

```bash
cd code && uv run python prob2.59.py
```

Scripts save figures to the current directory (`code/*.png` is git-ignored);
copy the ones you want in the write-up into `latex/fig/`.

## LaTeX

Requires a TeX distribution (MacTeX) providing `latexmk`, `chktex` and `latexindent`.

```bash
cd latex && latexmk          # build -> latex/build/main.pdf
cd latex && latexmk -c       # clean aux files
cd latex && chktex -q -l .chktexrc main.tex   # lint from the terminal
```

In VS Code (open the repo root), install the recommended extensions:

- **LaTeX Workshop** — builds on save, PDF preview, SyncTeX, ChkTeX linting in the Problems panel, `latexindent` formatting.
- **LTeX+** — grammar and spell checking (LanguageTool) for `.tex` and `.md`. Add words to the dictionary via the quick-fix, or in `ltex.dictionary` in `.vscode/settings.json`.
