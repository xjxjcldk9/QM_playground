# QM Playground

Numerical and visual solutions to quantum mechanics exercises, primarily from Griffiths' *Introduction to Quantum Mechanics*.

## Structure

Each file corresponds to a problem number:

| File | Problem | Description |
|------|---------|-------------|
| `prob1.13.py` | 1.13 | Probability density — histogram of random samples vs. the arcsine distribution |
| `prob2.55.py` | 2.55 | ODE solver for `y'' = (t² − K)·y` with an interactive slider for K |
| `prob2.56.py` | 2.56 | Same ODE, three panels side-by-side exploring different eigenvalue candidates |
| `prob2.57.py` | 2.57 | Harmonic oscillator `y'' = −K·y`, four energy levels with sliders |
| `prob2.59.py` | 2.59 | Airy function Ai(x): plot of ψ₁ and ψ₁₀, integrals from the 1st and 10th zeros |
| `prob2.59_calc.py` | 2.59 | Expectation values ⟨x⟩, ⟨x²⟩, σ_x, σ_p, and σ_x·σ_p for ψ₁ and ψ₁₀ |

## Requirements

- Python ≥ 3.12
- `scipy`, `matplotlib`

Install and run with [uv](https://github.com/astral-sh/uv):

```bash
uv run python prob2.59.py
```
