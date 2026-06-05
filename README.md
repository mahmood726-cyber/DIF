# DIF

> **Naming note:** despite the repository name `DIF`, this project is **not**
> Differential Item Functioning (psychometrics). The code is a computational
> cardiology study. The name is retained only to match the existing repo/remote.

## What it does

`dif_analysis.py` ("Deconvoluting Ischemic Factors with Cross-Model Validation
and Therapeutic Simulation") is a self-contained machine-learning study of
myocardial ischemia built on two cardiac single-cell electrophysiology models:

- **Luo-Rudy I (1991)** — `luo_rudy_1991_model`
- **Ten Tusscher et al. (2006)** epicardial — `ten_tusscher_2006_model`

Each model is a system of ODEs (membrane voltage + gating/ion-concentration
states) integrated with `scipy.integrate.solve_ivp` (BDF). The models are
parameterised by three ischemic factors:

- `Ko` — extracellular potassium (hyperkalemia),
- `pH_i` — intracellular pH (acidosis),
- `G_K_ATP` — ATP-sensitive K+ channel conductance.

The workflow (`main()`):

1. **Primary analysis** — simulate many Luo-Rudy cells across Healthy /
   Moderate / Severe ischemia classes, extract action-potential features
   (resting potential, peak voltage, amplitude, max upstroke velocity,
   APD50/APD90), and train two `RandomForestRegressor`s to predict `Ko` and
   `pH_i` from those features.
2. **Cross-model validation** — generate cells from the *Ten Tusscher* model and
   evaluate the Luo-Rudy-trained regressors on them.
3. **Therapeutic simulation** — pick a severely ischemic cell, simulate a
   pharmacological I_K(ATP) blocker (`G_K_ATP = 0`), and quantify the predicted
   change in `Ko` / `pH_i` before vs. after "treatment". A summary figure is
   saved to `pharmacological_rescue.png`.

The script generates all of its own data via simulation — **no external data
file is required.**

## Layout

| File | Purpose |
| --- | --- |
| `dif_analysis.py` | Models, simulation, feature extraction, ML workflow (`main()`). |
| `test_dif.py` | Pytest suite validating the importable core. |
| `requirements.txt` | Pinned runtime + test dependencies. |

Importing the module (`import dif_analysis`) is side-effect-free; the heavy
workflow only runs under `if __name__ == '__main__'`. Matplotlib uses the
headless `Agg` backend, so the script runs without a display.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.13 (tested) with numpy, scipy, pandas, scikit-learn,
matplotlib.

## Run

```bash
python dif_analysis.py
```

This prints progress for each stage and writes `pharmacological_rescue.png`.
Full execution simulates ~200 cardiac cells and takes a few minutes.

## Test

```bash
python -m pytest -q
```

The suite covers:

- import side-effect safety and the headless matplotlib backend;
- ODE simulation shape/finiteness for both models;
- feature-extraction invariants (amplitude consistency, APD50 ≤ APD90,
  rejection of flat non-AP traces, no divide-by-zero);
- a **planted-signal** check — a severely ischemic cell must be distinguishable
  from a healthy one, and the trained regressors must recover the correct
  Ko/pH ordering (severe → higher Ko, lower pH) while the healthy/clean case
  stays at the planted baseline (Ko ≈ 5.4, pH ≈ 7.4);
- determinism / symmetry of two identical healthy cells.
