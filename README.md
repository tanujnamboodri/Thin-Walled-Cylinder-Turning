# Physics-Informed Machine Learning for Chatter Detection in Thin-Walled Cylinder Turning

Cutting-force data and analysis code for chatter detection during turning of thin-walled 1.4301 (X5CrNi18-10) Austenitic Stainless Steel Cylinders

**Reference publication**

> To be updated

---

## What this does

Chatter is detected from cutting-force signals using five features extracted
per axial segment:

| # | Feature | Type |
|---|---------|------|
| 1 | RMS of the resultant cutting force | Force |
| 2 | Excess kurtosis of the resultant force | Force |
| 3 | Dominant non-harmonic frequency | Spectral |
| 4 | Normalised axial segment position | Geometry |
| 5 | Wall thickness | Geometry |

Features 4 and 5 are the geometry-informed inputs. 
Four classifiers are evaluated: Neural Network, Random Forest, Support Vector Machine (RBF kernel), and Logistic Regression.
*Note - NN uses four features.
---

## Validation schemes

All three are computed in a single run.

| Scheme | Training set | Test set | What it measures |
|---|---|---|---|
| **LOPO** | All passes except pass *k* | Pass *k* | Grouped generalisation (interpolation) |
| **FORWARD** | Passes 1 to *k*−1 | Pass *k* | Prospective deployment (extrapolation) |
| **FIXED** | Passes 1–10 | Passes 11–15 | Extrapolation beyond the trained thickness range |

Leave-One-Pass-Out holds out all nine segments of one complete machining pass per fold. Segments within a pass share the same wall thickness, cutting parameters, and process state, so mixing them across training and test sets would introduce temporal leakage.

---

## Feature ablation

Eight feature configurations, selected with `--mode`:

| Mode | Feature set |
|---|---|
| `A` | All five features (baseline) |
| `B` | Wall thickness replaced by pass number |
| `C` | Wall thickness removed |
| `D` | Force and spectral features only |
| `E` | RMS only |
| `F` | Force and spectral features only (alias of D) |
| `G` | Geometry only (segment position, wall thickness) |
| `H` | RMS removed |

Mode `H` tests whether classification depends on the RMS feature, which also participates in the chatter labelling rule. Mode `G` tests whether the two geometry features are sufficient on their own.

---

## Installation

```bash
git clone https://github.com/tanujnamboodri/Thin-Walled-Cylinder-Turning.git
cd Thin-Walled-Cylinder-Turning
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10 to 3.13. Python 3.14 currently has scipy and scikit-learn compatibility issues; 3.12 is recommended.

---

## Usage

**Interactive file selection**

```bash
python3 chatter_detection.py
```

**Batch mode with default data folder**

```bash
python3 chatter_detection.py --mode A --no-gui
```

**Explicit file paths**

```bash
python3 chatter_detection.py metadata.xlsx Pass1.xlsx Pass2.xlsx ...
```

**Full ablation study (all eight modes)**

```bash
bash run_ablation.sh --no-gui
```
---

## Input data format

**Pass files** — one per machining pass, `.xlsx` or `.csv`:

| Time | X | Y | Z |
|---|---|---|---|
| 0.001 | 83.30 | 62.93 | 24.05 |
| 0.002 | 94.61 | 65.78 | 29.15 |

Time in milliseconds, forces in newtons, sampled at 1000 Hz. Column names `Fx`, `Fy`, `Fz` are also accepted.

**Metadata file** — one row per pass, `.xlsx` or `.csv`

| Pass | Vc | ap | f | Outer Dia | Inner Dia | t |
|---|---|---|---|---|---|---|
| 1 | 280 | 0.2 | 0.16 | 37.6 | 30 | 3.8 |
| 2 | 280 | 0.2 | 0.16 | 37.2 | 30 | 3.6 |

Diameters and wall thickness in millimetres. Wall thickness `t` is derived as (OD − ID)/2 if not supplied.

---

## Output

Written to `results/ablation_<MODE>/`:

| File | Contents |
|---|---|
| `Fig1_RMS_Force_Map.pdf` | RMS force map with chatter labels |
| `Fig2_Predicted_Maps_{LOPO,FORWARD,FIXED}.pdf` | Actual vs predicted, all four models |
| `Fig3_Confusion_Matrices_{LOPO,FORWARD,FIXED}.pdf` | Confusion matrices, 2×2 |
| `Fig4_Safety_Analysis_{LOPO,FORWARD,FIXED}.pdf` | Recall bars and precision-recall |
| `Fig5_Diagnostics.pdf` | All four models, combined |
| `Fig12–Fig15_*_Diagnostics.pdf` | Per-model training curve and feature importance |
| `results/ablation_results.csv` | All metrics, all modes, all schemes |

The CSV deduplicates on `(Mode, Validation, Model)`, so re-running a mode does not create duplicate rows. Delete the file to force a full rewrite.

---

## Chatter labelling

A segment is labelled as chatter only when **both** criteria are met:

1. **Non-harmonic peak ratio ≥ 3.5** — the largest non-harmonic spectral peak
  divided by the median of the non-harmonic spectrum, after masking ±5 Hz
   around the first five spindle-frequency harmonics.
2. **RMS ratio ≥ 2.5** — segment RMS divided by the Pass 1 RMS baseline for the
   same segment position.

The mean is removed from each segment before the FFT to eliminate the DC component. A Hann window is applied, and the signal is zero-padded to the next power of two.

Labels were verified against visual inspection of chatter marks on the machined surface. The thresholds are empirical values calibrated for this setup.

---

## Reproducibility notes

- All random seeds are fixed. The per-fold seed is `SEED + pass_number`.
- No hyperparameter tuning or grid search was performed. Values are either scikit-learn defaults or fixed a priori.
- Minority-class augmentation is applied only to the training portion of each fold, after the fold split. No synthetic data enters any test set.
- Normalisation parameters are computed from the training fold only and applied unchanged to the test fold.

---

## Limitations

This dataset is from a **single workpiece** machined through 15 sequential passes under one set of cutting conditions. Wall thickness decreases monotonically with pass number and is therefore confounded with cumulative tool wear, thermal evolution, and residual deformation. The data establish an association between thinner walls and chatter; they do not isolate wall thickness as the sole causal factor.

The 135 segments are nine per pass across 15 passes. They are hierarchically structured, not independent. The effective independent sample size is 15 passes.

---

## Citation

To be updated

---

## Acknowledgements

Experimental work carried out at the University of Miskolc, Hungary, with machining resources provided by Interstop Kft., Miskolc, Hungary.

Project no. 2020–1.2.3-EUREKA-2022–00025 was implemented with support from the Ministry of Culture and Innovation of Hungary, from the National Research, Development and Innovation Fund, under the 2020–1.2.3-EUREKA funding scheme.

