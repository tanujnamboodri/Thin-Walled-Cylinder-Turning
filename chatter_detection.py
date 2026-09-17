#!/usr/bin/env python3
# =============================================================================
#  chatter_detection.py
#
#  Physics-Informed Machine Learning for Chatter Detection in
#  Thin-Walled Cylinder Turning of 1.4301 (X5CrNi18-10) Stainless Steel
#
#  Reference publication:
#    Namboodri, T.; Felho, C.; Sztankovics, I.
#    "Physics-Informed Machine Learning for Chatter Detection in
#     Thin-Walled Cylinder Turning of 1.4301 Steel"
#    Journal of Manufacturing and Materials Processing (JMMP), MDPI.
#
#  Repository:
#    https://github.com/tanujnamboodri/Thin-Walled-Cylinder-Turning
#
#  -------------------------------------------------------------------------
#  WHAT THIS SCRIPT DOES
#  -------------------------------------------------------------------------
#  Detects machining chatter from cutting-force signals recorded during
#  thin-walled cylinder turning. Five features are extracted per segment:
#    1. RMS of the resultant cutting force
#    2. Excess kurtosis of the resultant cutting force
#    3. Dominant non-harmonic frequency (after spindle-harmonic masking)
#    4. Normalised axial segment position  (geometry feature)
#    5. Wall thickness                     (geometry feature)
#
#  Four classifiers are evaluated: Neural Network, Random Forest,
#  Support Vector Machine (RBF), and Logistic Regression.
#
#  Three validation schemes are computed in a single run:
#    LOPO    - Leave-One-Pass-Out; hold out one complete pass per fold
#    FORWARD - forward chaining; predict pass k from passes 1..k-1 only
#    FIXED   - fixed early/late split; train on passes 1-10, test on 11-15
#
#  A feature ablation study (8 configurations) can be run via --mode.
#
#  -------------------------------------------------------------------------
#  USAGE
#  -------------------------------------------------------------------------
#    python3 chatter_detection.py                    # file dialog, mode A
#    python3 chatter_detection.py --mode H --no-gui  # batch, no dialog
#    python3 chatter_detection.py meta.xlsx P1.xlsx P2.xlsx ...
#
#  Ablation modes:
#    A  baseline            [RMS, Kurtosis, DomFreq, SegPos, WallThickness]
#    B  swap wall -> pass   [RMS, Kurtosis, DomFreq, SegPos, PassNumber]
#    C  drop wall thickness [RMS, Kurtosis, DomFreq, SegPos]
#    D  force + spectral    [RMS, Kurtosis, DomFreq]
#    E  RMS only            [RMS]
#    F  force + spectral    [RMS, Kurtosis, DomFreq]   (alias of D)
#    G  geometry only       [SegPos, WallThickness]
#    H  no RMS              [Kurtosis, DomFreq, SegPos, WallThickness]
#
#  -------------------------------------------------------------------------
#  INPUT DATA FORMAT
#  -------------------------------------------------------------------------
#  Pass files (one per machining pass), .xlsx or .csv, columns:
#    Time (ms) | X / Fx (N) | Y / Fy (N) | Z / Fz (N)
#  sampled at 1000 Hz.
#
#  Metadata file, .xlsx or .csv, one row per pass, columns:
#    Pass | Vc | ap | f | Outer Dia | Inner Dia | t
#
#  -------------------------------------------------------------------------
#  REQUIREMENTS
#  -------------------------------------------------------------------------
#    Python >= 3.10, < 3.14   (3.14 has scipy/scikit-learn compatibility issues)
#    numpy, pandas, matplotlib, scipy, scikit-learn, torch, openpyxl
#
#  See requirements.txt.
#
#  -------------------------------------------------------------------------
#  LICENCE
#  -------------------------------------------------------------------------
#  Released for academic use. If you use this code, please cite the
#  reference publication above.
# =============================================================================

import os, re, sys, glob, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import windows

import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

EPS = np.finfo(float).eps

# =============================================================================
#  FONT  (Palatino Linotype on Windows / Palatino on macOS — fallback to serif)
# =============================================================================
def _set_font(family="Palatino Linotype", size=16):
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    # Description asks: try 'Palatino Linotype' first, then 'Palatino', then 'Georgia'.
    for candidate in [family, "Palatino", "Georgia", "TeX Gyre Pagella", "DejaVu Serif"]:
        if candidate in available:
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"]  = [candidate]
            print(f"  Font set to: {candidate}")
            break
    else:
        matplotlib.rcParams["font.family"] = "serif"
        print("  Font: Palatino not found — using default serif")
    # Axis / label / tick / legend text = size 16 (per description).
    matplotlib.rcParams["font.size"]        = size
    matplotlib.rcParams["axes.titlesize"]   = size
    matplotlib.rcParams["axes.labelsize"]   = size
    matplotlib.rcParams["xtick.labelsize"]  = size
    matplotlib.rcParams["ytick.labelsize"]  = size
    matplotlib.rcParams["legend.fontsize"]  = size
    # Grid + white background (per description global style).
    matplotlib.rcParams["axes.grid"]        = True
    matplotlib.rcParams["grid.color"]       = "lightgray"
    matplotlib.rcParams["grid.linestyle"]   = "-"
    matplotlib.rcParams["grid.linewidth"]   = 0.5
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"]   = "white"
    matplotlib.rcParams["savefig.dpi"]      = 300

# Font size used for text drawn INSIDE the axes (annotations, "C" marks,
# confusion-matrix cell text). Description asks for 14 here (vs 16 for axes).
INSIDE_FONT = 14

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def read_any(path):
    path = str(path).strip()
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path)
        except ImportError:
            raise ImportError("openpyxl required: pip install openpyxl") from None
        except Exception as exc:
            raise RuntimeError(
                f"Could not read: {path}\n{exc}\n"
                "Is the file open in Excel?") from exc
    for delim in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=delim)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)


def nextpow2(x):
    return 1 if x < 1 else 2 ** int(np.ceil(np.log2(x)))


def standardise_meta_columns(meta):
    rename = {}
    for c in meta.columns:
        u = str(c).upper().strip()
        if   u.startswith("VC"):                          rename[c] = "Vc"
        elif u.startswith("OUTER") or u == "OD":         rename[c] = "OD"
        elif u.startswith("INNER") or u == "ID":         rename[c] = "ID"
        elif u.startswith("AP") or u.startswith("DEPTH"): rename[c] = "ap"
        elif u == "F" or u.startswith("F_") or u.startswith("FEED"): rename[c] = "f"
    meta = meta.rename(columns=rename)
    for col in ["Vc", "ap", "f", "OD", "ID"]:
        if col not in meta.columns:
            raise ValueError(f'Metadata column "{col}" not found. '
                             f'Present: {list(meta.columns)}')
    return meta


def load_dataset(pass_files, meta_file, num_passes, num_segments, fs,
                 peak_ratio, rms_ratio, n_harm, bw, pass_numbers, tag="A"):
    print(f"  [{tag}] Loading metadata...")
    meta = read_any(meta_file)
    meta.columns = [str(c).strip() for c in meta.columns]
    meta = standardise_meta_columns(meta)
    if len(meta) < max(pass_numbers):
        raise ValueError(
            f"Metadata has {len(meta)} rows but highest pass number is "
            f"{max(pass_numbers)}.")
    meta["thickness"] = (meta["OD"] - meta["ID"]) / 2
    meta["N_rpm"]     = meta["Vc"] * 1000 / (np.pi * meta["OD"])
    meta["f_spindle"] = meta["N_rpm"] / 60

    pass_data = []
    for p in range(num_passes):
        fp  = pass_files[p]; pn = pass_numbers[p]
        print(f"  [{tag}] Pass {pn:2d} | \"{os.path.basename(fp)}\" ", end="")
        T = read_any(fp)
        if T.shape[1] < 4:
            raise ValueError(f"Pass {pn}: need Time, Fx, Fy, Fz (4 cols).")
        T = T.iloc[:, :4].copy()
        T.columns = ["Time", "Fx", "Fy", "Fz"]
        T = T.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        pass_data.append(T)
        print(f"| {len(T)} samples  (OD={meta['OD'].iloc[pn-1]:.1f} mm, "
              f"thickness={meta['thickness'].iloc[pn-1]:.2f} mm)")

    seg_data  = [[None]*num_segments for _ in range(num_passes)]
    fft_store = [[None]*num_segments for _ in range(num_passes)]
    chatter_index = np.zeros((num_passes, num_segments))
    rms_store     = np.zeros((num_passes, num_segments))

    for p in range(num_passes):
        pn   = pass_numbers[p]
        f_sp = float(meta["f_spindle"].iloc[pn - 1])
        T = pass_data[p]; N = len(T); sL = N // num_segments
        for s in range(num_segments):
            seg = T.iloc[s*sL: min((s+1)*sL, N)]
            seg_data[p][s] = seg
            Fr  = np.sqrt(seg.Fx**2 + seg.Fy**2 + seg.Fz**2).values
            Fac = Fr - Fr.mean(); n = len(Fac)
            rms_store[p, s] = np.sqrt(np.mean(Fac**2))
            win  = windows.hann(n); NFFT = nextpow2(n)
            A    = (2/NFFT)*np.abs(np.fft.fft(Fac*win, NFFT))[:NFFT//2]
            f_ax = np.arange(NFFT//2)*(fs/NFFT)
            hm   = np.zeros_like(f_ax, dtype=bool)
            for h in range(1, n_harm+1):
                fh = h*f_sp; hm |= (f_ax>=fh-bw) & (f_ax<=fh+bw)
            nm   = (~hm) & (f_ax >= 5)
            nh   = A.copy(); nh[~nm] = 0
            valid = A[nm & (A>0)]
            nf = (np.median(valid) if valid.size else np.median(A[A>0])) + EPS
            if np.isnan(nf) or nf <= 0:
                nf = EPS
            chatter_index[p, s] = nh.max() / nf
            fft_store[p][s] = {"f_axis": f_ax, "amp": A}

    rms_base      = rms_store[0, :]
    rms_ratio_mat = rms_store / (rms_base + EPS)
    labels        = np.zeros((num_passes, num_segments))
    for p in range(num_passes):
        for s in range(num_segments):
            labels[p, s] = float(
                (chatter_index[p, s] > peak_ratio) and
                (rms_store[p, s]     > rms_ratio  * rms_base[s]))
    return dict(meta=meta, segData=seg_data, fftStore=fft_store,
                chatterIndex=chatter_index, rmsStore=rms_store,
                rmsBaseline=rms_base, rmsRatio=rms_ratio_mat,
                labels=labels, passNumbers=pass_numbers)


def extract_features(ds, num_passes, num_segments, n_harm=5, bw=5,
                     include_kurtosis=True):
    """
    Features (with kurtosis):
        [RMS_Resultant, Kurtosis_Resultant, Dominant_Freq_Hz,
         Seg_Position, Wall_Thickness_mm]
    Features (without kurtosis):
        [RMS_Resultant, Dominant_Freq_Hz, Seg_Position, Wall_Thickness_mm]
    """
    meta         = ds["meta"]
    pass_numbers = ds["passNumbers"]
    nf = 5 if include_kurtosis else 4
    F  = np.zeros((num_passes * num_segments, nf))
    L  = np.zeros(num_passes * num_segments)
    pv = np.zeros(num_passes * num_segments, dtype=int)
    row = 0
    for p in range(num_passes):
        pn     = pass_numbers[p]
        t_wall = float(meta["thickness"].iloc[pn - 1])
        f_sp   = float(meta["f_spindle"].iloc[pn - 1])
        for s in range(num_segments):
            seg   = ds["segData"][p][s]; ff = ds["fftStore"][p][s]
            Fr    = np.sqrt(seg.Fx**2 + seg.Fy**2 + seg.Fz**2).values
            Fr_ac = Fr - Fr.mean()
            rms   = np.sqrt(np.mean(Fr_ac**2))

            n_s  = len(Fr_ac); sig2 = np.var(Fr_ac)
            kurt = (np.mean((Fr_ac - Fr_ac.mean())**4)/(sig2**2+EPS) - 3.0
                    if (n_s >= 4 and sig2 > EPS) else 0.0)

            amp, f_ax = ff["amp"], ff["f_axis"]
            hm = np.zeros_like(f_ax, dtype=bool)
            for h in range(1, n_harm+1):
                fh = h*f_sp; hm |= (f_ax>=fh-bw) & (f_ax<=fh+bw)
            nm = (~hm) & (f_ax >= 5)
            amp_nh = amp.copy(); amp_nh[~nm] = 0
            dom_freq = f_ax[int(np.argmax(amp_nh))]
            seg_pos  = s / max(num_segments-1, 1)

            if include_kurtosis:
                F[row] = [rms, kurt, dom_freq, seg_pos, t_wall]
            else:
                F[row] = [rms, dom_freq, seg_pos, t_wall]
            L[row] = ds["labels"][p, s]
            pv[row] = pn
            row += 1
    return F, L, pv


def fix_nan(X):
    X = X.copy()
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            med = np.median(X[~bad, c])
            X[bad, c] = 0 if np.isnan(med) else med
    return X


def augment_minority(X, y, rng, imbal_thresh=1.5, jitter_sigma=0.05):
    """
    Augmentation is applied only when the stable:chatter ratio in the training
    fold exceeds imbal_thresh. Gaussian noise with standard deviation
    jitter_sigma (as a fraction of the training standard deviation) is added to
    minority-class samples in z-scored feature space.
    Both values are not formally validated — they are engineering choices.
    """
    nc = int(y.sum()); ns = int((y==0).sum())
    if nc == 0 or ns == 0 or ns/(nc+EPS) <= imbal_thresh:
        return X, y
    n_add = ns - nc
    ci    = np.where(y == 1)[0]
    idx   = ci[np.arange(n_add) % nc]
    X_ex  = X[idx] + jitter_sigma * rng.standard_normal((n_add, X.shape[1]))
    Xa    = np.vstack([X, X_ex]); ya = np.concatenate([y, np.ones(n_add)])
    sh    = rng.permutation(len(ya))
    return Xa[sh], ya[sh]


def all_metrics(y_true, y_pred, p_pred, name):
    TN = int(((y_true==0)&(y_pred==0)).sum()); FP = int(((y_true==0)&(y_pred==1)).sum())
    FN = int(((y_true==1)&(y_pred==0)).sum()); TP = int(((y_true==1)&(y_pred==1)).sum())
    acc  = (TP+TN)/(TN+FP+FN+TP+EPS)
    prec = TP/(TP+FP+EPS); rec = TP/(TP+FN+EPS)
    f1   = 2*prec*rec/(prec+rec+EPS)
    mse  = np.mean((p_pred-y_true)**2)
    ss_r = np.sum((y_true-p_pred)**2); ss_t = np.sum((y_true-y_true.mean())**2)
    r2   = 1 - ss_r/(ss_t+EPS)
    print(f"  [{name:<22}] Acc={acc*100:5.1f}%  Prec={prec*100:5.1f}%  "
          f"Rec={rec*100:5.1f}%  F1={f1*100:5.1f}%  MSE={mse:.4f}  R2={r2:.4f}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, mse=mse, r2=r2,
                C=np.array([[TN, FP],[FN, TP]]))


def pos_proba(clf, X):
    p = clf.predict_proba(X); cls = list(clf.classes_)
    return p[:, cls.index(1)] if 1 in cls else np.zeros(len(X))


class PatternNet(nn.Module):
    def __init__(self, n_input=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 32), nn.Tanh(),
            nn.Linear(32, 16),      nn.Tanh(),
            nn.Linear(16, 1),       nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(X, y, device, seed, n_input=4, epochs=300, lr=0.01):
    """Train the two-layer MLP. Learning rate and epoch count are fixed across
    all folds; no per-fold early stopping or hyperparameter tuning is applied."""
    torch.manual_seed(seed)
    net = PatternNet(n_input=n_input).to(device)
    Xt  = torch.tensor(X, dtype=torch.float32, device=device)
    yt  = torch.tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit= nn.BCELoss()
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = crit(net(Xt), yt); loss.backward(); opt.step()
    net.eval()
    return net


# ---- RMS Force Map colormap --------------------------------------------------
def rms_force_colormap(max_ratio, thresh):
    """
    thresh: the RMS ratio value where the colour transitions to cyan.
    This is a DISPLAY threshold, not the labelling threshold.
    Set to rmsDisplayThreshold in Section 0.
    """
    span = max(max_ratio - 1.0, 1e-6)
    t    = min(max((thresh - 1.0)/span, 0.05), 0.9)
    anchors = [
        (0.0,                (0.03, 0.05, 0.45)),
        (max(t*0.55, 0.02),  (0.05, 0.35, 0.95)),
        (t,                  (0.10, 0.95, 1.00)),
        (t+(1-t)*0.22,       (1.00, 0.95, 0.15)),
        (t+(1-t)*0.55,       (1.00, 0.55, 0.00)),
        (1.0,                (0.85, 0.05, 0.05)),
    ]
    pos = [max(0.0, min(1.0, a[0])) for a in anchors]
    for i in range(1, len(pos)):
        if pos[i] <= pos[i-1]:
            pos[i] = min(1.0, pos[i-1]+1e-3)
    return LinearSegmentedColormap.from_list(
        "rms_chatter", list(zip(pos, [a[1] for a in anchors])))


def draw_force_map(ax, ratio_mat, mark_mat, num_passes, num_segments,
                   title, max_ratio, thresh, pass_labels=None,
                   show_ylabel=True):
    """Draw one RMS-force heatmap with chatter marks.
    show_ylabel=False suppresses 'Pass Number' label — use for right-column
    panels in multi-panel figures to avoid overlap with the shared colorbar."""
    if pass_labels is None:
        pass_labels = list(range(1, num_passes+1))
    cmap = rms_force_colormap(max_ratio, thresh)
    im   = ax.imshow(ratio_mat, cmap=cmap, vmin=1.0, vmax=max_ratio, aspect='auto')
    ax.grid(False)                          # no gridlines over the heatmap cells
    # Thin cell separators.
    for r in np.arange(-0.5, num_passes, 1):
        ax.axhline(r, color=[0.35,0.35,0.35], lw=0.4)
    for c in np.arange(-0.5, num_segments, 1):
        ax.axvline(c, color=[0.35,0.35,0.35], lw=0.4)
    # Confirmed-chatter cells: white "C" + orange/red border highlight.
    from matplotlib.patches import Rectangle
    for p in range(num_passes):
        for s in range(num_segments):
            if mark_mat[p, s] == 1:
                ax.add_patch(Rectangle((s-0.5, p-0.5), 1, 1, fill=False,
                                       edgecolor="#FF6600", lw=2.0, zorder=3))
                ax.text(s, p, "C", ha='center', va='center',
                        color='w', fontsize=INSIDE_FONT, fontweight='bold', zorder=4)
    ax.set_xticks(range(num_segments))
    ax.set_xticklabels([f"S{s+1}" for s in range(num_segments)])
    ax.set_yticks(range(num_passes))
    ax.set_yticklabels([f"P{pn}" for pn in pass_labels])
    ax.set_xlabel("Segment  (S1 = Free End  to  S9 = Clamped End)")
    if show_ylabel:
        ax.set_ylabel("Pass Number")
    else:
        ax.set_yticks(range(numPasses := num_passes))
        ax.set_yticklabels([f"P{pn}" for pn in pass_labels])
        ax.set_ylabel("")          # no label; tick labels still visible
    ax.set_title(title)
    ax.tick_params(length=0)
    return im


def save_pdf(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', format='pdf')
    print(f"  Saved: {path}")


# ---- Confusion-matrix colourmap: white -> yellow -> orange -> red -----------
CM_CMAP = LinearSegmentedColormap.from_list(
    "cm_yor", [(0.0, "#FFFFFF"), (0.25, "#FFD700"),
               (0.60, "#FF6600"), (1.0, "#CC0000")])


# =============================================================================
#  FILE SELECTION
# =============================================================================

def natkey(path):
    nums = re.findall(r"(\d+)", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def _macos_choose_file(prompt, initial_dir, multiple=False):
    import subprocess
    start         = 'set startDir to POSIX file "' + initial_dir + '" as alias'
    ftype         = 'of type {"xlsx","xls","csv"}'
    prompt_clause = 'with prompt "' + prompt + '"'
    if multiple:
        lines = [start,
                 "set theFiles to choose file " + prompt_clause
                 + " default location startDir " + ftype
                 + " with multiple selections allowed",
                 'set out to ""',
                 "repeat with f in theFiles",
                 '    set out to out & POSIX path of f & "\\n"',
                 "end repeat", "out"]
    else:
        lines = [start,
                 "POSIX path of (choose file " + prompt_clause
                 + " default location startDir " + ftype + ")"]
    result = subprocess.run(["osascript", "-e", "\n".join(lines)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("osascript: " + result.stderr.strip())
    out = result.stdout.strip()
    if multiple:
        return [p.strip() for p in out.split("\n") if p.strip()]
    return out


def select_files(data_folder_default, pass_folder_default, meta_file_default,
                 force_no_gui=False):
    # Positional file args: only those that are NOT argparse flags.
    # (Strip --mode X / --no-gui so batch runs like `--mode B` are not
    #  mistaken for file paths.)
    flag_free = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == '--mode':
            skip_next = True            # also skip its value
            continue
        if a.startswith('--'):
            continue                    # --no-gui, --mode=X, etc.
        flag_free.append(a)
    if len(flag_free) >= 2:
        return flag_free[0], sorted(flag_free[1:], key=natkey)

    if force_no_gui:
        print("  --no-gui set: using hard-coded default paths.")
        files = []
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            files += glob.glob(os.path.join(pass_folder_default, ext))
        meta_abs = os.path.abspath(meta_file_default)
        files = [f for f in files if os.path.abspath(f) != meta_abs]
        return meta_file_default, sorted(files, key=natkey)

    if sys.platform == "darwin":
        try:
            print("  [Dialog 1/2] Select METADATA file...")
            mf = _macos_choose_file(
                "Select METADATA FILE  [Vc | ap | f | Outer Dia | Inner Dia]",
                initial_dir=data_folder_default, multiple=False)
            print("  [Dialog 2/2] Select ALL pass files (Cmd+A to select all)...")
            pf = _macos_choose_file(
                "Select ALL PASS FILES  [Time | Fx | Fy | Fz]",
                initial_dir=pass_folder_default, multiple=True)
            return mf, sorted(pf, key=natkey)
        except RuntimeError as e:
            print(f"  (Dialog failed: {e}) — using hard-coded paths.")
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            mf = filedialog.askopenfilename(
                title="SELECT METADATA FILE",
                initialdir=data_folder_default,
                filetypes=[("Excel/CSV","*.xlsx *.xls *.csv"),("All","*.*")])
            try: root.quit()
            except Exception: pass
            try: root.destroy()
            except Exception: pass
            if not mf:
                raise RuntimeError("No metadata file selected.")
            root2 = tk.Tk(); root2.withdraw()
            pf = filedialog.askopenfilenames(
                title="SELECT ALL PASS FILES",
                initialdir=pass_folder_default,
                filetypes=[("Excel/CSV","*.xlsx *.xls *.csv"),("All","*.*")])
            try: root2.quit()
            except Exception: pass
            try: root2.destroy()
            except Exception: pass
            if not pf:
                raise RuntimeError("No pass files selected.")
            return mf, sorted(list(pf), key=natkey)
        except Exception as e:
            print(f"  (No GUI: {e})")
    print(f"  Falling back to hard-coded paths.")
    files = []
    for ext in ("*.xlsx","*.xls","*.csv"):
        files += glob.glob(os.path.join(pass_folder_default, ext))
    meta_abs = os.path.abspath(meta_file_default)
    files = [f for f in files if os.path.abspath(f) != meta_abs]
    return meta_file_default, sorted(files, key=natkey)


# =============================================================================
#  MAIN
# =============================================================================

def run_validation(val_mode, pass_numbers, pvA, FA5, FA_nn, LA,
                   featureNames_5, featureNames_nn, nFeat_nn,
                   device, SEED, imbal_thresh, jitter_sigma,
                   nn_epochs_lopo, nn_lr_lopo, nTrees, rfMaxFeatures,
                   svm_C, svm_gamma, lr_max_iter,
                   fixed_train_frac=0.667):
    """
    Run pass-level validation in one of three schemes.

      val_mode == 'LOPO'    : hold out pass k, train on ALL other passes
                              (both earlier and later).  Interpolation — each
                              held-out pass is bracketed by thicker and thinner
                              training passes, so wall thickness is in-range.

      val_mode == 'FORWARD' : predict pass k using ONLY passes 1..k-1
                              (chronological forward-chaining).  True prospective
                              deployment: future passes have not been machined.
                              Chatter onset is not predictable before it is first
                              seen, so early passes are skipped.

      val_mode == 'FIXED'   : train on the FIRST ~2/3 of passes, test on the
                              LAST ~1/3 (e.g. train 1-10, test 11-15).  This is
                              the original split.  It places the entire test set
                              BELOW the training range of wall thickness
                              (extrapolation), which tree ensembles cannot do —
                              reported to demonstrate WHY LOPO was adopted.

    Returns (pred_dict, prob_dict) with per-segment predictions.  Unevaluated
    passes are left as NaN.
    """
    models_list = ["NN", "RF", "SVM", "LR"]
    pred_dict = {m: np.full(len(LA), np.nan) for m in models_list}
    prob_dict = {m: np.full(len(LA), np.nan) for m in models_list}

    sorted_passes = sorted(pass_numbers)

    # -------- FIXED split is a single train/test partition, not a loop --------
    if val_mode == 'FIXED':
        n_pass    = len(sorted_passes)
        n_train   = max(1, int(round(fixed_train_frac * n_pass)))
        train_ps  = set(sorted_passes[:n_train])
        test_ps   = set(sorted_passes[n_train:])
        tr_mask   = np.isin(pvA, list(train_ps))
        te_mask   = np.isin(pvA, list(test_ps))
        print(f"  FIXED split: train on passes "
              f"{sorted_passes[0]}-{sorted_passes[n_train-1]} "
              f"({int(tr_mask.sum())} seg), "
              f"test on passes {sorted_passes[n_train]}-{sorted_passes[-1]} "
              f"({int(te_mask.sum())} seg)")
        # Wall thickness is the last column of FA5; report the extrapolation gap.
        try:
            w_idx = featureNames_5.index('Wall_Thickness_mm')
            w_tr  = FA5[tr_mask, w_idx]; w_te = FA5[te_mask, w_idx]
            print(f"  Wall thickness  train: {w_tr.min():.2f}-{w_tr.max():.2f} mm"
                  f"  |  test: {w_te.min():.2f}-{w_te.max():.2f} mm")
            if w_te.max() < w_tr.min():
                print(f"  >> Every test wall is thinner than every training wall "
                      f"-> EXTRAPOLATION (RF cannot extrapolate).")
        except ValueError:
            pass

        _fit_predict_block(tr_mask, te_mask, pvA, FA5, FA_nn, LA,
                           featureNames_5, featureNames_nn, nFeat_nn,
                           device, SEED, imbal_thresh, jitter_sigma,
                           nn_epochs_lopo, nn_lr_lopo, nTrees, rfMaxFeatures,
                           svm_C, svm_gamma, lr_max_iter,
                           pred_dict, prob_dict, seed_tag=999)
        # Report block prediction counts
        for k in models_list:
            got = pred_dict[k][te_mask]
            if not np.isnan(got).all():
                print(f"    {k}: predicted {int(np.nansum(got))} chatter "
                      f"of {int(te_mask.sum())} test segments")
        return pred_dict, prob_dict

    # -------- LOPO and FORWARD are per-pass loops --------
    for held in sorted_passes:
        if val_mode == 'LOPO':
            tr_mask = pvA != held
        elif val_mode == 'FORWARD':
            tr_mask = pvA < held           # only earlier passes
        else:
            raise ValueError(val_mode)
        te_mask = pvA == held

        ytr_check = LA[tr_mask]
        # FORWARD: earliest passes have no / single-class history -> skip.
        if val_mode == 'FORWARD':
            if tr_mask.sum() < 9:                       # < 1 full pass of data
                print(f"  P{held:2d}: skipped (only {int(tr_mask.sum())} "
                      f"training segments before it)")
                continue
            if len(np.unique(ytr_check)) < 2:
                print(f"  P{held:2d}: skipped (training history is single-class "
                      f"— no chatter seen yet)")
                continue

        _fit_predict_block(tr_mask, te_mask, pvA, FA5, FA_nn, LA,
                           featureNames_5, featureNames_nn, nFeat_nn,
                           device, SEED, imbal_thresh, jitter_sigma,
                           nn_epochs_lopo, nn_lr_lopo, nTrees, rfMaxFeatures,
                           svm_C, svm_gamma, lr_max_iter,
                           pred_dict, prob_dict, seed_tag=int(held))

        print(f"  P{held:2d}: " +
              "  ".join(
                  f"{k}={int(pred_dict[k][te_mask].sum()) if not np.isnan(pred_dict[k][te_mask]).all() else '-'}/{int(te_mask.sum())}"
                  for k in models_list))

    return pred_dict, prob_dict


def _fit_predict_block(tr_mask, te_mask, pvA, FA5, FA_nn, LA,
                       featureNames_5, featureNames_nn, nFeat_nn,
                       device, SEED, imbal_thresh, jitter_sigma,
                       nn_epochs_lopo, nn_lr_lopo, nTrees, rfMaxFeatures,
                       svm_C, svm_gamma, lr_max_iter,
                       pred_dict, prob_dict, seed_tag):
    """Train all four models on tr_mask and predict te_mask. Fills pred/prob dicts.
    Shared by LOPO, FORWARD and FIXED so all three use identical preprocessing,
    augmentation and model settings."""
    # LR / SVM / RF: full feature arrays
    Xtr5_raw = FA5[tr_mask]; Xte5_raw = FA5[te_mask]
    mu5 = Xtr5_raw.mean(0); sig5 = Xtr5_raw.std(0, ddof=1); sig5[sig5 == 0] = 1
    Xtr5 = fix_nan((Xtr5_raw - mu5) / sig5)
    Xte5 = fix_nan((Xte5_raw - mu5) / sig5)

    # NN feature arrays
    Xtr_nn_raw = FA_nn[tr_mask]; Xte_nn_raw = FA_nn[te_mask]
    mu_nn  = Xtr_nn_raw.mean(0)
    sig_nn = Xtr_nn_raw.std(0, ddof=1); sig_nn[sig_nn == 0] = 1
    Xtr_nn = fix_nan((Xtr_nn_raw - mu_nn) / sig_nn)
    Xte_nn = fix_nan((Xte_nn_raw - mu_nn) / sig_nn)

    ytr = LA[tr_mask]
    rng = np.random.default_rng(SEED + seed_tag)

    Xa5, ya = augment_minority(Xtr5, ytr, rng, imbal_thresh, jitter_sigma)
    nn_cols = [featureNames_5.index(nm) for nm in featureNames_nn]
    Xa_nn   = Xa5[:, nn_cols]
    ya_nn   = ya
    has_both = len(np.unique(ya)) > 1
    held = seed_tag                                     # for messages

    # 1. Neural Network
    try:
        net = train_nn(Xa_nn, ya_nn, device, seed=SEED + int(held),
                       n_input=nFeat_nn,
                       epochs=nn_epochs_lopo, lr=nn_lr_lopo)
        with torch.no_grad():
            pn = net(torch.tensor(Xte_nn, dtype=torch.float32,
                                  device=device)).cpu().numpy()
        prob_dict["NN"][te_mask] = pn
        pred_dict["NN"][te_mask] = (pn >= 0.5).astype(float)
    except Exception as e:
        print(f"    NN failed P{held}: {e}")

    # 2. Random Forest
    try:
        rf = RandomForestClassifier(
            n_estimators=nTrees, bootstrap=True,
            max_features=rfMaxFeatures,
            min_samples_leaf=2, random_state=SEED + int(held))
        rf.fit(Xa5, ya)
        prob_dict["RF"][te_mask] = pos_proba(rf, Xte5)
        pred_dict["RF"][te_mask] = rf.predict(Xte5).astype(float)
    except Exception as e:
        print(f"    RF failed P{held}: {e}")

    # 3. SVM
    if has_both:
        try:
            svm = SVC(kernel='rbf', C=svm_C, gamma=svm_gamma,
                      probability=True, random_state=SEED + int(held))
            svm.fit(Xa5, ya)
            prob_dict["SVM"][te_mask] = pos_proba(svm, Xte5)
            pred_dict["SVM"][te_mask] = svm.predict(Xte5).astype(float)
        except Exception as e:
            print(f"    SVM failed P{held}: {e}")

    # 4. Logistic Regression
    if has_both:
        try:
            lr_m = LogisticRegression(max_iter=lr_max_iter,
                                      random_state=SEED + int(held))
            lr_m.fit(Xa5, ya)
            prob_dict["LR"][te_mask] = pos_proba(lr_m, Xte5)
            pred_dict["LR"][te_mask] = lr_m.predict(Xte5).astype(float)
        except Exception as e:
            print(f"    LR failed P{held}: {e}")


def main():
    # =========================================================================
    #  SECTION 0 — CONFIGURATION
    #  All parameters that govern the analysis are defined here. Each is
    #  annotated to state whether it is an experimentally measured value or a
    #  methodological choice made in this study.
    # =========================================================================

    # ---- Signal / segmentation ----
    numSegments      = 9       # Each pass is divided into 9 equal-length
                               # axial segments (S1 = free end, S9 = clamped
                               # end). Methodological choice; see manuscript
                               # Section 3.5.
    fs               = 1000    # [Hz] Sampling frequency of the force signal.
                               # Measured value; Nyquist limit = 500 Hz.

    # ---- Chatter labelling thresholds ----
    # A segment is labelled as chatter only when BOTH criteria are satisfied.
    # These thresholds were calibrated through human-in-the-loop verification:
    # automatically generated labels were compared against visual inspection of
    # chatter marks on the machined surface and against the force records, and
    # the thresholds were adjusted until agreement was reached. They are
    # empirical values for this setup, not derived from first principles.
    # See manuscript Section 3.7.1.
    chatterPeakRatio = 3.5     # Criterion 1: non-harmonic spectral peak divided
                               # by the non-harmonic noise floor (median).
    chatterRMSRatio  = 2.5     # Criterion 2: segment RMS divided by the Pass 1
                               # RMS baseline for the SAME segment position.

    # ---- FFT / harmonic masking ----
    numHarmonics     = 5       # Number of spindle-frequency harmonics masked
                               # before the non-harmonic peak is located.
    harmonicBW       = 5       # [Hz] Half-width of the mask applied around each
                               # harmonic. Methodological choice; see Section 3.7.1.

    # ---- Display threshold on the RMS Force Map ----
    # Controls WHERE the colour transitions to cyan on the heatmap.
    # This does NOT change which segments are labelled chatter —
    # it only sets the visual landmark on the colorbar.
    rmsDisplayThreshold = 2.5  # [× baseline] — matches chatterRMSRatio (the labelling
                               # criterion). Colormap transitions at the actual chatter
                               # threshold so the force map and labelling are consistent.

    # ---- Features ----
    # featureNames (all 5): used by LR, SVM, RF.
    # NN uses 4 features by default (kurtosis excluded) to recover recall.
    featureNames_5 = ['RMS_Resultant', 'Kurtosis_Resultant',
                      'Dominant_Freq_Hz', 'Seg_Position', 'Wall_Thickness_mm']
    featureNames_4 = ['RMS_Resultant', 'Dominant_Freq_Hz',
                      'Seg_Position', 'Wall_Thickness_mm']

    # Kurtosis is excluded from the Neural Network input by default. Empirical
    # testing showed that including it as a fifth NN input reduced NN chatter
    # recall below the 90% acceptance threshold used in this study, while the
    # other three classifiers were unaffected. Set to True to use all five
    # features for every classifier. See manuscript Section 4.4.1.
    NN_USE_KURTOSIS = False    # False -> NN uses 4 features.
                               # True  -> NN uses 5 (same as LR/SVM/RF).
    featureNames_nn = featureNames_5 if NN_USE_KURTOSIS else featureNames_4
    nFeat_full      = 5
    nFeat_nn        = 5 if NN_USE_KURTOSIS else 4

    # ---- FEATURE ABLATION CONFIGURATIONS ----------------------------------
    #  Wall thickness is collinear with pass number in a single-workpiece
    #  sequential experiment, because t_wall = t0 - ap*(pass - 1). The ablation
    #  configurations below isolate the contribution of each feature group
    #  under an identical validation protocol.
    #
    #  Set via command line:  --mode A  ...  --mode H
    #
    #   A  baseline             [RMS, Kurtosis, DomFreq, SegPos, WallThickness]
    #   B  swap wall -> pass    [RMS, Kurtosis, DomFreq, SegPos, PassNumber]
    #   C  drop wall thickness  [RMS, Kurtosis, DomFreq, SegPos]
    #   D  force + spectral     [RMS, Kurtosis, DomFreq]
    #   E  RMS only             [RMS]
    #   F  force + spectral     [RMS, Kurtosis, DomFreq]        (alias of D)
    #   G  geometry only        [SegPos, WallThickness]
    #   H  no RMS               [Kurtosis, DomFreq, SegPos, WallThickness]
    #
    #  Interpretation of each configuration (recall relative to mode A):
    #   B  tests whether wall thickness carries information beyond pass order.
    #   C  tests what wall thickness contributes when force features are present.
    #   D/F tests classification using force and spectral features only.
    #   E  single-feature baseline using the force amplitude alone.
    #   G  tests whether the two geometry features are sufficient on their own.
    #   H  tests whether performance depends on the RMS feature, which also
    #      participates in the chatter labelling rule (label-feature dependence).
    # -------------------------------------------------------------------------
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument('--mode', default='A',
                     choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                     help="ablation mode (A=baseline ... H=no-RMS)")
    _ap.add_argument('--no-gui', action='store_true',
                     help="skip file dialogs; use hard-coded default paths")
    _args, _ = _ap.parse_known_args()
    ABLATION_MODE = _args.mode

    # ---- Model hyperparameters ----
    # No grid search or hyperparameter optimisation was performed. Values are
    # either scikit-learn defaults or were fixed a priori. This is a deliberate
    # choice: with 135 observations, hyperparameter tuning on the same data used
    # for evaluation would risk optimistic bias. See manuscript Section 3.7.5.
    rfMaxFeatures    = 1       # Features considered per split. Set to 1 so that
                               # no single feature dominates every tree split.
    nTrees           = 200     # Random Forest ensemble size.
    svm_C            = 1.0     # scikit-learn default.
    svm_gamma        = 'scale' # scikit-learn default.
    lr_max_iter      = 1000    # Convergence budget for Logistic Regression.
    nn_epochs_lopo   = 300     # Fixed epoch count; no per-fold early stopping.
    nn_lr_lopo       = 0.01    # Learning rate for the Adam optimiser.
    nn_epochs_full   = 300     # Section-9 full-model training.
    nn_lr_full       = 0.001
    nn_patience      = 20      # Early stopping patience (Section 9 only).

    # ---- Minority-class augmentation ----
    # Applied ONLY to the training portion of each fold, after the fold split,
    # so no synthetic data enters any test set. See manuscript Section 3.7.4.
    imbal_thresh     = 1.5     # Augment when the stable:chatter ratio in the
                               # training fold exceeds this value.
    jitter_sigma     = 0.05    # Gaussian noise standard deviation, expressed as
                               # a fraction of the training standard deviation
                               # (5%), applied in z-scored feature space.

    # ---- Evaluation ----
    SEED             = 42      # Fixed random seed for reproducibility. The
                               # per-fold seed is SEED + pass_number, so each
                               # fold is independently reproducible.
    # The decision threshold is 0.5 for all classifiers. It was not optimised
    # for recall. Lowering the threshold would increase recall at the cost of
    # precision; this trade-off is not explored in this study.

    # ---- Output ----
    # Figures and the results CSV are written here.
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"ablation_{ABLATION_MODE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_NAMES = {
        "NN":  "Neural Network",
        "RF":  "Random Forest",
        "SVM": "SVM (RBF)",
        "LR":  "Logistic Regression",
    }
    # Colours keyed by feature NAME, so any subset (modes A-H) gets stable colours.
    FEAT_COLOUR = {
        'RMS_Resultant':      [0.20, 0.50, 0.85],   # blue
        'Kurtosis_Resultant': [0.60, 0.20, 0.80],   # purple
        'Dominant_Freq_Hz':   [0.85, 0.20, 0.20],   # red
        'Seg_Position':       [0.20, 0.75, 0.40],   # green
        'Wall_Thickness_mm':  [0.70, 0.50, 0.20],   # brown
        'Pass_Number':        [0.45, 0.45, 0.45],   # grey (mode B only)
    }
    FEAT_DISPLAY_NAMES = {
        'RMS_Resultant':      'Resultant CF RMS',
        'Kurtosis_Resultant': 'Kurtosis',
        'Dominant_Freq_Hz':   'Dominant Frequency (Hz)',
        'Seg_Position':       'Segment Position',
        'Wall_Thickness_mm':  'Wall Thickness (mm)',
        'Pass_Number':        'Pass Number',
    }
    def _clrs_for(names):
        return np.array([FEAT_COLOUR.get(n, [0.5, 0.5, 0.5]) for n in names])
    def _display_name(n):
        return FEAT_DISPLAY_NAMES.get(n, n.replace('_', ' '))

    # Default data paths
    # Default data folder. Change this to point at your own data, or pass the
    # metadata file and pass files as command-line arguments, or use the
    # interactive file dialog (default when no arguments are given).
    DATA_FOLDER_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    PASS_FOLDER_DEFAULT = os.path.join(DATA_FOLDER_DEFAULT, "Pass")
    META_FILE_DEFAULT   = os.path.join(DATA_FOLDER_DEFAULT,
                                       "meta_data_Tube_2026_1.4.xlsx")

    # ---- Font ----
    _set_font("Palatino Linotype", size=16)

    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    #  SECTION 1 — LOAD DATA
    # =========================================================================
    print("=== SECTION 1: LOADING DATA ===")
    metaFile, passFiles = select_files(DATA_FOLDER_DEFAULT,
                                       PASS_FOLDER_DEFAULT,
                                       META_FILE_DEFAULT,
                                       force_no_gui=_args.no_gui)
    numPasses = len(passFiles)
    if numPasses < 3:
        raise RuntimeError(f"Only {numPasses} files — need >= 3 for LOPO.")
    pass_numbers = [natkey(f) for f in passFiles]
    print(f"  Metadata  : {metaFile}")
    print(f"  Pass files: {numPasses}  passes: {pass_numbers}")
    print(f"  Device    : {device}")

    full_range = list(range(min(pass_numbers), max(pass_numbers)+1))
    missing    = [n for n in full_range if n not in pass_numbers]
    if missing:
        print(f"\n  *** WARNING: Missing pass files: {missing} ***")
        print(f"  *** Metadata indexed by pass number — alignment is correct. ***\n")
    else:
        print("  Pass numbering complete — no gaps.")

    ds = load_dataset(passFiles, metaFile, numPasses, numSegments, fs,
                      chatterPeakRatio, chatterRMSRatio,
                      numHarmonics, harmonicBW, pass_numbers, "A")

    print("\n--- Label summary ---")
    print(f"  Segments: {numPasses*numSegments}  "
          f"Chatter: {int(ds['labels'].sum())} "
          f"({100*ds['labels'].mean():.0f}%)")

    # =========================================================================
    #  SECTION 2 — FEATURE EXTRACTION
    #  5 features for LR / SVM / RF; 4 (no kurtosis) for NN unless overridden.
    # =========================================================================
    print("\n=== SECTION 2: FEATURE EXTRACTION ===")
    FA5, LA, pvA = extract_features(ds, numPasses, numSegments,
                                    numHarmonics, harmonicBW,
                                    include_kurtosis=True)

    # ---- APPLY ABLATION -----------------------------------------------------
    #  FA5 from extract_features has fixed column order:
    #     [0]=RMS  [1]=Kurtosis  [2]=DomFreq  [3]=SegPos  [4]=WallThick
    #
    #  Strategy: name every requested feature per mode, then BUILD the feature
    #  matrix by selecting/replacing columns explicitly.  This is robust for all
    #  8 modes and removes the previous assumption that "kurtosis is always at
    #  index 1" (which broke when RMS or other columns were dropped).
    #
    #  We keep two matrices:
    #     FA_full  -> features for RF / LR / SVM   (all requested features)
    #     FA_nn    -> features for the NN          (FA_full minus kurtosis,
    #                                               unless NN_USE_KURTOSIS=True,
    #                                               and only if kurtosis is present)
    _RMS, _KURT, _DOM, _SEG, _WALL = 0, 1, 2, 3, 4   # column indices in FA5

    if ABLATION_MODE == 'A':
        cols  = [_RMS, _KURT, _DOM, _SEG, _WALL]
        names = ['RMS_Resultant', 'Kurtosis_Resultant',
                 'Dominant_Freq_Hz', 'Seg_Position', 'Wall_Thickness_mm']
        FA_full = FA5[:, cols].copy()

    elif ABLATION_MODE == 'B':                 # swap t_wall -> pass number
        cols  = [_RMS, _KURT, _DOM, _SEG, _WALL]
        names = ['RMS_Resultant', 'Kurtosis_Resultant',
                 'Dominant_Freq_Hz', 'Seg_Position', 'Pass_Number']
        FA_full = FA5[:, cols].copy()
        FA_full[:, 4] = pvA.astype(float)      # overwrite wall column with pass #

    elif ABLATION_MODE == 'C':                 # drop t_wall
        cols  = [_RMS, _KURT, _DOM, _SEG]
        names = ['RMS_Resultant', 'Kurtosis_Resultant',
                 'Dominant_Freq_Hz', 'Seg_Position']
        FA_full = FA5[:, cols].copy()

    elif ABLATION_MODE in ('D', 'F'):          # force/spectral only (no geometry)
        cols  = [_RMS, _KURT, _DOM]
        names = ['RMS_Resultant', 'Kurtosis_Resultant', 'Dominant_Freq_Hz']
        FA_full = FA5[:, cols].copy()

    elif ABLATION_MODE == 'E':                 # RMS only
        cols  = [_RMS]
        names = ['RMS_Resultant']
        FA_full = FA5[:, cols].copy()

    elif ABLATION_MODE == 'G':                 # geometry only (make-or-break)
        cols  = [_SEG, _WALL]
        names = ['Seg_Position', 'Wall_Thickness_mm']
        FA_full = FA5[:, cols].copy()

    elif ABLATION_MODE == 'H':                 # NO RMS (circularity test)
        cols  = [_KURT, _DOM, _SEG, _WALL]
        names = ['Kurtosis_Resultant', 'Dominant_Freq_Hz',
                 'Seg_Position', 'Wall_Thickness_mm']
        FA_full = FA5[:, cols].copy()

    featureNames_5 = names                     # (kept name for downstream code)

    # ---- Build the NN feature matrix ----------------------------------------
    # NN excludes kurtosis by default (NN_USE_KURTOSIS=False) to protect recall.
    # But only drop it if kurtosis is actually present in this mode.
    if (not NN_USE_KURTOSIS) and ('Kurtosis_Resultant' in featureNames_5):
        keep_nn = [i for i, nm in enumerate(featureNames_5)
                   if nm != 'Kurtosis_Resultant']
        FA_nn           = FA_full[:, keep_nn].copy()
        featureNames_nn = [featureNames_5[i] for i in keep_nn]
    else:
        FA_nn           = FA_full.copy()
        featureNames_nn = list(featureNames_5)

    # ---- Feature counts (flow into PatternNet input size) --------------------
    nFeat_full = FA_full.shape[1]
    nFeat_nn   = FA_nn.shape[1]

    # Guard: some modes leave the NN with 1 feature (e.g. Mode E after dropping
    # kurtosis leaves only RMS; Mode D/F leaves RMS+DomFreq).  A 1-feature NN is
    # valid but degenerate; warn so the result is interpreted with care.
    if nFeat_nn < 1:
        raise RuntimeError(f"Mode {ABLATION_MODE}: NN has 0 features after "
                           f"kurtosis removal. Set NN_USE_KURTOSIS=True for this mode.")
    if nFeat_nn == 1:
        print(f"  [NOTE] Mode {ABLATION_MODE}: NN has only 1 input feature "
              f"({featureNames_nn[0]}) — interpret NN result with care.")

    # From here on, downstream code expects FA5 to be the RF/LR/SVM matrix.
    FA5 = FA_full

    print(f"  ABLATION_MODE = {ABLATION_MODE}  |  "
          f"nFeat_full = {nFeat_full}  |  nFeat_nn = {nFeat_nn}")
    print(f"  RF/LR/SVM features: {' | '.join(featureNames_5)}")
    print(f"  NN features       : {' | '.join(featureNames_nn)}")
    print(f"  Samples: {len(LA)} | {100*LA.mean():.0f}% chatter")

    # =========================================================================
    #  SECTION 3 — RMS FORCE MAP  (Fig 1)
    # =========================================================================
    print("\n=== SECTION 3: RMS FORCE MAP ===")
    ratio_mat = ds["rmsRatio"]
    max_ratio = float(np.nanmax(ratio_mat))
    fig1, ax1 = plt.subplots(figsize=(13, 7))
    im = draw_force_map(ax1, ratio_mat, ds["labels"], numPasses, numSegments,
                        'RMS Force Map  |  Blue = stable, Red = severe  |  '
                        '"C" = confirmed chatter',
                        max_ratio, rmsDisplayThreshold, pass_labels=pass_numbers)
    cbar = fig1.colorbar(im, ax=ax1,
                         ticks=[1.0, rmsDisplayThreshold, max_ratio])
    cbar.ax.set_yticklabels([
        "1.0  (Stable baseline)",
        f"{rmsDisplayThreshold:.1f}x  (Threshold)",
        f"{max_ratio:.1f}x  (Severe -- max)"])
    fig1.tight_layout()
    save_pdf(fig1, "Fig1_RMS_Force_Map.pdf", OUTPUT_DIR)
    print(f"  Max RMS ratio = {max_ratio:.2f}\u00d7")

    # =========================================================================
    #  SECTION 4 — CROSS-VALIDATION  (LOPO  +  FORWARD-CHAINING)
    #  Both schemes are run.  LOPO holds out pass k and trains on all other
    #  passes (interpolation).  FORWARD trains only on passes 1..k-1
    #  (prospective deployment / extrapolation).  Sections 5-8 then produce a
    #  full set of metrics and figures for EACH scheme.
    # =========================================================================
    models_list = ["NN", "RF", "SVM", "LR"]

    validation_runs = {}   # val_mode -> (pred_dict, prob_dict, metrics)

    for val_mode in ['LOPO', 'FORWARD', 'FIXED']:
        tag = val_mode
        print(f"\n=== SECTION 4: {val_mode} CROSS-VALIDATION ===")
        pred_dict, prob_dict = run_validation(
            val_mode, pass_numbers, pvA, FA5, FA_nn, LA,
            featureNames_5, featureNames_nn, nFeat_nn,
            device, SEED, imbal_thresh, jitter_sigma,
            nn_epochs_lopo, nn_lr_lopo, nTrees, rfMaxFeatures,
            svm_C, svm_gamma, lr_max_iter)

        # ---- Section 5: metrics for this scheme ----
        print(f"\n=== SECTION 5: {val_mode} METRICS ===")
        metrics = {}
        for k in models_list:
            if not np.isnan(pred_dict[k]).all():
                metrics[k] = all_metrics(LA, pred_dict[k], prob_dict[k],
                                         MODEL_NAMES[k])
            else:
                print(f"  [{MODEL_NAMES[k]:<22}] Skipped — no valid predictions")

        # ---- Per-fold recall (pass-level) ----
        # Recall for each held-out pass separately.  Only folds that contain
        # at least one chatter segment are included — folds with zero chatter
        # have undefined recall and are excluded from the mean/SD.
        print(f"\n  Per-fold recall (pass-level, chatter-containing folds only):")
        fold_recall_summary = {}
        for k in models_list:
            yp = pred_dict[k]
            if np.isnan(yp).all():
                continue
            fold_recs = []
            for held in sorted(set(pvA)):
                mask = pvA == held
                yt_f = LA[mask]
                yp_f = yp[mask]
                valid = ~np.isnan(yp_f)
                yt_v  = yt_f[valid]; yp_v = yp_f[valid]
                n_chat = int(yt_v.sum())
                if n_chat == 0:
                    continue          # no chatter in this fold → skip
                tp = int(((yt_v == 1) & (yp_v == 1)).sum())
                rec = 100.0 * tp / n_chat
                fold_recs.append((held, n_chat, tp, rec))
            if fold_recs:
                recs = [r[3] for r in fold_recs]
                mn   = np.mean(recs); sd = np.std(recs, ddof=1)
                mn_r = np.min(recs);  mx_r = np.max(recs)
                fold_recall_summary[k] = {
                    'mean': mn, 'sd': sd, 'min': mn_r, 'max': mx_r,
                    'n_folds': len(fold_recs), 'folds': fold_recs
                }
                print(f"  [{MODEL_NAMES[k]:<22}] mean={mn:.1f}%  SD={sd:.1f}%"
                      f"  range=[{mn_r:.0f}%–{mx_r:.0f}%]"
                      f"  ({len(fold_recs)} folds with chatter)")
                for held, nc, tp, rec in fold_recs:
                    print(f"    P{held:2d}: {tp}/{nc} chatter detected"
                          f"  recall={rec:.1f}%")

        validation_runs[val_mode] = (pred_dict, prob_dict, metrics)

        # FORWARD and FIXED evaluate only a subset of passes; metrics are
        # computed over predicted segments only.  Note the coverage.
        if val_mode in ('FORWARD', 'FIXED'):
            n_pred = int(np.sum(~np.isnan(pred_dict['RF'])))
            note = ("early passes with insufficient history skipped"
                    if val_mode == 'FORWARD'
                    else "only the held-out test block is scored")
            print(f"  ({val_mode} metrics computed over {n_pred}/{len(LA)} "
                  f"segments; {note}.)")

        if not metrics:
            print(f"  No valid {val_mode} metrics — skipping figures for this scheme.")
            continue

        # ---- Section 6: predicted force maps (Fig 16) ----
        # Layout: ACTUAL spans the full top row, then the four models in a
        # 2x2 block below (RF, LR / NN, SVM).  No figure-level title.
        print(f"\n=== SECTION 6: {val_mode} PREDICTED FORCE MAPS ===")
        from matplotlib.gridspec import GridSpec
        # Fixed model order for the 2x2 block (rows: RF,LR then NN,SVM).
        map_order = [k for k in ["RF", "LR", "NN", "SVM"] if k in metrics]
        fig3 = plt.figure(figsize=(16, 18))
        # 3 rows (ACTUAL, then 2 model rows) x 2 cols, + slim colorbar column.
        gs3  = GridSpec(3, 3, figure=fig3,
                        width_ratios=[10, 10, 0.5],
                        height_ratios=[1, 1, 1],
                        left=0.06, right=0.93, top=0.97, bottom=0.05,
                        hspace=0.28, wspace=0.12)
        ax_actual = fig3.add_subplot(gs3[0, 0:2])       # ACTUAL spans both cols
        draw_force_map(ax_actual, ratio_mat, ds["labels"], numPasses, numSegments,
                       'ACTUAL\n("C" = confirmed chatter)',
                       max_ratio, rmsDisplayThreshold, pass_labels=pass_numbers)
        im_p = None
        cells = [(1, 0), (1, 1), (2, 0), (2, 1)]        # RF,LR / NN,SVM
        for (r, c), k in zip(cells, map_order):
            m  = metrics[k]
            ax = fig3.add_subplot(gs3[r, c])
            mk = pred_dict[k].reshape(numPasses, numSegments)
            im_p = draw_force_map(
                ax, ratio_mat, mk, numPasses, numSegments,
                f'{MODEL_NAMES[k]} ({tag})\nAcc = {m["acc"]*100:.0f}%  F1 = {m["f1"]*100:.0f}%',
                max_ratio, rmsDisplayThreshold, pass_labels=pass_numbers,
                show_ylabel=(c == 0))          # only left column gets y-axis label
        # One shared colorbar spanning the two model rows.
        cax3 = fig3.add_subplot(gs3[1:3, 2])
        cb = fig3.colorbar(im_p, cax=cax3,
                           ticks=[1.0, rmsDisplayThreshold, max_ratio])
        cb.ax.set_yticklabels(["1.0", f"{rmsDisplayThreshold:.1f}x",
                               f"{max_ratio:.1f}x"])
        save_pdf(fig3, f"Fig2_Predicted_Maps_{tag}.pdf", OUTPUT_DIR)

        # ---- Section 7: confusion matrices (Fig 17) — 2x2 with SVM ----
        print(f"\n=== SECTION 7: {val_mode} CONFUSION MATRICES ===")
        cm_order = [k for k in ["RF", "LR", "NN", "SVM"] if k in metrics]
        fig4, axc = plt.subplots(2, 2, figsize=(11, 10))
        axc = axc.flatten()
        for ax in axc[len(cm_order):]:
            ax.axis('off')                              # hide unused panels
        for i, k in enumerate(cm_order):
            m  = metrics[k]
            C  = m["C"]
            Cp = 100 * C / (C.sum(axis=1, keepdims=True) + EPS)
            im_cm = axc[i].imshow(Cp, cmap=CM_CMAP, vmin=0, vmax=100)
            lbl = [["TN","FP"],["FN","TP"]]
            for r in range(2):
                for cc in range(2):
                    clr = 'w' if Cp[r,cc] >= 55 else 'k'
                    axc[i].text(cc, r,
                                f"{lbl[r][cc]}\n{Cp[r,cc]:.0f}%\n({C[r,cc]})",
                                ha='center', va='center',
                                fontweight='bold', fontsize=INSIDE_FONT, color=clr)
            axc[i].set_xticks([0,1]); axc[i].set_xticklabels(["Pred Stable","Pred Chatter"])
            axc[i].set_yticks([0,1]); axc[i].set_yticklabels(["True Stable","True Chatter"])
            axc[i].grid(False)                          # no grid over the matrix
            axc[i].set_title(f"{MODEL_NAMES[k]}\n"
                             f"Acc={m['acc']*100:.0f}%  F1={m['f1']*100:.0f}%")
            fig4.colorbar(im_cm, ax=axc[i], ticks=[0,50,100], fraction=0.046, pad=0.04)
        fig4.tight_layout()
        save_pdf(fig4, f"Fig3_Confusion_Matrices_{tag}.pdf", OUTPUT_DIR)

        # ---- Section 8: safety analysis (Fig 4) ----
        print(f"\n=== SECTION 8: {val_mode} SAFETY ANALYSIS ===")
        ordered = sorted(metrics.items(), key=lambda x: x[1]["rec"], reverse=True)
        rec_vals  = np.array([m["rec"]  for _,m in ordered])*100
        prec_vals = np.array([m["prec"] for _,m in ordered])*100
        mdl_lbls  = [MODEL_NAMES[k] for k,_ in ordered]
        clrs5 = plt.cm.tab10(np.linspace(0, 0.5, len(ordered)))
        fig5, ax5 = plt.subplots(1, 2, figsize=(14, 6))
        n_mdl = len(ordered)
        ax5[0].barh(range(n_mdl), rec_vals, color=clrs5)
        ax5[0].axvline(90, color='k', ls='--', lw=1.5, label="90% target")
        ax5[0].set_yticks(range(n_mdl)); ax5[0].set_yticklabels(mdl_lbls)
        ax5[0].set_xlabel("Recall (%) — chatters correctly caught")
        ax5[0].set_title(f"RECALL (safety-critical) — {tag}\nMissed chatter = part scrapped",
                         fontweight='bold')
        ax5[0].set_xlim(0, 108); ax5[0].legend(); ax5[0].grid(True, axis='x')
        for i, v in enumerate(rec_vals):
            ax5[0].text(v+0.5, i, f"{v:.0f}%", va='center', fontweight='bold')
        for i, lbl in enumerate(mdl_lbls):
            ax5[1].scatter(prec_vals[i], rec_vals[i], s=220,
                           color=clrs5[i], edgecolors='k', zorder=3, label=lbl)
        ax5[1].scatter(100, 100, s=280, marker='*', c='gold',
                       edgecolors='k', label='Perfect', zorder=4)
        pg = np.arange(1, 100, 0.5)
        for f1_iso in [70, 80, 90]:
            rg = (f1_iso*pg)/(2*pg-f1_iso); rg[(rg<0)|(rg>100)] = np.nan
            ax5[1].plot(pg, rg, color=[0.7,0.7,0.7], ls='--', lw=0.8)
            if np.any(~np.isnan(rg)):
                ax5[1].text(pg[~np.isnan(rg)][-1]+0.5, rg[~np.isnan(rg)][-1],
                            f"F1={f1_iso}", fontsize=9, color='grey')
        ax5[1].set_xlabel("Precision (%)"); ax5[1].set_ylabel("Recall (%)")
        ax5[1].set_title("Precision vs Recall\nUpper-right = best", fontweight='bold')
        ax5[1].set_xlim(0, 110); ax5[1].set_ylim(0, 108); ax5[1].grid(True)
        ax5[1].legend(loc='lower left')
        # (no figure-level title — per style rules)
        fig5.tight_layout()
        save_pdf(fig5, f"Fig4_Safety_Analysis_{tag}.pdf", OUTPUT_DIR)

    # Expose LOPO results to the remaining sections (diagnostics, summary,
    # CSV export) which are scheme-independent and use the LOPO fit by default.
    pred_dict, prob_dict, metrics = validation_runs['LOPO']

    # =========================================================================
    #  SECTION 9 — FINAL-MODEL DIAGNOSTICS  (Fig 6)
    #  Trained on ALL data (inspection only — not the LOPO score).
    #  Shows: NN train/val loss  |  RF OOB error
    #         SVM learning curve |  LR learning curve
    #         Feature importance for each model
    # =========================================================================
    print("\n=== SECTION 9: FINAL-MODEL DIAGNOSTICS ===")

    # Normalise full datasets
    mu5  = FA5.mean(0); sig5_g = FA5.std(0, ddof=1); sig5_g[sig5_g==0]=1
    X_all5 = fix_nan((FA5-mu5)/sig5_g)
    mu_nn  = FA_nn.mean(0); sig_nn_g = FA_nn.std(0,ddof=1); sig_nn_g[sig_nn_g==0]=1
    X_all_nn = fix_nan((FA_nn-mu_nn)/sig_nn_g)

    rng_full = np.random.default_rng(SEED)

    # Split full NN data: train/val BEFORE augmentation
    nv    = max(1, int(0.15*len(LA)))
    perm  = np.random.permutation(len(LA))
    vi, ti = perm[:nv], perm[nv:]
    Xv_real = X_all_nn[vi]; yv_real = LA[vi]
    Xa_nn_f, ya_f = augment_minority(X_all_nn[ti], LA[ti], rng_full,
                                     imbal_thresh, jitter_sigma)
    rng_full2 = np.random.default_rng(SEED+9999)
    Xa5_f, ya5_f = augment_minority(X_all5, LA, rng_full2, imbal_thresh, jitter_sigma)

    # ---- Neural Network (full model) ----
    torch.manual_seed(SEED)
    net_f = PatternNet(n_input=nFeat_nn).to(device)
    Xt_f  = torch.tensor(Xa_nn_f, dtype=torch.float32, device=device)
    yt_f  = torch.tensor(ya_f,    dtype=torch.float32, device=device)
    Xv_t  = torch.tensor(Xv_real, dtype=torch.float32, device=device)
    yv_t  = torch.tensor(yv_real, dtype=torch.float32, device=device)
    opt_f = torch.optim.Adam(net_f.parameters(), lr=nn_lr_full)
    crit_f= nn.BCELoss()
    tr_loss, val_loss = [], []
    best_v=float('inf'); best_ep=0; bad=0
    best_wts = {k: v.clone() for k,v in net_f.state_dict().items()}
    for ep in range(nn_epochs_full):
        net_f.train(); opt_f.zero_grad()
        l = crit_f(net_f(Xt_f), yt_f); l.backward(); opt_f.step()
        net_f.eval()
        with torch.no_grad():
            vl = crit_f(net_f(Xv_t), yv_t).item()
        tr_loss.append(l.item()); val_loss.append(vl)
        if vl < best_v - 1e-5:
            best_v=vl; best_ep=ep+1; bad=0
            best_wts = {k: v.clone() for k,v in net_f.state_dict().items()}
        else:
            bad += 1
            if bad >= nn_patience:
                break
    net_f.load_state_dict(best_wts)
    nn_imp = np.abs(net_f.net[0].weight.detach().cpu().numpy()).sum(axis=0)
    print(f"  NN full: stopped ep {len(tr_loss)}, best val ep {best_ep}")

    # ---- RF (full model) with OOB curve ----
    rf_f = RandomForestClassifier(n_estimators=1, oob_score=True, bootstrap=True,
                                   max_features=rfMaxFeatures, min_samples_leaf=2,
                                   warm_start=True, random_state=SEED)
    oob_curve = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(1, nTrees+1):
            rf_f.set_params(n_estimators=k)
            rf_f.fit(Xa5_f, ya5_f)
            oob_curve.append(1-rf_f.oob_score_ if hasattr(rf_f,'oob_score_') else np.nan)
    rf_imp = permutation_importance(rf_f, X_all5, LA, n_repeats=10,
                                    random_state=SEED).importances_mean
    print(f"  RF full: OOB {oob_curve[-1]:.3f}")

    # ---- SVM and LR: learning curves (sklearn) ----
    # Compute train/val F1 vs training set size using stratified k-fold.
    # Uses the FULL original (un-augmented) data so the curves are honest.
    lc_sizes = np.linspace(0.2, 1.0, 8)
    svm_f = SVC(kernel='rbf', C=svm_C, gamma=svm_gamma, probability=True, random_state=SEED)
    lr_f  = LogisticRegression(max_iter=lr_max_iter, random_state=SEED)

    print("  Computing SVM learning curve...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_lc_sz, svm_tr, svm_vl = learning_curve(
            svm_f, X_all5, LA, cv=5, scoring='f1',
            train_sizes=lc_sizes, n_jobs=-1)
    print("  Computing LR learning curve...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_lc_sz, lr_tr, lr_vl = learning_curve(
            lr_f, X_all5, LA, cv=5, scoring='f1',
            train_sizes=lc_sizes, n_jobs=-1)

    # Fit final full models for importance
    svm_f.fit(Xa5_f, ya5_f)
    lr_f.fit(Xa5_f, ya5_f)
    svm_imp = permutation_importance(svm_f, X_all5, LA, n_repeats=10,
                                     random_state=SEED).importances_mean
    lr_imp  = np.abs(lr_f.coef_[0])

    # ---- Build Fig 6 (2 rows × 4 cols) ----
    fig6, ax6 = plt.subplots(2, 4, figsize=(22, 10))

    # Row 0: training / convergence curves
    ep = np.arange(1, len(tr_loss)+1)
    ax6[0,0].plot(ep, tr_loss, 'b-', lw=1.8, label="Train")
    ax6[0,0].plot(ep, val_loss,'r--', lw=1.8, label="Val")
    ax6[0,0].axvline(best_ep, color='k', ls=':', lw=1.2)
    ax6[0,0].set_xlabel("Epoch"); ax6[0,0].set_ylabel("BCE Loss")
    ax6[0,0].set_title("NN — Train / Val Loss", fontweight='bold')
    ax6[0,0].legend(); ax6[0,0].grid(True)

    ax6[0,1].plot(range(1, nTrees+1), oob_curve, 'g-', lw=1.8)
    ax6[0,1].set_xlabel("# Trees"); ax6[0,1].set_ylabel("OOB Error")
    ax6[0,1].set_title("RF — OOB Error", fontweight='bold')
    ax6[0,1].grid(True)

    for ax_lc, sz, tr_sc, vl_sc, name, clr in [
            (ax6[0,2], svm_lc_sz, svm_tr, svm_vl, "SVM", 'brown'),
            (ax6[0,3], lr_lc_sz,  lr_tr,  lr_vl,  "LR",  'green')]:
        ax_lc.plot(sz, tr_sc.mean(1), '-',  color=clr, lw=1.8, label="Train F1")
        ax_lc.fill_between(sz, tr_sc.mean(1)-tr_sc.std(1),
                            tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color=clr)
        ax_lc.plot(sz, vl_sc.mean(1), '--', color=clr, lw=1.8, label="Val F1 (5-fold)")
        ax_lc.fill_between(sz, vl_sc.mean(1)-vl_sc.std(1),
                            vl_sc.mean(1)+vl_sc.std(1), alpha=0.15, color=clr)
        ax_lc.set_xlabel("Training set size"); ax_lc.set_ylabel("F1 Score")
        ax_lc.set_title(f"{name} — Learning Curve", fontweight='bold')
        ax_lc.set_ylim(0, 1.05); ax_lc.legend(); ax_lc.grid(True)

    # Row 1: feature importances
    imp_data = [("NN", nn_imp, featureNames_nn, _clrs_for(featureNames_nn)),
                ("RF", rf_imp, featureNames_5,  _clrs_for(featureNames_5)),
                ("SVM", svm_imp, featureNames_5, _clrs_for(featureNames_5)),
                ("LR",  lr_imp,  featureNames_5, _clrs_for(featureNames_5))]

    for col, (key, imp, fnames, fclrs) in enumerate(imp_data):
        o  = np.argsort(imp)
        ax6[1,col].barh(range(len(fnames)), imp[o], color=fclrs[o])
        ax6[1,col].set_yticks(range(len(fnames)))
        # Clean display names — no underscores
        ax6[1,col].set_yticklabels([_display_name(fnames[i]) for i in o],
                                    fontsize=INSIDE_FONT)
        ax6[1,col].set_title(f"{MODEL_NAMES[key]}\nFeature Importance",
                              fontweight='bold')
        ax6[1,col].grid(True, axis='x')

    # (no figure-level title — per style rules; the "inspection only" caveat
    #  belongs in the figure caption, not on the figure itself.)
    fig6.tight_layout()
    save_pdf(fig6, "Fig5_Diagnostics.pdf", OUTPUT_DIR)

    # ---- Separate per-model figures (Figs 12–15, one file per model) ----
    # Each is a 1×2 panel: training curve (left) + feature importance (right).
    # All variables are captured explicitly here — no reliance on outer-scope
    # names that might not exist in all code paths.
    _diag_items = [
        {
            "key":    "NN",
            "fignum": 12,
            "kind":   "loss",         # loss curve
            "tr":     list(tr_loss),
            "vl":     list(val_loss),
            "best":   int(best_ep),
            "imp":    nn_imp,
            "fnames": list(featureNames_nn),
        },
        {
            "key":    "RF",
            "fignum": 13,
            "kind":   "oob",          # OOB error curve
            "oob":    list(oob_curve),
            "nTrees": int(nTrees),
            "imp":    rf_imp,
            "fnames": list(featureNames_5),
        },
        {
            "key":    "SVM",
            "fignum": 14,
            "kind":   "lc",           # learning curve (2-D arrays)
            "sz":     svm_lc_sz,
            "tr":     svm_tr,
            "vl":     svm_vl,
            "imp":    svm_imp,
            "fnames": list(featureNames_5),
        },
        {
            "key":    "LR",
            "fignum": 15,
            "kind":   "lc",
            "sz":     lr_lc_sz,
            "tr":     lr_tr,
            "vl":     lr_vl,
            "imp":    lr_imp,
            "fnames": list(featureNames_5),
        },
    ]

    for _d in _diag_items:
        _key   = _d["key"]
        _fnum  = _d["fignum"]
        _imp   = _d["imp"]
        _fnames= _d["fnames"]

        _fig, (_aL, _aR) = plt.subplots(1, 2, figsize=(12, 5.5))
        _fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.13,
                             wspace=0.38)

        # --- Left panel: training diagnostic ---
        if _d["kind"] == "loss":
            _ep_ax = range(1, len(_d["tr"]) + 1)
            _aL.plot(_ep_ax, _d["tr"], color='#1f6eb5', lw=1.8, label="Train")
            _aL.plot(_ep_ax, _d["vl"], color='#c0392b', lw=1.8, ls='--', label="Val")
            _aL.axvline(_d["best"], color='k', ls=':', lw=1.2,
                        label=f"Best val ep {_d['best']}")
            _aL.set_xlabel("Epoch")
            _aL.set_ylabel("BCE Loss")
            _aL.set_title("NN \u2014 Train / Val Loss")
            _aL.legend(); _aL.grid(True)

        elif _d["kind"] == "oob":
            _xs = range(1, _d["nTrees"] + 1)
            _aL.plot(_xs, _d["oob"], color='#27ae60', lw=1.8)
            _aL.set_xlabel("Number of Trees")
            _aL.set_ylabel("OOB Error")
            _aL.set_title("RF \u2014 OOB Error")
            _aL.grid(True)

        elif _d["kind"] == "lc":
            _sz = _d["sz"]
            _tr_2d = _d["tr"]       # shape: (n_sizes, n_cv_folds)
            _vl_2d = _d["vl"]
            _tr_mean = _tr_2d.mean(axis=1)
            _tr_std  = _tr_2d.std(axis=1)
            _vl_mean = _vl_2d.mean(axis=1)
            _vl_std  = _vl_2d.std(axis=1)
            _clr = '#6e2fa8' if _key == 'SVM' else '#1e8449'
            _aL.plot(_sz, _tr_mean, color=_clr,     lw=1.8, label="Train")
            _aL.plot(_sz, _vl_mean, color='#c0392b', lw=1.8, ls='--',
                     label="Val (5-fold)")
            _aL.fill_between(_sz, _tr_mean - _tr_std, _tr_mean + _tr_std,
                             alpha=0.15, color=_clr)
            _aL.fill_between(_sz, _vl_mean - _vl_std, _vl_mean + _vl_std,
                             alpha=0.15, color='#c0392b')
            _aL.set_xlabel("Training samples")
            _aL.set_ylabel("F1 Score")
            _aL.set_title(f"{MODEL_NAMES[_key]} \u2014 Learning Curve")
            _aL.set_ylim(0, 1.05)
            _aL.legend(); _aL.grid(True)

        # --- Right panel: feature importance (clean labels, sorted) ---
        _o    = np.argsort(_imp)
        _clrs = _clrs_for(_fnames)
        _aR.barh(range(len(_fnames)), _imp[_o], color=_clrs[_o])
        _aR.set_yticks(range(len(_fnames)))
        _aR.set_yticklabels([_display_name(_fnames[i]) for i in _o],
                             fontsize=INSIDE_FONT)
        _aR.set_title(f"{MODEL_NAMES[_key]}\nFeature Importance")
        _aR.set_xlabel("Importance")
        _aR.grid(True, axis='x')

        _fig.tight_layout()
        _safe = (MODEL_NAMES[_key]
                 .replace(' ', '_').replace('(', '').replace(')', ''))
        save_pdf(_fig, f"Fig{_fnum}_{_safe}_Diagnostics.pdf", OUTPUT_DIR)
        plt.close(_fig)
        print(f"  Saved: Fig{_fnum}_{_safe}_Diagnostics.pdf")

    # =========================================================================
    #  SECTION 10 — SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*75)
    print("  FINAL RESULTS — Leave-One-Pass-Out cross-validation")
    print("="*75)
    print(f"{'Model':<22} | Acc    | Prec   | Recall | F1     | MSE    | R2")
    print("-"*22 + "-|--------"*6)
    for k in sorted(metrics, key=lambda x: metrics[x]["f1"], reverse=True):
        m = metrics[k]
        print(f"{MODEL_NAMES[k]:<22} | {m['acc']*100:5.1f}% | "
              f"{m['prec']*100:5.1f}% | {m['rec']*100:5.1f}% | "
              f"{m['f1']*100:5.1f}% | {m['mse']:.4f} | {m['r2']:.4f}")
    print("="*75)
    print("  Recall = safety metric (missed chatter = part scrapped).")
    print("  R2/MSE on 0/1 labels: reported for continuity, not standard metrics.")
    print(f"\n  All figures saved as PDF to: {OUTPUT_DIR}")

    # =========================================================================
    #  SECTION 11 — RESULT EXPORT
    #  One row per (mode, validation-scheme, model).  Deduplication prevents
    #  double-writing when the same mode is run more than once.
    # =========================================================================
    import csv as _csv

    ablation_csv = os.path.join(os.path.dirname(OUTPUT_DIR),
                                "ablation_results.csv")

    # Load existing rows to check for duplicates before writing.
    existing_keys = set()
    header = ["Mode", "Validation", "Features", "Model",
              "TN", "FP", "FN", "TP",
              "Accuracy_%", "Precision_%", "Recall_%", "F1_%",
              "N_predicted"]
    if os.path.exists(ablation_csv):
        with open(ablation_csv, newline="") as _f:
            for row in _csv.DictReader(_f):
                existing_keys.add((row["Mode"], row["Validation"], row["Model"]))

    new_rows = []
    for vmode in ['LOPO', 'FORWARD', 'FIXED']:
        if vmode not in validation_runs:
            continue
        vp, _, vmetrics = validation_runs[vmode]
        for k in ["RF", "LR", "NN", "SVM"]:
            if k not in vmetrics:
                continue
            key = (ABLATION_MODE, vmode, MODEL_NAMES[k])
            if key in existing_keys:
                print(f"  [skip duplicate] Mode={ABLATION_MODE} "
                      f"Validation={vmode} Model={MODEL_NAMES[k]}")
                continue
            m   = vmetrics[k]
            yp  = vp[k]
            ok  = ~np.isnan(yp)
            yt_, yp_ = LA[ok], yp[ok]
            TN = int(np.sum((yt_ == 0) & (yp_ == 0)))
            FP = int(np.sum((yt_ == 0) & (yp_ == 1)))
            FN = int(np.sum((yt_ == 1) & (yp_ == 0)))
            TP = int(np.sum((yt_ == 1) & (yp_ == 1)))
            feats = featureNames_nn if k == "NN" else featureNames_5
            new_rows.append([ABLATION_MODE, vmode, "+".join(feats),
                             MODEL_NAMES[k], TN, FP, FN, TP,
                             f"{m['acc']*100:.1f}", f"{m['prec']*100:.1f}",
                             f"{m['rec']*100:.1f}", f"{m['f1']*100:.1f}",
                             int(ok.sum())])

    if new_rows:
        write_header = not os.path.exists(ablation_csv)
        with open(ablation_csv, "a", newline="") as fcsv:
            w = _csv.writer(fcsv)
            if write_header:
                w.writerow(header)
            w.writerows(new_rows)
        print(f"  {len(new_rows)} new row(s) written to: {ablation_csv}")
    else:
        print(f"  No new rows — all already in: {ablation_csv}")
        print(f"  (Delete the CSV to force a full re-write.)")

    plt.show()


if __name__ == "__main__":
    main()
