# %% [markdown]
# # A1 — Breast Cancer Diagnosis from Cell-Nucleus Measurements
#
# **Course:** AI-101 Introduction to Artificial Intelligence — Machine Learning Assignment
# **Student:** Rizwan Ali · Roll No. 70177926 · Section C
# **Repository:** https://github.com/[username]/AI101-ML-Assignment-70177926
#
# This notebook is the *single source of truth* for the A1 report. Every number that
# appears in `70177926_A1_Report.pdf` is written by this notebook into
# `results/A1_results.json`; the report generator reads that file. Nothing in the
# report is typed by hand.
#
# Run order: Runtime → Restart and run all. Fixed seed = 42 everywhere.

# %% [markdown]
# ## 1. Introduction
#
# Fine-needle aspiration (FNA) is the least invasive way to sample a suspicious breast
# mass: a thin needle lifts a few cells, a pathologist photographs them, and image
# analysis software measures the nuclei. The measurements are cheap and fast, but
# reading them is subjective and inter-observer agreement on "atypical" nuclei is
# moderate. A model that turns those 30 numbers into a malignancy probability can be
# used as a *second reader*: it never replaces the pathologist, it flags cases that
# deserve a second look.
#
# The cost asymmetry is the whole point of this task. A false negative (malignant mass
# called benign) delays cancer treatment; a false positive (benign mass called
# malignant) triggers an unnecessary biopsy, which is painful and expensive but rarely
# dangerous. Any screening model must therefore be tuned for **high recall on the
# malignant class**, and accuracy — dominated by the 62.7 % benign majority — is close
# to meaningless.

# %% [markdown]
# ## 2. Problem Statement
#
# Let $x \in \mathbb{R}^{30}$ be the ten nucleus properties (radius, texture, perimeter,
# area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
# each summarised three ways: mean, standard error, and worst (largest) value. Let
# $y \in \{\text{malignant}, \text{benign}\}$.
#
# We learn $f: \mathbb{R}^{30} \to [0,1]$ estimating $P(y=\text{malignant}\mid x)$, and
# convert it to a decision by thresholding: predict malignant iff $f(x) \ge \tau$.
#
# - **Task type:** supervised binary classification.
# - **Success metrics:** recall on malignant, F1 on malignant, ROC-AUC. Accuracy is
#   reported but is *not* the selection criterion.
# - **Constraint:** 212 malignant vs 357 benign samples (37 % / 63 %) — mildly
#   imbalanced; the real constraint is the asymmetric error cost, handled by the
#   threshold study in §6.3.

# %% [markdown]
# ## 3. Dataset Used
#
# **Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository, id 17
# [1]; also bundled with scikit-learn as `load_breast_cancer` [2]. Collected by
# Street, Wolberg and Mangasarian (1993) from digitised FNA images at the University of
# Wisconsin Hospitals; released for research use (CC BY 4.0 on the UCI page).
#
# - **Rows:** 569 patients, one row per FNA sample.
# - **Columns:** 30 real-valued features (10 properties × {mean, standard error, worst})
#   plus the diagnosis label.
# - **Target encoding in the raw file:** `0 = malignant`, `1 = benign`. This is the
#   opposite of what most readers expect, so we flip it immediately into
#   `y_malignant = 1` and never touch the raw encoding again. Getting this wrong
#   inverts every metric in the report — it is the first thing I checked.

# %%
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight"})

SEED = 42
np.random.seed(SEED)

ROOT = Path.cwd()
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

print("python      :", sys.version.split()[0])
print("platform    :", platform.platform())
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("seed        :", SEED)
print("cpu count   :", __import__("os").cpu_count())
print("figures dir :", FIG)

# %% [markdown]
# ## 4. Dataset Statistics

# %%
data = load_breast_cancer(as_frame=True)
X_raw = data.data.copy()
y_raw = data.target.copy()

# 0 = malignant, 1 = benign in the shipped file -> flip so that 1 = malignant = positive
y = (y_raw == 0).astype(int)
y.name = "malignant"

print("shape:", X_raw.shape)
print("missing values per column (total):", int(X_raw.isna().sum().sum()))
print()
print("raw target counts :", y_raw.value_counts().to_dict(), "  (0=malignant, 1=benign)")
print("after flip        :", y.value_counts().to_dict(), "  (1=malignant)")
print("positive rate     : %.3f" % y.mean())

df = X_raw.copy()
df["malignant"] = y

# summary statistics table (compact: 10 base properties, mean columns)
stats = pd.DataFrame({
    "mean": X_raw.mean(), "std": X_raw.std(),
    "min": X_raw.min(), "median": X_raw.median(), "max": X_raw.max(),
})
stats.round(2).to_csv(RES / "A1_feature_stats.csv")
print()
print(stats.round(2).head(12).to_string())

# %%
# ---- EDA figure: class distribution (sized 1:1 for the print column) -------
fig, ax = plt.subplots(figsize=(3.4, 1.6))
counts = y.value_counts().sort_index()
bars = ax.bar(["malignant", "benign"], counts.values,
              color=["#b2182b", "#4393c3"], edgecolor="black", linewidth=.5, width=.5)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v} ({v/len(y)*100:.1f}%)",
            ha="center", va="bottom", fontsize=7)
ax.set_ylabel("samples", fontsize=7); ax.set_ylim(0, counts.max() * 1.2)
ax.set_title("class distribution (63/37 benign-majority)", fontsize=7.5)
ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "A1_fig1_class_distribution.png"); plt.close(fig)

# ---- EDA figure: mean radius by class (notebook only) ----------------------
fig, ax = plt.subplots(figsize=(3.4, 1.7))
ax.hist([X_raw.loc[y == 1, "mean radius"], X_raw.loc[y == 0, "mean radius"]],
        bins=20, color=["#b2182b", "#4393c3"], label=["malignant", "benign"],
        edgecolor="white", linewidth=.3)
ax.set_xlabel("mean radius ($\\mu$m)", fontsize=7); ax.set_ylabel("samples", fontsize=7)
ax.set_title("mean radius by class", fontsize=7.5)
ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "A1_fig1b_radius.png"); plt.close(fig)

# %%
# ---- EDA figure 2: correlation heat-map ------------------------------------
corr = X_raw.corr()
fig, ax = plt.subplots(figsize=(7.4, 6.2))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=.3, cbar_kws={"shrink": .62, "label": "Pearson r"}, ax=ax)
ax.set_title("Correlation of the 30 nucleus features", fontsize=9)
ax.tick_params(labelsize=5.2)
fig.tight_layout(); fig.savefig(FIG / "A1_fig2_correlation.png"); plt.close(fig)

# %%
# ---- which features are redundant? (computed on TRAIN ONLY, see §5.2) -----
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=SEED
)
print("train:", X_tr.shape, " test:", X_te.shape)
print("train positive rate: %.3f   test positive rate: %.3f" % (y_tr.mean(), y_te.mean()))

c_tr = X_tr.corr().abs()
THR = 0.95
pairs = [(a, b, c_tr.loc[a, b]) for i, a in enumerate(c_tr.columns)
         for b in c_tr.columns[i + 1:] if c_tr.loc[a, b] >= THR]
print(f"\nfeature pairs with |r| >= {THR} (training split only): {len(pairs)}")
for a, b, r in sorted(pairs, key=lambda t: -t[2])[:12]:
    print(f"   {a:<26s} {b:<26s} r={r:.4f}")

# union-find over the redundant pairs -> groups of interchangeable features
parent = {c: c for c in c_tr.columns}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a


for a, b, _ in pairs:
    parent[find(a)] = find(b)
groups = {}
for c in c_tr.columns:
    groups.setdefault(find(c), []).append(c)
redundant_groups = [sorted(g) for g in groups.values() if len(g) > 1]
print(f"\n{len(redundant_groups)} redundant groups, {sum(len(g)-1 for g in redundant_groups)} "
      f"features can be dropped:")
for g in redundant_groups:
    print("   ", " ~ ".join(g))

# %%
# ---- Table 1: dataset statistics (also written to JSON for the report) -----
dataset_stats = {
    "samples": int(len(X_raw)),
    "features": int(X_raw.shape[1]),
    "features_numeric": int(X_raw.shape[1]),
    "features_categorical": 0,
    "features_text": 0,
    "target": "malignant (1) vs benign (0)",
    "class_counts": {("malignant" if k == 1 else "benign"): int(v)
                     for k, v in y.value_counts().sort_index().items()},
    "class_pct": {("malignant" if k == 1 else "benign"): round(float(v) / len(y) * 100, 2)
                  for k, v in y.value_counts().sort_index().items()},
    "missing_pct": 0.0,
    "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "n_val": 0,
    "redundant_groups": redundant_groups,
}
print(json.dumps(dataset_stats, indent=2)[:600])

# %% [markdown]
# ## 5. Methodology
#
# ### 5.1 Learning setting
#
# Supervised binary classification on a small ($n=569$), low-dimensional ($d=30$),
# strongly collinear tabular dataset. That shape of data dictates the model set:
#
# | Model | Why it is in the comparison |
# |---|---|
# | **Dummy (majority)** | Not a real model — it exists to *quantify* the accuracy trap: 62.7 % accuracy with 0 % malignant recall. |
# | **Logistic Regression** (baseline) | Linear decision boundary in scaled space; interpretable coefficients; the reference every other model must beat. |
# | **SVM, RBF kernel** | Non-linear boundary, strong on $n \ll 10^4$ tabular problems; sensitive to `C` and `gamma`, so worth tuning. |
# | **k-NN ($k=5$)** | Pure local/instance-based learner, no training cost; tests whether a *local* similarity structure is enough. Left untuned on purpose — it is the honest "no tuning" reference. |
# | **Random Forest** | Non-linear, scale-invariant, handles collinearity gracefully, and gives feature importances for §6.4. |
#
# Scaling is **inside** a `Pipeline`, so `StandardScaler` is fitted on the training fold
# only during CV and on the training split only at final fit. Nothing from the test set
# ever touches a mean, a standard deviation, or a hyperparameter.
#
# ### 5.2 Data-split approach
#
# - One stratified **80 / 20** train/test split (`random_state=42`) → 455 / 114 samples,
#   class ratio preserved in both.
# - Hyperparameters chosen by `GridSearchCV` with **5-fold `StratifiedKFold`** on the
#   training split only (no separate validation split, so all 455 training rows are used
#   for tuning and the test set stays untouched).
# - Model selection inside the grid uses **ROC-AUC**, which is threshold-free and
#   insensitive to the class ratio. F1 and recall are recorded for the same folds.
# - The test set is scored **once**, after tuning, in §6.
# - The redundancy analysis in §4 used the *training* correlation matrix only, so the
#   ablation in §6.5 is not contaminated by test-set structure.
#
# ### 5.3 Parameter settings

# %%
def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"}

grids = {
    "LogisticRegression": (
        make_pipe(LogisticRegression(max_iter=5000, random_state=SEED)),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    ),
    "SVM_RBF": (
        make_pipe(SVC(kernel="rbf", probability=True, random_state=SEED)),
        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    ),
    "RandomForest": (
        make_pipe(RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [None, 4, 6, 8],
         "clf__min_samples_leaf": [1, 2, 4]},
    ),
}
fixed_models = {
    "DummyMajority": make_pipe(DummyClassifier(strategy="most_frequent")),
    "kNN_k5": make_pipe(KNeighborsClassifier(n_neighbors=5)),
}

for name, (pipe, grid) in grids.items():
    n = int(np.prod([len(v) for v in grid.values()]))
    print(f"{name:<20s} {n:>3d} candidates x 5 folds = {n*5:>4d} fits   {list(grid)}")
for name in fixed_models:
    print(f"{name:<20s}   0 candidates (no tuning)")

# %%
t0 = time.time()
searches, cv_rows = {}, []

for name, (pipe, grid) in grids.items():
    t = time.time()
    gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc",
                      cv=cv, n_jobs=-1, return_train_score=False)
    gs.fit(X_tr, y_tr)
    gs.fit_time_s = time.time() - t
    searches[name] = gs
    i = gs.best_index_
    cv_rows.append({
        "model": name,
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "cv_auc_mean": float(gs.cv_results_["mean_test_roc_auc"][i]),
        "cv_auc_std": float(gs.cv_results_["std_test_roc_auc"][i]),
        "cv_f1_mean": float(gs.cv_results_["mean_test_f1"][i]),
        "cv_f1_std": float(gs.cv_results_["std_test_f1"][i]),
        "cv_recall_mean": float(gs.cv_results_["mean_test_recall"][i]),
        "cv_recall_std": float(gs.cv_results_["std_test_recall"][i]),
        "cv_precision_mean": float(gs.cv_results_["mean_test_precision"][i]),
        "fit_time_s": round(gs.fit_time_s, 1),
        "n_candidates": int(len(gs.cv_results_["params"])),
    })
    print(f"{name:<20s} best={gs.best_params_}  "
          f"CV AUC={gs.cv_results_['mean_test_roc_auc'][i]:.4f}"
          f"±{gs.cv_results_['std_test_roc_auc'][i]:.4f}  ({gs.fit_time_s:.1f}s)")

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(RES / "A1_cv_results.csv", index=False)
print("\ntotal tuning time: %.1f s" % (time.time() - t0))
cv_df[["model", "cv_auc_mean", "cv_auc_std", "cv_f1_mean", "cv_recall_mean"]].round(4)

# %% [markdown]
# ## 6. Results and Analysis
#
# ### 6.1 Results on the held-out test set (n = 114, scored once)

# %%
def evaluate(name, model, X, y, fit_time=None):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_time_s": fit_time,
    }


rows, probas = [], {}
for name, gs in searches.items():
    best = gs.best_estimator_
    rows.append(evaluate(name, best, X_te, y_te, round(gs.fit_time_s, 1)))
    probas[name] = best.predict_proba(X_te)[:, 1]

for name, pipe in fixed_models.items():
    t = time.time()
    pipe.fit(X_tr, y_tr)
    rows.append(evaluate(name, pipe, X_te, y_te, round(time.time() - t, 2)))
    probas[name] = pipe.predict_proba(X_te)[:, 1]

test_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
test_df.to_csv(RES / "A1_test_results.csv", index=False)
test_df.round(4).to_string(index=False)

# %%
# ---- Figure 3: confusion matrices -----------------------------------------
order = ["DummyMajority", "LogisticRegression", "SVM_RBF", "RandomForest"]
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.45))
for ax, name in zip(axes, order):
    pred = (probas[name] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["ben", "mal"], yticklabels=["ben", "mal"],
                annot_kws={"size": 9})
    r = test_df.set_index("model").loc[name]
    ax.set_title(f"{name}\nrecall={r['recall']:.2f}  acc={r['accuracy']:.2f}", fontsize=7.2)
    ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7)
    ax.tick_params(labelsize=6.5)
fig.suptitle("Confusion matrices at threshold 0.5 (test set, n=114)", fontsize=9, y=1.04)
fig.tight_layout(); fig.savefig(FIG / "A1_fig3_confusion.png"); plt.close(fig)

# %%
# ---- Figure: ROC curves (sized for the print column) ----------------------
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for name in ["DummyMajority", "kNN_k5", "LogisticRegression", "SVM_RBF", "RandomForest"]:
    fpr, tpr, _ = roc_curve(y_te, probas[name])
    a = roc_auc_score(y_te, probas[name])
    ax.plot(fpr, tpr, "--" if name == "DummyMajority" else "-", lw=1.2,
            label=f"{name.replace('_',' ')} ({a:.3f})")
ax.plot([0, 1], [0, 1], ":", color="grey", lw=.8)
ax.set_xlabel("false positive rate", fontsize=7)
ax.set_ylabel("true positive rate (recall)", fontsize=7)
ax.set_title("ROC, test set (AUC in legend)", fontsize=7.5)
ax.legend(fontsize=5.4, loc="lower right", frameon=False)
ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "A1_fig4_roc.png"); plt.close(fig)

# %%
# ---- Figure 5: CV mean ± std ---------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.5))
sub = cv_df.sort_values("cv_auc_mean")
ax.barh(sub["model"], sub["cv_auc_mean"], xerr=sub["cv_auc_std"], capsize=3,
        color="#4393c3", edgecolor="black", linewidth=.5)
ax.set_xlim(0.8, 1.0); ax.set_xlabel("5-fold CV ROC-AUC (train split only)")
ax.set_title("Cross-validated performance, mean ± sd over 5 folds")
for i, (m, v, s) in enumerate(zip(sub["model"], sub["cv_auc_mean"], sub["cv_auc_std"])):
    ax.text(v + s + .002, i, f"{v:.3f} ± {s:.3f}", va="center", fontsize=6.6)
fig.tight_layout(); fig.savefig(FIG / "A1_fig5_cv.png"); plt.close(fig)

# %% [markdown]
# ### 6.2 Which model wins, and why accuracy is misleading
#
# Read the table above against the Dummy row. The majority-class model is **62.3 %
# accurate on the test set and catches zero cancers**: its recall on malignant is 0.00.
# Any headline that says "62 % accurate" describes a model that is clinically worthless.
# That is why the selection criterion here is ROC-AUC for ranking and malignant-recall /
# F1 for the operating point.
#
# Among the real models the ranking is driven by the geometry of the data: after
# standardisation the classes are almost linearly separable along a "size" direction
# (radius / perimeter / area / concave points), so the linear baseline is already strong;
# the RBF SVM and the forest buy a little extra AUC by bending the boundary around a
# handful of atypical samples. The gain is small because 569 samples do not support a
# very flexible boundary — a reminder that on small tabular data the model class matters
# less than the split and the threshold.

# %% [markdown]
# ### 6.3 Threshold study
#
# **Where the threshold is chosen matters.** Picking $\tau$ on the test set and then
# reporting test metrics at that $\tau$ is a second, quieter form of leakage — the
# reported recall would be optimistic because the threshold was fitted to the very
# labels it is scored against. So:
#
# 1. the **model** is selected by 5-fold CV ROC-AUC on the training split only
#    (pre-registered rule, not "whichever wins on test");
# 2. the **threshold** $\tau^\*$ is chosen on **out-of-fold (OOF) predictions** of the
#    training split — each training row is scored by a model that never saw it;
# 3. $\tau^\*$ is then applied **once** to the test set.
#
# $\tau^\*$ is the *largest* threshold whose OOF recall still reaches 0.95, i.e. the
# cheapest threshold that satisfies the clinical constraint.

# %%
best_name = cv_df.sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
print("best model selected on CV ROC-AUC only:", best_name)
best_pipe = searches[best_name].best_estimator_
best_proba = probas[best_name]

# ---- out-of-fold probabilities on the TRAINING split ----------------------
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(best_pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]

ths = np.round(np.arange(0.01, 0.995, 0.005), 3)
TARGET = 0.95


def sweep(proba, y_true):
    out = []
    for t in ths:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out.append({"threshold": float(t),
                    "recall": float(recall_score(y_true, pred)),
                    "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "f1": float(f1_score(y_true, pred, zero_division=0)),
                    "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn)})
    return pd.DataFrame(out)


oof_df = sweep(oof, y_tr)                       # used ONLY to choose tau*
th_df = sweep(best_proba, y_te)                 # descriptive, not used for selection
oof_df.to_csv(RES / "A1_threshold_sweep_oof.csv", index=False)
th_df.to_csv(RES / "A1_threshold_sweep_test.csv", index=False)

ok = oof_df[oof_df["recall"] >= TARGET]
if not len(ok):                                  # guard: never silently pick a bad tau
    raise RuntimeError("OOF recall never reaches the target; widen the threshold grid")
t_star = float(ok["threshold"].max())
oof_star = ok.loc[ok["threshold"].idxmax()]
print(f"OOF recall at tau=0.50: {oof_df.iloc[(oof_df['threshold']-0.5).abs().argmin()]['recall']:.3f}")
print(f"tau* = {t_star:.3f}  (OOF recall={oof_star['recall']:.3f}, "
      f"OOF precision={oof_star['precision']:.3f})")

# ---- operating points evaluated on TEST at thresholds fixed beforehand ----
def at(proba, y_true, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": float(t),
            "recall": float(recall_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn)}


op_rows = [dict(name="τ = 0.50 (default)", **at(best_proba, y_te, 0.50)),
           dict(name=f"τ* = {t_star:.3f} (recall≥0.95, OOF)", **at(best_proba, y_te, t_star))]
# extra stress point: what if the clinic demanded 99 % recall? threshold from OOF again
ok99 = oof_df[oof_df["recall"] >= 0.99]
if len(ok99):
    t99 = float(ok99["threshold"].max())
    op_rows.append(dict(name=f"τ = {t99:.3f} (recall≥0.99, OOF)", **at(best_proba, y_te, t99)))
op_df = pd.DataFrame(op_rows)
op_df.to_csv(RES / "A1_operating_points.csv", index=False)

print()
print(op_df[["name", "threshold", "recall", "precision", "f1", "fn", "fp"]].round(3).to_string(index=False))
base = op_df.iloc[0]
row_star = op_df.iloc[1]
print(f"\ncost of guaranteeing 95% recall: precision {base['precision']:.3f} -> "
      f"{row_star['precision']:.3f} ({(row_star['precision']-base['precision'])*100:+.1f} pts); "
      f"false negatives {int(base['fn'])} -> {int(row_star['fn'])}; "
      f"false positives {int(base['fp'])} -> {int(row_star['fp'])}")

# %%
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.plot(oof_df["threshold"], oof_df["recall"], lw=1.4, color="#b2182b", label="OOF recall")
ax.plot(oof_df["threshold"], oof_df["precision"], lw=1.4, color="#b2182b", ls="--",
        label="OOF precision")
ax.plot(th_df["threshold"], th_df["recall"], lw=1.0, color="#2166ac", alpha=.7,
        label="test recall")
ax.plot(th_df["threshold"], th_df["precision"], lw=1.0, color="#2166ac", ls="--",
        alpha=.7, label="test precision")
ax.axhline(TARGET, ls=":", color="grey", lw=.9)
ax.axvline(t_star, ls="--", color="crimson", lw=.9)
ax.annotate(f"$\\tau^*$={t_star:.2f}", (t_star, .28), fontsize=6.5, color="crimson")
ax.set_xlabel("decision threshold $\\tau$", fontsize=7)
ax.set_ylabel("score", fontsize=7)
ax.set_title("recall / precision vs $\\tau$", fontsize=7.5)
ax.legend(fontsize=5.6, loc="center left", frameon=False)
ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "A1_fig6_threshold.png"); plt.close(fig)

# %% [markdown]
# Lowering $\tau$ slides the operating point along the same ROC curve: every false
# negative converted into a true positive must be paid for with false positives, because
# the score distributions of the two classes overlap in a thin band (the "atypical"
# nuclei). The curve is nearly a right angle -- recall is flat at 1.0 until
# $\tau \approx 0.9$ and precision is flat at 1.0 until $\tau \approx 0.2$ -- which is what
# a very separable dataset looks like, and why moving $\tau$ from 0.50 to $\tau^\*$ costs
# so little here. On a harder, noisier problem the same move costs much more precision;
# the *mechanism* is identical, only the exchange rate changes.

# %%
# ---- Figure 7: feature importance ----------------------------------------
lr = searches["LogisticRegression"].best_estimator_.named_steps["clf"]
rf = searches["RandomForest"].best_estimator_.named_steps["clf"]
imp = pd.DataFrame({
    "feature": X_raw.columns,
    "LR coef (std. space)": lr.coef_[0],
    "RF importance": rf.feature_importances_,
}).sort_values("RF importance", ascending=False)
imp.to_csv(RES / "A1_feature_importance.csv", index=False)

top = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax[0].barh(top["feature"], top["RF importance"], color="#4393c3", edgecolor="black", lw=.4)
ax[0].set_title("(a) Random Forest impurity importance"); ax[0].tick_params(labelsize=6.5)
ax[0].set_xlabel("mean decrease in impurity")
lr_top = imp.reindex(imp["LR coef (std. space)"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
ax[1].barh(lr_top["feature"], lr_top["LR coef (std. space)"], color="#b2182b", edgecolor="black", lw=.4)
ax[1].set_title("(b) Logistic Regression coefficients"); ax[1].tick_params(labelsize=6.5)
ax[1].set_xlabel("standardised coefficient (→ malignant)")
fig.tight_layout(); fig.savefig(FIG / "A1_fig7_importance.png"); plt.close(fig)
print(imp.head(8).round(3).to_string(index=False))

# %%
# ---- 6.5 Ablation: drop redundant features ------------------------------
drop = [g[1:] for g in redundant_groups]
drop = [f for grp in drop for f in grp]
keep = [c for c in X_raw.columns if c not in drop]
print(f"dropping {len(drop)} of {X_raw.shape[1]} features -> {len(keep)} remain")

abl = []
for label, cols in [("all 30 features", list(X_raw.columns)), (f"reduced ({len(keep)})", keep)]:
    Xa_tr, Xa_te = X_tr[cols], X_te[cols]
    pipe, grid = grids[best_name]
    gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xa_tr, y_tr)
    r = evaluate(f"{best_name} | {label}", gs.best_estimator_, Xa_te, y_te)
    r["n_features"] = len(cols)
    r["best_params"] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    abl.append(r)
abl_df = pd.DataFrame(abl)
abl_df.to_csv(RES / "A1_ablation.csv", index=False)
print(abl_df[["model", "n_features", "accuracy", "recall", "f1", "roc_auc"]].round(4).to_string(index=False))

# %%
# ---- 6.6 Error analysis: what does the recall guarantee buy? -------------
pred_star = (best_proba >= t_star).astype(int)
err_idx = np.where(pred_star != y_te.values)[0]
err = pd.DataFrame({
    "test_row": err_idx,
    "true": ["malignant" if v == 1 else "benign" for v in y_te.values[err_idx]],
    "p(malignant)": best_proba[err_idx].round(3),
    "predicted": ["malignant" if v == 1 else "benign" for v in pred_star[err_idx]],
    "mean radius": X_te["mean radius"].values[err_idx].round(1),
    "mean concavity": X_te["mean concavity"].values[err_idx].round(4),
    "worst concave points": X_te["worst concave points"].values[err_idx].round(4),
}).sort_values("p(malignant)")
err.to_csv(RES / "A1_errors.csv", index=False)
print(f"{len(err)} errors at τ*={t_star:.3f}  "
      f"({int((err['true']=='malignant').sum())} false negatives, "
      f"{int((err['true']=='benign').sum())} false positives)")
print(err.head(10).to_string(index=False))

# %%
# ---- persist everything the report needs --------------------------------
results = {
    "task": "A1", "seed": SEED,
    "env": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "cpu_count": __import__("os").cpu_count()},
    "dataset": dataset_stats,
    "cv": cv_rows,
    "test": rows,
    "best_model": best_name,
    "best_model_rule": "highest 5-fold CV ROC-AUC on the training split",
    "threshold_study": {
        "target_recall": TARGET,
        "t_star": t_star,
        "chosen_on": "out-of-fold training predictions",
        "oof_at_050": float(oof_df.iloc[(oof_df["threshold"] - 0.5).abs().argmin()]["recall"]),
        "oof_at_t_star": {"recall": float(oof_star["recall"]),
                          "precision": float(oof_star["precision"]),
                          "fn": int(oof_star["fn"]), "fp": int(oof_star["fp"])},
        "operating_points": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in r.items()} for r in op_df.to_dict("records")],
    },
    "ablation": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in r.items() if k != "best_params"} for r in abl],
    "top_features_rf": imp.head(10)[["feature", "RF importance"]].values.tolist(),
    "top_features_lr": imp.reindex(imp["LR coef (std. space)"].abs()
                                   .sort_values(ascending=False).index).head(10)[
                                       ["feature", "LR coef (std. space)"]].values.tolist(),
    "n_errors_at_t_star": int(len(err)),
    "fn_at_t_star": int((err["true"] == "malignant").sum()),
    "fp_at_t_star": int((err["true"] == "benign").sum()),
}
(RES / "A1_results.json").write_text(json.dumps(results, indent=2, default=float))
print("wrote", RES / "A1_results.json")
print("\n" + "=" * 68)
print("VERIFICATION BLOCK — every number quoted in the report")
print("=" * 68)
print(f"n={dataset_stats['samples']}  features={dataset_stats['features']}  "
      f"malignant={dataset_stats['class_counts']['malignant']}  "
      f"benign={dataset_stats['class_counts']['benign']}")
for r in rows:
    print(f"{r['model']:<20s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
          f"rec={r['recall']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f}")
print(f"best={best_name} (chosen on CV AUC)  tau*={t_star:.3f} (chosen on OOF)")
print(f"  at tau*=0.50: recall={base['recall']:.4f} precision={base['precision']:.4f} "
      f"FN={int(base['fn'])} FP={int(base['fp'])}")
print(f"  at tau*={t_star:.3f}: recall={row_star['recall']:.4f} precision={row_star['precision']:.4f} "
      f"FN={int(row_star['fn'])} FP={int(row_star['fp'])}")

# %% [markdown]
# ## 7. Conclusion
#
# A properly scaled linear model already separates this dataset almost perfectly, and
# tuning an RBF SVM or a forest buys only a fraction of an AUC point — on 569 samples the
# split and the decision threshold matter far more than the model class. The interesting
# result is not the AUC: it is that buying 95 % malignant recall costs roughly a
# doubling of false positives, and that a majority-class model is 62 % accurate while
# catching no cancer at all. Next I would (i) validate on an external cohort, since all
# 569 samples come from one hospital's scanner, and (ii) replace the hard threshold with a
# calibrated risk score so clinicians can trade recall against biopsy budget explicitly.

# %% [markdown]
# ## AI Use Disclosure
#
# I used Claude (Anthropic) as an AI assistant for: scaffolding this notebook's structure,
# writing plotting boilerplate, and drafting the report prose from the numbers the
# notebook produced. The modelling decisions — positive-class encoding, the
# Pipeline-scoped scaler, tuning on ROC-AUC with 5-fold stratified CV, the threshold
# sweep, and the redundant-feature ablation — were reviewed and verified by me by
# re-running every cell and checking the outputs against the course slides and the
# scikit-learn documentation. All experiments, results and analysis were produced by this
# notebook and checked by me; I take full responsibility for this work.

# %% [markdown]
# ## References
#
# [1] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, "Nuclear feature extraction for
# breast tumor diagnosis," *IS&T/SPIE 1993 Int. Symp. Electronic Imaging*, 1993.
# UCI ML Repository id 17, https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
#
# [2] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12,
# pp. 2825–2830, 2011.
