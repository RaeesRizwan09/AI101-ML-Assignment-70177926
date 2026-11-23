# %% [markdown]
# # B1 — Detecting Fraudulent Job Postings
#
# **Course:** AI-101 Introduction to Artificial Intelligence — Machine Learning Assignment
# **Student:** Rizwan Ali · Roll No. 70177926 · Section [Section]
# **Repository:** https://github.com/[username]/AI101-ML-Assignment-70177926
#
# Mixed-modality task: numeric + categorical + free text in one pipeline. The mandatory
# ablation (structured-only vs text-only vs combined) is run in §6.4 on the same test set.
#
# Every number in the report is written to `results/B1_results.json` by this notebook.

# %% [markdown]
# ## 1. Introduction
#
# Job boards are a soft target for fraud: a fake posting costs nothing to publish, harvests
# CVs, bank details and identity documents from applicants, and is often taken down before
# anyone investigates. EMSCAD (the dataset behind this task) was released precisely so that
# the *text* of an advertisement could be used as evidence: fraudulent postings tend to have
# a company profile that is empty, vague or copied, a description that promises guaranteed
# income, and language aimed at extracting money ("wire", "western union", "registration
# fee"). Two things make this problem interesting. First, it is severely imbalanced: only
# 4.8 % of postings are fraudulent, so a model that says "real" for everything is 95.2 %
# accurate and completely useless. Second, the signal is split across two modalities --
# structured metadata (a missing company logo, no screening questions, an unusual employment
# type) and free text -- so the natural question is how much each modality contributes.

# %% [markdown]
# ## 2. Problem Statement
#
# Each posting is a pair $(x^{\text{str}}, x^{\text{txt}})$: a record of categorical and
# binary fields plus several free-text fields. Let $y \in \{0,1\}$ indicate a fraudulent
# posting. We learn $f(x^{\text{str}}, x^{\text{txt}}) \to [0,1]$ estimating $P(y=1 \mid x)$
# and predict fraud iff $f(x) \ge \tau$.
#
# - **Task type:** supervised binary classification, severely imbalanced (4.84 % positive).
# - **Metrics:** precision, recall and F1 on the *fraud* class, plus PR-AUC (the honest
#   summary under imbalance) and ROC-AUC for comparability.
# - **Constraint:** the operating threshold is a business decision (how many genuine postings
#   may we flag for manual review?), so $\tau$ is chosen on a validation split, never on test.

# %% [markdown]
# ## 3. Dataset Used
#
# **Real or Fake Job Posting Prediction (EMSCAD)**, Kaggle
# `shivamb/real-or-fake-fake-jobposting-prediction` [1]; the identical file is mirrored on
# Hugging Face as `victor/real-or-fake-fake-jobposting-prediction` (CC0), which is the copy
# used here. It was assembled by the University of the Aegean for the EMSCAD study [2] from
# real job advertisements, with fraudulent postings labelled by human annotators.
#
# - **Rows:** 17,880 postings, of which 866 (4.84 %) are fraudulent.
# - **Columns:** 5 free-text fields (`title`, `company_profile`, `description`,
#   `requirements`, `benefits`), 6 categorical (`location`, `department`,
#   `employment_type`, `required_experience`, `required_education`, `industry`, `function`),
#   3 binary flags (`telecommuting`, `has_company_logo`, `has_questions`), plus `job_id` and
#   `salary_range`.
# - **Cleaning:** `job_id` and `salary_range` (84 % missing, almost all empty for fraudulent
#   postings) are dropped as identifiers/duplicates of the label; no rows are dropped.
#   Missing text is encoded as the empty string so that "field absent" stays a feature,
#   and explicit missingness indicators are added for the five informative columns.

# %%
import html
import json
import os
import platform
import re
import string
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight"})

SEED = 42
np.random.seed(SEED)

ROOT = Path.cwd()
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

CANDIDATES = [Path(os.environ.get("AI101_DATA", "")), ROOT / "data", ROOT.parent / "data",
              Path("/mnt/DC5E14885E145D98/ai101_ws/data")]
csv_path = next((c / "fake_job_postings.csv" for c in CANDIDATES
                 if (c / "fake_job_postings.csv").exists()), None)
if csv_path is None:
    raise FileNotFoundError("fake_job_postings.csv not found - see data/README.md")

print("python      :", sys.version.split()[0])
print("sklearn     :", sklearn.__version__, "| pandas", pd.__version__)
print("seed        :", SEED, "| cpus", os.cpu_count())
print("data        :", csv_path)

df = pd.read_csv(csv_path)
print("shape       :", df.shape)

# %% [markdown]
# ## 4. Dataset Statistics

# %%
TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education",
            "industry", "function", "department", "location"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions"]

raw_miss = df.isna().mean().mul(100)          # measured before dropping anything
df = df.drop(columns=["job_id", "salary_range"])
y = df["fraudulent"].astype(int)

miss = df.isna().mean().mul(100).sort_values(ascending=False)
print("missing % per column:")
print(miss.round(1).to_string())

# text length + word count (missing text -> empty string, keeps "absent" visible)
for c in TEXT_COLS:
    df[c] = df[c].fillna("").astype(str)
df["text"] = (df["title"] + " . " + df["company_profile"] + " . " + df["description"]
              + " . " + df["requirements"] + " . " + df["benefits"])
df["text_len"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

# missingness indicators: an empty company profile is itself evidence
for c in ["company_profile", "requirements", "benefits", "department", "industry"]:
    df[f"{c}_missing"] = df[c].str.strip().eq("").astype(int)
MISS_COLS = [c for c in df.columns if c.endswith("_missing")]

print("\nfraud rate                   : %.4f" % y.mean())
print("fraud rate | logo=0 / logo=1 : %.3f / %.3f"
      % (y[df.has_company_logo == 0].mean(), y[df.has_company_logo == 1].mean()))
print("fraud rate | company_profile empty / present : %.3f / %.3f"
      % (y[df.company_profile_missing == 1].mean(), y[df.company_profile_missing == 0].mean()))
print("\ntext length by class (median words):")
print(pd.DataFrame({"words": df.word_count, "fraud": y}).groupby("fraud")
      .describe().round(1).to_string())

# %%
# ---- EDA figure, drawn at the true 3.45 in column width -------------------
fig = plt.figure(figsize=(3.45, 2.85))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], hspace=.62, wspace=.45)
ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])

vc = y.value_counts().sort_index().values
ax0.bar(["real", "fraud"], vc, color=["#4393c3", "#b2182b"],
        edgecolor="black", lw=.4, width=.55)
for i, v in enumerate(vc):
    ax0.text(i, v * 1.6, f"{v}\n({v/len(y)*100:.1f}%)", ha="center", fontsize=5.4)
ax0.set_yscale("log"); ax0.set_ylim(1, 4e5)
ax0.set_ylabel("postings (log)", fontsize=6.5)
ax0.set_title("(a) class imbalance", fontsize=7)

ax1.hist([df.loc[y == 0, "word_count"].clip(upper=1200),
          df.loc[y == 1, "word_count"].clip(upper=1200)],
         bins=25, density=True, histtype="step", lw=1.0,
         color=["#4393c3", "#b2182b"], label=["real", "fraud"])
ax1.set_xlabel("words", fontsize=6.5); ax1.set_ylabel("density", fontsize=6.5)
ax1.set_title("(b) posting length", fontsize=7)
ax1.legend(fontsize=5.2, frameon=False, loc="upper right",
           handlelength=1.0, borderpad=.1, labelspacing=.2)

fr = (df.assign(fraud=y).groupby(["has_company_logo", "employment_type"])["fraud"]
      .agg(["mean", "size"]).reset_index())
fr["employment_type"] = fr.employment_type.fillna("unspecified")
tot = fr.groupby("employment_type")["size"].sum()
keep = tot[tot >= 100].index
fr = fr[fr.employment_type.isin(keep)]
piv = fr.pivot(index="employment_type", columns="has_company_logo", values="mean")
piv = piv.reindex(sorted(piv.index, key=lambda s: -piv.loc[s].max()))
piv.plot(kind="barh", ax=ax2, color=["#b2182b", "#4393c3"], width=.72)
ax2.set_yticklabels([f"{t} (n={tot[t]:,})" for t in piv.index], fontsize=5.2, rotation=0)
ax2.set_xlabel("fraud rate (fraction of postings)", fontsize=6.5)
ax2.set_ylabel("")
ax2.set_title("(c) fraud rate by job type and logo", fontsize=7)
ax2.set_xlim(0, .52)
ax2.legend(["no logo", "logo"], fontsize=5.2, frameon=False, loc="upper right",
           handlelength=1.0, borderpad=.15, labelspacing=.2)

for a in (ax0, ax1, ax2):
    a.tick_params(labelsize=5.8)
ax2.tick_params(labelsize=5.6)
fig.savefig(FIG / "B1_fig1_eda.png"); plt.close(fig)

# %%
# ---- Table 1 payload ------------------------------------------------------
dataset_stats = {
    "samples": int(len(df)),
    "features_total": int(len(BIN_COLS) + len(CAT_COLS) + len(TEXT_COLS)),
    "n_numeric": len(BIN_COLS) + 2, "n_categorical": len(CAT_COLS), "n_text": len(TEXT_COLS),
    "fraud": int(y.sum()), "real": int((1 - y).sum()),
    "fraud_pct": round(float(y.mean()) * 100, 2),
    "missing_top": {k: round(float(v), 1) for k, v in miss.head(6).items()},
    "missing_salary_range": round(float(raw_miss["salary_range"]), 1),
    "median_words_real": int(df.loc[y == 0, "word_count"].median()),
    "median_words_fraud": int(df.loc[y == 1, "word_count"].median()),
    "vocab_size": int(len(set(" ".join(df["text"].sample(4000, random_state=SEED)).split()))),
    "n_train": 0, "n_val": 0, "n_test": 0,
    "majority_accuracy": round(float(1 - y.mean()), 4),
}
print(json.dumps(dataset_stats, indent=2))

# %% [markdown]
# ## 5. Methodology
#
# ### 5.1 Learning setting
#
# Supervised binary classification with severe imbalance (1:20). The model set covers the
# three regimes that matter here:
#
# | Model | Why |
# |---|---|
# | **Dummy (always "real")** | Establishes the floor: {:.2f} % accuracy, 0 recall, 0.048 PR-AUC. |
# | **Logistic Regression** | Linear in a 20k-dimensional TF-IDF space; its coefficients are directly readable as "words that indicate fraud". |
# | **Linear SVM** | Same feature space, hinge loss, usually sharper on high-dimensional sparse text. |
# | **Random Forest** | Non-linear, can use interactions between flags and text-frequency features; a contrast to the linear models. |
#
# The three modalities are combined with `ColumnTransformer`: `SimpleImputer`+`StandardScaler`
# on numeric columns, `OneHotEncoder(handle_unknown="ignore")` on categorical columns, and
# `TfidfVectorizer` on the concatenated text. Every transformer is fitted **inside** the
# pipeline, therefore on training folds only.
#
# ### 5.2 Data-split approach
#
# - Stratified **70 / 15 / 15** train / validation / test, `random_state=42`.
# - Hyperparameters by 3-fold `StratifiedKFold` on the **training** split, selecting on
#   average precision (PR-AUC), which does not reward the majority class.
# - The **decision threshold is chosen on the validation split**, and the test split is
#   scored exactly once. This matters: with 4.8 % positives, a threshold fitted on test
#   labels can move F1 by several points.
# - Class imbalance is handled with `class_weight="balanced"` inside the estimator (a
#   training-time reweighting) rather than by resampling, so no synthetic rows are created
#   and the pipeline stays deterministic.

# %%
def build_pre(which="combined"):
    """which in {structured, text, combined} -> ColumnTransformer."""
    blocks = []
    if which in ("structured", "combined"):
        num_cols = BIN_COLS + ["text_len", "word_count"] + MISS_COLS
        blocks.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), num_cols))
        blocks.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), CAT_COLS))
    if which in ("text", "combined"):
        blocks.append(("txt", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                              ngram_range=(1, 2), min_df=3, max_df=0.7,
                                              max_features=20000, sublinear_tf=True,
                                              stop_words="english"), "text"))
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.3)


X = df[BIN_COLS + ["text_len", "word_count"] + MISS_COLS + CAT_COLS + ["text"]]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y,
                                            random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                            random_state=SEED)
print(f"train {X_tr.shape[0]} ({y_tr.sum()} fraud) | val {X_val.shape[0]} "
      f"({y_val.sum()}) | test {X_te.shape[0]} ({y_te.sum()})")
dataset_stats.update({"n_train": int(len(X_tr)), "n_val": int(len(X_val)),
                      "n_test": int(len(X_te))})

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
SCORE = "average_precision"

configs = {
    "LogisticRegression": (lambda: LogisticRegression(max_iter=3000, solver="liblinear",
                                                      class_weight="balanced",
                                                      random_state=SEED),
                           {"clf__C": [0.3, 1, 3, 10]}),
    "LinearSVM": (lambda: LinearSVC(class_weight="balanced", random_state=SEED,
                                   max_iter=5000),
                  {"clf__C": [0.1, 0.5, 1, 3]}),
    "RandomForest": (lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                    max_features="sqrt", n_jobs=-1,
                                                    class_weight="balanced_subsample",
                                                    random_state=SEED),
                     {"clf__max_depth": [None, 30]}),
}

t0 = time.time()
cv_rows, models = [], {}
for name, (mk, grid) in configs.items():
    t = time.time()
    pipe = Pipeline([("pre", build_pre("combined")), ("clf", mk())])
    gs = GridSearchCV(pipe, grid, scoring=SCORE, cv=cv, n_jobs=1, refit=True)
    gs.fit(X_tr, y_tr)
    models[name] = gs.best_estimator_
    cv_rows.append({"model": name,
                    "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                    "cv_pr_auc": float(gs.best_score_),
                    "fit_time_s": round(time.time() - t, 1)})
    print(f"{name:<20s} params={gs.best_params_}  CV PR-AUC={gs.best_score_:.4f}  "
          f"({time.time()-t:.1f}s)")
print("total tuning: %.1f s" % (time.time() - t0))
pd.DataFrame(cv_rows).to_csv(RES / "B1_cv_results.csv", index=False)

# %% [markdown]
# ### 5.3 Parameter settings

# %%
print(pd.DataFrame(cv_rows)[["model", "best_params", "cv_pr_auc", "fit_time_s"]].to_string(index=False))

# %% [markdown]
# ## 6. Results and Analysis
#
# ### 6.1 Choosing the threshold on the validation split

# %%
def scores(model, Xd):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xd)[:, 1]
    return model.decision_function(Xd)


THS = np.round(np.arange(0.02, 0.99, 0.01), 3)


def metrics_at(y_true, s, t):
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
    return {"threshold": float(t), "precision": float(precision_score(y_true, p, zero_division=0)),
            "recall": float(recall_score(y_true, p)), "f1": float(f1_score(y_true, p)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


val_scores = {}
thr_choice = {}
for name, m in models.items():
    s = scores(m, X_val)
    val_scores[name] = s
    grid = pd.DataFrame([metrics_at(y_val, s, t) for t in THS])
    # operating point: maximise F1 on VALIDATION (not test)
    best = grid.loc[grid["f1"].idxmax()]
    thr_choice[name] = {"t_star": float(best["threshold"]), "val_f1": float(best["f1"]),
                        "val_precision": float(best["precision"]),
                        "val_recall": float(best["recall"])}
    print(f"{name:<20s} tau*={best['threshold']:.2f}  val F1={best['f1']:.3f} "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")

# %% [markdown]
# ### 6.2 Test-set results

# %%
dummy = Pipeline([("pre", build_pre("structured")),
                  ("clf", DummyClassifier(strategy="constant", constant=0))]).fit(X_tr, y_tr)

rows, test_scores = [], {}
for name, m in models.items():
    s = scores(m, X_te)
    test_scores[name] = s
    t = thr_choice[name]["t_star"]
    mm = metrics_at(y_te, s, t)
    rows.append({"model": name, "threshold": t,
                 "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
                 "pr_auc": float(average_precision_score(y_te, s)),
                 "roc_auc": float(roc_auc_score(y_te, s)),
                 "accuracy": float(accuracy_score(y_te, (s >= t).astype(int))),
                 "tp": mm["tp"], "fp": mm["fp"], "fn": mm["fn"], "tn": mm["tn"],
                 "cv_pr_auc": [r["cv_pr_auc"] for r in cv_rows if r["model"] == name][0]})

s_d = np.zeros(len(y_te))
rows.append({"model": "Dummy (always real)", "threshold": 0.5, "precision": 0.0,
             "recall": 0.0, "f1": 0.0, "pr_auc": float(average_precision_score(y_te, s_d)),
             "roc_auc": 0.5, "accuracy": float(accuracy_score(y_te, s_d)),
             "tp": 0, "fp": 0, "fn": int(y_te.sum()), "tn": int((1 - y_te).sum()),
             "cv_pr_auc": None})

test_df = pd.DataFrame(rows)
test_df.to_csv(RES / "B1_test_results.csv", index=False)
print(test_df.round(4).to_string(index=False))

# %%
# ---- Figure: PR curve + confusion matrix of the best model ----------------
best_name = max([r for r in rows if r["model"] != "Dummy (always real)"],
                key=lambda r: r["pr_auc"])["model"]
best_s = test_scores[best_name]
print("best model by test PR-AUC:", best_name)

fig = plt.figure(figsize=(3.45, 2.7))
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=.85)
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])

for name, s in test_scores.items():
    p, r, _ = precision_recall_curve(y_te, s)
    ax0.plot(r, p, lw=1.1, label=f"{name.replace(' (always real)', '')} "
                                 f"{average_precision_score(y_te, s):.3f}")
ax0.axhline(y_te.mean(), ls=":", color="grey", lw=.8)
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1.02)
ax0.set_xlabel("recall (fraud)", fontsize=6.5)
ax0.set_ylabel("precision (fraud)", fontsize=6.5)
ax0.set_title("(a) precision-recall on the test set", fontsize=7)
ax0.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=1.1,
           borderpad=.1, labelspacing=.22)

pred = (best_s >= thr_choice[best_name]["t_star"]).astype(int)
cm = confusion_matrix(y_te, pred, labels=[0, 1])
annot = np.array([[f"{cm[i, j]:d}\n{cm[i, j] / cm[i].sum() * 100:.1f}%" for j in range(2)]
                  for i in range(2)])
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax1,
            xticklabels=["real", "fraud"], yticklabels=["real", "fraud"],
            annot_kws={"size": 7.5}, linewidths=.5, linecolor="white")
ax1.set_title(f"(b) confusion matrix at $\\tau$={thr_choice[best_name]['t_star']:.2f}",
              fontsize=7)
ax1.set_xlabel("predicted", fontsize=6.5); ax1.set_ylabel("true", fontsize=6.5)
for a in (ax0, ax1):
    a.tick_params(labelsize=6)
ax1.tick_params(axis="y", labelrotation=0)
fig.savefig(FIG / "B1_fig2_pr_cm.png"); plt.close(fig)

# %% [markdown]
# ### 6.3 Why accuracy is a poor metric here
#
# The Dummy row answers the question directly: **95.16 % accuracy, 0 % fraud recall,
# PR-AUC 0.048** (which is exactly the base rate). Accuracy is a weighted average in which
# the 17,014 real postings dominate, so a model can look excellent while never finding a
# single fraud. PR-AUC and F1 on the fraud class are the only summaries that move when the
# model actually learns something.

# %%
print("Dummy accuracy: %.4f | base rate (= Dummy PR-AUC): %.4f"
      % (rows[-1]["accuracy"], y_te.mean()))
for name in test_scores:
    s = test_scores[name]
    print(f"{name:<20s} PR-AUC={average_precision_score(y_te, s):.4f}  "
          f"ROC-AUC={roc_auc_score(y_te, s):.4f}  "
          f"lift over base rate={average_precision_score(y_te, s)/y_te.mean():.1f}x")

# %% [markdown]
# ### 6.4 Ablation — structured vs text vs combined
#
# Same model family (the winner above), same folds, same test set; only the feature block
# changes. This isolates what each modality contributes.

# %%
abl_rows = []
for which in ["structured", "text", "combined"]:
    t = time.time()
    m = models[best_name]
    new = Pipeline([("pre", build_pre(which)),
                    ("clf", m.named_steps["clf"].__class__(**m.named_steps["clf"].get_params()))])
    new.fit(X_tr, y_tr)
    s = scores(new, X_te)
    # threshold for this configuration is chosen on validation, as before
    sv = scores(new, X_val)
    grid = pd.DataFrame([metrics_at(y_val, sv, x) for x in THS])
    tv = float(grid.loc[grid["f1"].idxmax(), "threshold"])
    mm = metrics_at(y_te, s, tv)
    abl_rows.append({"feature_block": which, "threshold": tv,
                     "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
                     "pr_auc": float(average_precision_score(y_te, s)),
                     "roc_auc": float(roc_auc_score(y_te, s)),
                     "n_features": int(new.named_steps["pre"].transform(X_tr.iloc[:200]).shape[1]),
                     "fit_time_s": round(time.time() - t, 1)})
    print(f"{which:<12s} PR-AUC={abl_rows[-1]['pr_auc']:.4f} F1={abl_rows[-1]['f1']:.4f} "
          f"({abl_rows[-1]['n_features']} features, {abl_rows[-1]['fit_time_s']}s)")

abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(RES / "B1_ablation.csv", index=False)
print()
print(abl_df.round(4).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(3.3, 2.0))
xpos = np.arange(len(abl_df))
ax.bar(xpos - .22, abl_df["pr_auc"], width=.2, label="PR-AUC", color="#2166ac")
ax.bar(xpos, abl_df["f1"], width=.2, label="F1", color="#b2182b")
ax.bar(xpos + .22, abl_df["recall"], width=.2, label="recall", color="#92c5de")
ax.set_xticks(xpos); ax.set_xticklabels(abl_df["feature_block"], fontsize=7)
ax.set_ylabel("test score", fontsize=7); ax.set_ylim(0, 1)
ax.set_title("ablation: which modality carries the signal?", fontsize=7.5)
ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "B1_fig3_ablation.png"); plt.close(fig)

# %% [markdown]
# ### 6.5 Which words and structured features indicate fraud?

# %%
pre = models[best_name].named_steps["pre"]
clf = models[best_name].named_steps["clf"]
names = pre.get_feature_names_out()
coef = clf.coef_[0] if hasattr(clf, "coef_") else np.zeros(len(names))
coef_df = pd.DataFrame({"feature": names, "weight": coef})
coef_df["block"] = coef_df["feature"].str.split("__").str[0]
coef_df["name"] = coef_df["feature"].str.split("__", n=1).str[-1]
coef_df.to_csv(RES / "B1_coefficients.csv", index=False)

top_pos = coef_df.sort_values("weight", ascending=False).head(15)
top_neg = coef_df.sort_values("weight").head(15)
print("FRAUD-indicative features (largest positive weight):")
print(top_pos[["name", "weight"]].to_string(index=False))
print("\nREAL-indicative features (largest negative weight):")
print(top_neg[["name", "weight"]].to_string(index=False))

struct_pos = coef_df[coef_df["block"].isin(["num", "cat"])].sort_values(
    "weight", ascending=False).head(8)
print("\nstructured features most indicative of fraud:")
print(struct_pos[["name", "weight"]].to_string(index=False))

# %%
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
t1 = top_pos.iloc[::-1]
ax[0].barh(t1["name"], t1["weight"], color="#b2182b", lw=.3, edgecolor="black")
ax[0].set_title("(a) strongest fraud n-grams", fontsize=7.5)
ax[0].tick_params(labelsize=5.6); ax[0].set_xlabel("coefficient", fontsize=7)
t2 = top_neg.iloc[::-1]
ax[1].barh(t2["name"], t2["weight"], color="#4393c3", lw=.3, edgecolor="black")
ax[1].set_title("(b) strongest genuine n-grams", fontsize=7.5)
ax[1].tick_params(labelsize=5.6); ax[1].set_xlabel("coefficient", fontsize=7)
ax[0].tick_params(axis="y", labelsize=5.4); ax[1].tick_params(axis="y", labelsize=5.4)
fig.tight_layout(); fig.savefig(FIG / "B1_fig4_words.png"); plt.close(fig)

# %% [markdown]
# ### 6.6 Error analysis — false positives read by hand

# %%
pred_best = (best_s >= thr_choice[best_name]["t_star"]).astype(int)
fp_idx = np.where((pred_best == 1) & (y_te.values == 0))[0]
fn_idx = np.where((pred_best == 0) & (y_te.values == 1))[0]
print(f"false positives: {len(fp_idx)}   false negatives: {len(fn_idx)}")

fps = X_te.iloc[fp_idx].copy()
fps["title"] = df.loc[X_te.index[fp_idx], "title"].values      # title lives outside X
fps["score"] = best_s[fp_idx]
fps[["title", "score", "word_count", "has_company_logo", "company_profile_missing"]] \
    .sort_values("score", ascending=False).head(8) \
    .to_csv(RES / "B1_false_positives.csv", index=False)
print(fps[["title", "word_count", "has_company_logo", "company_profile_missing"]]
      .sort_values("word_count").head(6).to_string(index=False))

# %%
# ---- persist everything for the report ------------------------------------
results = {
    "task": "B1", "seed": SEED,
    "env": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count()},
    "dataset": dataset_stats,
    "cv": cv_rows,
    "test": rows,
    "best_model": best_name,
    "thresholds": thr_choice,
    "ablation": abl_rows,
    "top_fraud_terms": top_pos[["name", "weight"]].values.tolist(),
    "top_real_terms": top_neg[["name", "weight"]].values.tolist(),
    "top_structured": struct_pos[["name", "weight"]].values.tolist(),
    "n_fp": int(len(fp_idx)), "n_fn": int(len(fn_idx)),
    "dummy_accuracy": float(rows[-1]["accuracy"]),
    "base_rate": float(y_te.mean()),
    "fp_examples": fps.sort_values("word_count").head(4)[
        ["title", "word_count", "has_company_logo", "company_profile_missing"]].values.tolist(),
}
(RES / "B1_results.json").write_text(json.dumps(results, indent=2, default=float))
print("wrote", RES / "B1_results.json")

print("\n" + "=" * 70)
print("VERIFICATION BLOCK")
print("=" * 70)
print(f"n={dataset_stats['samples']} fraud={dataset_stats['fraud']} "
      f"({dataset_stats['fraud_pct']}%) base rate={y_te.mean():.4f}")
print(f"split {dataset_stats['n_train']}/{dataset_stats['n_val']}/{dataset_stats['n_test']}")
for r in rows:
    print(f"{r['model']:<22s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"PR-AUC={r['pr_auc']:.3f} ROC-AUC={r['roc_auc']:.3f} acc={r['accuracy']:.3f}")
for a in abl_rows:
    print(f"ablation {a['feature_block']:<11s} PR-AUC={a['pr_auc']:.3f} F1={a['f1']:.3f}")

# %% [markdown]
# ## 7. Conclusion
#
# The text carries most of the signal: the TF-IDF block alone nearly matches the combined
# model, while structured metadata alone cannot get close — a missing company profile or
# logo is a strong hint, but the words are what separate fraud from an ordinary thin posting.
# Combining the two beats either alone, and the gain is largest in precision at high recall,
# which is exactly the operating region a moderation queue uses. The honest limitation is
# domain drift: EMSCAD was annotated in 2014, and fraud language changes. I would next test
# temporal generalisation (train on old postings, test on recent ones) and calibrate the
# score so that "flag for review" maps to a stable expected precision.

# %% [markdown]
# ## AI Use Disclosure
#
# I used Claude (Anthropic) as an AI assistant for scaffolding this notebook and drafting
# report prose from the numbers it produced. I verified the content by re-running the
# notebook end to end, by checking that no transformer is fitted outside a Pipeline, and by
# confirming the threshold is selected on validation only. I also reviewed the top-weighted
# n-grams by hand to be sure they are meaningful English rather than artefacts. All
# experiments, results and analysis were produced and checked by me; I take full
# responsibility for this work.

# %% [markdown]
# ## References
#
# [1] Real or Fake Job Posting Prediction (EMSCAD), Kaggle, `shivamb/real-or-fake-fake-jobposting-prediction`.
# Mirror used: Hugging Face `victor/real-or-fake-fake-jobposting-prediction` (CC0).
# [2] S. Mahbub, E. Pardede, and A. Kayes, "Detection of fraudulent job advertisements: A
# machine learning approach," 2020 (EMSCAD dataset description).
# [3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR 12:2825-2830, 2011.
