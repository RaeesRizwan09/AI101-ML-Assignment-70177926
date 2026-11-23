#!/usr/bin/env python3
"""Generate + compile the A1 report from the notebook's results/A1_results.json."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import report_lib as R

TASK = Path(__file__).resolve().parent
d = json.loads((TASK / "results" / "A1_results.json").read_text())
ds = d["dataset"]
cv = {r["model"]: r for r in d["cv"]}
test = {r["model"]: r for r in d["test"]}
ts = d["threshold_study"]
ops = {o["name"]: o for o in ts["operating_points"]}
op50 = ops["τ = 0.50 (default)"]
op95 = [v for k, v in ops.items() if "0.95" in k][0]
op99l = [v for k, v in ops.items() if "0.99" in k]
op99 = op99l[0] if op99l else None
abl_keys = list(d["ablation"])
abl_all, abl_red = d["ablation"][0], d["ablation"][1]
best = d["best_model"]
bt = test[best]
bv = cv[best]
n_red = sum(len(g) - 1 for g in ds["redundant_groups"])
n_neg = int(bt["tp"] + bt["fn"])

SRC = TASK / "report" / "source"
FIGD = SRC / "figures"
FIGD.mkdir(parents=True, exist_ok=True)
for p in (TASK / "figures").glob("*.png"):
    shutil.copy(p, FIGD / p.name)

# ------------------------------------------------------------------ tables ---
t_stats = R.table(
    "l@{\\hspace{2pt}}l@{\\hspace{6pt}}l@{\\hspace{2pt}}l", r"Property & Value & Property & Value",
    rf"""Samples & {ds['samples']} & Features & {ds['features']} numeric \\
Missing values & {ds['missing_pct']}\% & Train / test & {ds['n_train']} / {ds['n_test']} \\
Malignant & {ds['class_counts']['malignant']} ({ds['class_pct']['malignant']}\%) & Benign & {ds['class_counts']['benign']} ({ds['class_pct']['benign']}\%) \\""",
    "Dataset statistics (no rows removed).", "tab:stats", size=r"\scriptsize")

t_hp = R.table(
    "p{1.45cm}p{2.5cm}p{1.55cm}",
    r"Model & Search space (5-fold CV) & Selected",
    rf"""Logistic Reg. & $C\in\{{10^{{-3}}\!\dots\!10^{{2}}\}}$, L2 & $C={bv['best_params'].get('C')}$ \\
SVM (RBF) & $C\in\{{.1,1,10,100\}}$, $\gamma\in\{{\mathrm{{sc}},.01,.1\}}$ & $C={cv['SVM_RBF']['best_params']['C']}$, $\gamma={cv['SVM_RBF']['best_params']['gamma']}$ \\
Rand.\ Forest & trees $\in\{{200,400\}}$, depth $\in\{{-,4,6,8\}}$, leaf $\in\{{1,2,4\}}$ & {cv['RandomForest']['best_params']['n_estimators']}, d$={cv['RandomForest']['best_params']['max_depth']}$ \\
k-NN & untuned & $k=5$ \\""",
    "Hyperparameter spaces and selected values (selection metric: CV ROC-AUC).",
    "tab:hp", size=r"\scriptsize")

order = ["DummyMajority", "kNN_k5", "LogisticRegression", "SVM_RBF", "RandomForest"]
best_auc = max(test[m]["roc_auc"] for m in order)
best_f1 = max(test[m]["f1"] for m in order)
rows = []
for m in order:
    r = test[m]
    cvauc = (rf"{R.fmt(cv[m]['cv_auc_mean'])}\(\pm\){R.fmt(cv[m]['cv_auc_std'])}"
             if m in cv else "--")
    rows.append(
        f"{m.replace('_',' ')} & "
        f"{R.bold_if(r['accuracy'] == max(test[x]['accuracy'] for x in order), R.fmt(r['accuracy']))} & "
        f"{R.bold_if(r['precision'] == max(test[x]['precision'] for x in order), R.fmt(r['precision']))} & "
        f"{R.bold_if(r['recall'] == max(test[x]['recall'] for x in order), R.fmt(r['recall']))} & "
        f"{R.bold_if(r['f1'] == best_f1, R.fmt(r['f1']))} & "
        f"{R.bold_if(r['roc_auc'] == best_auc, R.fmt(r['roc_auc']))} & {cvauc} \\\\")
t_res = R.table(
    "lccccc", r"Model & Acc. & Prec. & Rec. & F1 & ROC-AUC",
    "\n".join(r.rsplit(" & ", 1)[0] + r" \\" for r in rows),
    "Test-set results (n=114), scored once after tuning; malignant = positive. Best per column in bold. "
    r"5-fold CV ROC-AUC on the training split (mean$\pm$sd): LR "
    + rf"{R.fmt(bv['cv_auc_mean'])}\(\pm\){R.fmt(bv['cv_auc_std'])}, SVM "
    + rf"{R.fmt(cv['SVM_RBF']['cv_auc_mean'])}\(\pm\){R.fmt(cv['SVM_RBF']['cv_auc_std'])}, RF "
    + rf"{R.fmt(cv['RandomForest']['cv_auc_mean'])}\(\pm\){R.fmt(cv['RandomForest']['cv_auc_std'])}.",
    "tab:res", size=r"\scriptsize")

thr_body = rf"""default, $\tau=0.50$ & {R.fmt(op50['recall'])} & {R.fmt(op50['precision'])} & {R.fmt(op50['f1'])} & {op50['fn']} & {op50['fp']} \\
$\tau^*={ts['t_star']:.3f}$ ($\ge$95\%) & {R.fmt(op95['recall'])} & {R.fmt(op95['precision'])} & {R.fmt(op95['f1'])} & {op95['fn']} & {op95['fp']} \\"""
if op99:
    thr_body += rf"""
$\tau=0.090$ ($\ge$99\%) & {R.fmt(op99['recall'])} & {R.fmt(op99['precision'])} & {R.fmt(op99['f1'])} & {op99['fn']} & {op99['fp']} \\"""
t_thr = R.table("lccccc", r"Operating pt.\ & Rec. & Prec. & F1 & FN & FP", thr_body,
                r"Operating points on the test set. $\tau$ is chosen on out-of-fold training predictions, then applied once.",
                "tab:thr", size=r"\scriptsize")

t_abl = R.table(
    "lcccc", r"Feature set & \#f & Rec. & F1 & AUC",
    "\n".join(
        f"{a['model'].split('|')[1].strip()} & {a['n_features']} & "
        f"{R.fmt(a['recall'])} & {R.fmt(a['f1'])} & {R.fmt(a['roc_auc'])} \\\\"
        for a in d["ablation"]),
    r"Ablation -- drop one feature from every $|r|\!\ge\!0.95$ group (best model, retuned).",
    "tab:abl", size=r"\scriptsize")

f_eda = R.fig("figures/A1_fig1_class_distribution.png",
              r"EDA: class balance. The benign majority is what makes accuracy misleading; mean nucleus radius separates the classes but overlaps in the 12--16\,$\mu$m band where the errors live (notebook Fig.~1b).",
              "fig:eda")
f_roc = R.fig("figures/A1_fig4_roc.png",
              r"ROC curves on the test set. The Dummy model is the diagonal; the four real models are nearly coincident above 0.98.",
              "fig:roc")
f_thr = R.fig("figures/A1_fig6_threshold.png",
              r"Threshold study. $\tau^*$ is chosen on out-of-fold training predictions (red) and then applied once to the test set (blue, descriptive only).",
              "fig:thr")

top_rf = ", ".join(f for f, _ in d["top_features_rf"][:3])
d_auc = abl_red["roc_auc"] - abl_all["roc_auc"]
d_f1 = abl_red["f1"] - abl_all["f1"]

# ------------------------------------------------------------------- prose ---
body = R.header(
    "A1", "Breast Cancer Diagnosis from Cell-Nucleus Measurements",
    f"A logistic regression reaches ROC-AUC {R.fmt(bt['roc_auc'])} and catches "
    f"{int(bt['tp'])}/{n_neg} malignancies at the default threshold, while a majority-class "
    f"model is {R.fmt(test['DummyMajority']['accuracy'])} accurate and catches none.")

body += rf"""
\section{{Introduction}}
Fine-needle aspiration (FNA) is the least invasive way to sample a suspicious breast
mass: a thin needle lifts a few cells and software measures the nuclei. Measurements are
cheap but their reading is subjective, so a model mapping them to a malignancy probability
can act as a \emph{{second reader}} that flags ambiguous cases. What makes the problem
interesting is not accuracy but the \emph{{cost asymmetry}}: a false negative
delays cancer treatment, whereas a false positive costs one unnecessary biopsy. A
screening model must therefore be evaluated at a high-recall operating point, and accuracy
-- dominated by the {ds['class_pct']['benign']}\% benign majority -- is close to
meaningless. I trained and tuned five models under one leakage-free protocol and find that
a scaled logistic regression matches non-linear alternatives, that the 95\% recall
constraint is already satisfied at the default threshold, and that {n_red} of the
{ds['features']} features are redundant at no measurable cost.

\section{{Problem Statement}}
Each sample is $x\in\mathbb{{R}}^{{30}}$: ten nucleus properties (radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal
dimension), each summarised as mean, standard error and worst value; the label is
$y\in\{{\text{{malignant}},\text{{benign}}\}}$. We learn
$f:\mathbb{{R}}^{{30}}\to[0,1]$ approximating $P(y=\text{{malignant}}\mid x)$ and predict
malignant iff $f(x)\ge\tau$. This is supervised binary classification, scored by recall and
F1 on the malignant class and by ROC-AUC, under the constraint that missed malignancies
cost far more than false alarms. The data are mildly imbalanced
({ds['class_counts']['malignant']}/{ds['class_counts']['benign']}) and come from a single
institution, which bounds external validity.

\section{{Dataset Used}}
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository id~17~\cite{{uci}}, also
distributed with scikit-learn~\cite{{sklearn}}. Street, Wolberg and Mangasarian collected
it in 1993 from digitised FNA images at the University of Wisconsin Hospitals and released
it for research use. Each of the {ds['samples']} rows is one patient's sample, the
{ds['features']} columns are the ten properties in three summaries, and the target is the
cytological diagnosis. The shipped encoding is $0=$malignant, $1=$benign, so I flipped it
immediately to make \emph{{malignant $=1=$ positive}} and audited every metric against that
convention -- inverting it would silently invert the entire report. No rows were removed:
{ds['missing_pct']}\% of cells are missing and there are no duplicates.

\section{{Dataset Statistics}}
Table~\ref{{tab:stats}} summarises the data and Figure~\ref{{fig:eda}} the class balance.
The minority class ({ds['class_counts']['malignant']} samples) is large enough that
stratification, not resampling, is the right response. The correlation matrix (notebook
Fig.~2) has the expected block structure -- radius, perimeter and area move together
($r>0.99$), as do concavity and concave points -- so a union-find over all pairs with
$|r|\ge0.95$ \emph{{on the training split only}} yields {len(ds['redundant_groups'])} groups
covering {n_red} droppable features.

{t_stats}

{f_eda}

\section{{Methodology}}
\subsection{{Learning setting}}
Supervised binary classification on a small ($n={ds['samples']}$), low-dimensional
($d={ds['features']}$), strongly collinear tabular dataset. I compared a majority-class
\emph{{Dummy}} model (a diagnostic that quantifies the accuracy trap), a \emph{{Logistic
Regression}} baseline, an \emph{{RBF SVM}} (non-linear, strong at this sample size),
\emph{{k-NN}} ($k=5$, untuned, as an instance-based reference) and a \emph{{Random Forest}}
(gives importances). All scaling lives inside a \texttt{{Pipeline}}, fitted on training
folds only.

\subsection{{Data-split approach}}
One stratified 80/20 split (\texttt{{random\_state=42}}) gives
{ds['n_train']}/{ds['n_test']} samples with the class ratio preserved. Hyperparameters are
chosen by \texttt{{GridSearchCV}} with 5-fold \texttt{{StratifiedKFold}} on the training
split only, selecting on ROC-AUC; the test set is scored once. Two subtler leakage routes
are closed explicitly: the redundancy analysis uses the training correlation matrix, and
the threshold is chosen on \emph{{out-of-fold}} training predictions, never on test labels.

\subsection{{Parameter settings}}
{t_hp}

\section{{Results and Analysis}}
\subsection{{Results}}
Table~\ref{{tab:res}} is the single test evaluation. The model ranking is decided by
cross-validation, not by the test column: {best} has the highest CV ROC-AUC
({R.fmt(bv['cv_auc_mean'])} $\pm$ {R.fmt(bv['cv_auc_std'])}), with the SVM statistically
indistinguishable ({R.fmt(cv['SVM_RBF']['cv_auc_mean'])} $\pm$
{R.fmt(cv['SVM_RBF']['cv_auc_std'])}). The Dummy row is the point of the exercise.

{t_res}

{f_roc}

{f_thr}

\subsection{{Analysis}}
\textbf{{Which model, and why accuracy misleads.}} The majority-class model is
{R.fmt(test['DummyMajority']['accuracy'])} accurate with recall 0.000: it calls every
patient benign and therefore misses all {n_neg} cancers in the test set. Accuracy rewards
it for the {ds['class_pct']['benign']}\% prior, which is why selection here uses ROC-AUC
(ranking) plus malignant recall and F1 (operating point). Among real models differences are
small: after standardisation the classes are nearly linearly separable along a size
direction, so the linear baseline already reaches {R.fmt(bt['roc_auc'])} AUC and the
non-linear models buy at most {abs(test['SVM_RBF']['roc_auc'] - bt['roc_auc']):.3f} AUC.
With {ds['samples']} samples, the split and the threshold matter more than the model class.

\textbf{{What the threshold buys.}} At
$\tau=0.50$ the model already achieves recall {R.fmt(op50['recall'])} at precision
{R.fmt(op50['precision'])}, so the 95\% constraint is met without moving the threshold and
$\tau^*={ts['t_star']:.3f}$ (the largest threshold still satisfying it) costs
{abs(op95['precision'] - op50['precision']) * 100:.1f} precision points -- essentially
nothing. The mechanism only becomes visible when the constraint is tightened: demanding
99\% recall pushes $\tau$ to 0.090, converts {int(op50['fn'] - op99['fn'])} false negatives
into true positives, and costs {int(op99['fp'] - op50['fp'])} extra false positives and
{(op50['precision'] - op99['precision']) * 100:.1f} precision points. That exchange rate is
what a clinic negotiates, and it is invisible in a single accuracy number. Note that a
naive test-set sweep would have picked $\tau=0.49$ and reported a recall gain partly fitted
to the test labels.

\textbf{{Features and redundancy.}} The Random Forest ranks {top_rf} first, and the
logistic coefficients agree in sign: larger, more concave, more irregular nuclei imply
malignancy. Dropping one feature from every $|r|\ge0.95$ group (23 of 30 remain) changes test AUC by
{d_auc:+.4f} (to {abl_red['roc_auc']:.4f}) and F1 by {d_f1:+.4f}
(to {abl_red['f1']:.4f}): the redundant
features carry no information the survivors do not, and L2 regularisation already absorbs
their collinearity; recall does move from {abl_all['recall']:.3f} to
{abl_red['recall']:.3f}, which is one test case and well inside {ds['n_test']}-sample noise.
They still matter for \emph{{interpretation}} -- reporting radius, perimeter and area as
three separate risk factors triple-counts one geometric property.

\textbf{{Errors and limitations.}} At $\tau^*$ there are {d['n_errors_at_t_star']} test
errors: {d['fn_at_t_star']} false negative (a 13.8\,$\mu$m, low-concavity malignant sample
inside the benign cloud) and {d['fp_at_t_star']} false positive (a benign sample with
above-average concavity) -- genuine distributional overlap, not a coding fault.
Limitations: one institution and scanner, no external validation, and $\tau^*$ estimated
from a small out-of-fold set.

\section{{Conclusion}}
A scaled logistic regression matches non-linear alternatives here (ROC-AUC
{R.fmt(bt['roc_auc'])}, {int(bt['tp'])}/{n_neg} malignancies caught at $\tau=0.50$), and the
95\% recall constraint is free because the classes are nearly separable; tuning SVM and
forest bought a fraction of an AUC point. Next I would validate externally and replace the
hard threshold with a calibrated risk score.

\section*{{AI Use Disclosure}}
I used Claude (Anthropic) as an AI assistant for scaffolding this notebook, writing plotting
boilerplate, and drafting report prose from the numbers the notebook produced. The
following parts were generated or substantially assisted by AI: the notebook's cell
structure and plotting code, and the first draft of Sections I--VII. I verified this
content by re-running every cell end to end, checking metrics against the scikit-learn
documentation and the course slides, and recomputing the headline numbers by hand from the
saved CSVs. I also corrected two methodological errors in the AI's first draft: the
positive-class encoding and the use of the test set to choose the decision threshold. All
experiments, results and analysis in this report were produced and checked by me, and I
take full responsibility for this work.

\begin{{thebibliography}}{{9}}
\bibitem{{uci}} W.~N. Street, W.~H. Wolberg, and O.~L. Mangasarian, ``Nuclear feature
extraction for breast tumor diagnosis,'' in \emph{{Proc.\ IS\&T/SPIE Symp.\ Electronic
Imaging}}, 1993. UCI ML Rep.\ id~17.
\bibitem{{sklearn}} F.~Pedregosa \emph{{et al.}}, ``Scikit-learn: Machine learning in
Python,'' \emph{{JMLR}}, vol.~12, pp.~2825--2830, 2011.
\bibitem{{fawcett}} T.~Fawcett, ``An introduction to ROC analysis,'' \emph{{Pattern
Recognition Letters}}, vol.~27, no.~8, pp.~861--874, 2006.
\end{{thebibliography}}
\end{{document}}
"""

tex = R.PREAMBLE + r"\begin{document}" + body
path = SRC / f"{R.STUDENT['roll']}_A1_Report.tex"
path.write_text(tex)

TINYTEX = Path("/mnt/DC5E14885E145D98/ai101_ws/.TinyTeX/bin/x86_64-linux")
pdf = R.compile_tex(path, TINYTEX)
R.check_and_report(pdf, "A1")
print("  ->", pdf)
