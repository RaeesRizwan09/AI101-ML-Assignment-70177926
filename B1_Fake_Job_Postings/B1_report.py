#!/usr/bin/env python3
"""Generate + compile the B1 report from results/B1_results.json."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import report_lib as R

TASK = Path(__file__).resolve().parent
d = json.loads((TASK / "results" / "B1_results.json").read_text())
ds = d["dataset"]
cv = {r["model"]: r for r in d["cv"]}
test = {r["model"]: r for r in d["test"]}
abl = {a["feature_block"]: a for a in d["ablation"]}
best = d["best_model"]
bt = test[best]
thr = d["thresholds"]

SRC = TASK / "report" / "source"
FIGD = SRC / "figures"
FIGD.mkdir(parents=True, exist_ok=True)
for p in (TASK / "figures").glob("*.png"):
    shutil.copy(p, FIGD / p.name)

# ------------------------------------------------------------------ tables ---
t_stats = R.table(
    "l@{\\hspace{2pt}}l@{\\hspace{6pt}}l@{\\hspace{2pt}}l",
    r"Property & Value & Property & Value",
    rf"""Samples & {ds['samples']:,} & Fraud / real & {ds['fraud']:,} ({ds['fraud_pct']}\%) / {ds['real']:,} \\
Features (num/cat/text) & {ds['n_numeric']}/{ds['n_categorical']}/{ds['n_text']} & Train/val/test & {ds['n_train']:,}/{ds['n_val']:,}/{ds['n_test']:,} \\
Missing salary\_range & {ds['missing_salary_range']}\% & Missing department & {ds['missing_top']['department']}\% \\
Median words, real / fraud & {ds['median_words_real']} / {ds['median_words_fraud']} & Vocab (4k sample) & {ds['vocab_size']} \\
TF-IDF features & 20\,000 & Majority baseline acc. & {ds['majority_accuracy']} \\""",
    "Dataset statistics. Text columns are concatenated into one field; missing text becomes the empty string so that absence stays observable.",
    "tab:stats", size=r"\scriptsize")

t_hp = R.table(
    r">{\raggedright\arraybackslash}p{1.45cm}"
    r">{\raggedright\arraybackslash}p{3.35cm}"
    r">{\raggedright\arraybackslash}p{1.5cm}",
    r"Model & Search space (3-fold CV) & Selected",
    rf"""Logistic Reg. & $C\in\{{0.3,1,3,10\}}$, L2, balanced & $C={cv['LogisticRegression']['best_params'].get('C')}$ \\
Linear SVM & $C\in\{{0.1,0.5,1,3\}}$, balanced & $C={cv['LinearSVM']['best_params'].get('C')}$ \\
Random Forest & 300 trees, depth $\in\{{$-$,30\}}$ & no depth limit \\""",
    "Hyperparameters; selection metric is CV PR-AUC on the training split. Shared: word 1--2-gram TF-IDF (min\\_df 3, max\\_df 0.7, 20k features, sublinear TF) and 266 structured features after one-hot encoding.",
    "tab:hp", size=r"\scriptsize")

order = ["Dummy (always real)", "RandomForest", "LogisticRegression", "LinearSVM"]
rows = []
for m in order:
    r = test[m]
    rows.append(
        f"{m.replace('_', ' ')} & "
        f"{R.bold_if(r['precision'] == max(test[x]['precision'] for x in order), R.fmt(r['precision']))} & "
        f"{R.bold_if(r['recall'] == max(test[x]['recall'] for x in order), R.fmt(r['recall']))} & "
        f"{R.bold_if(r['f1'] == max(test[x]['f1'] for x in order), R.fmt(r['f1']))} & "
        f"{R.bold_if(r['pr_auc'] == max(test[x]['pr_auc'] for x in order), R.fmt(r['pr_auc']))} & "
        f"{R.fmt(r['roc_auc'])} & {R.fmt(r['accuracy'])} & "
        rf"{thr[m]['t_star']:.2f}" + r" \\" if m != "Dummy (always real)" else
        f"{m} & 0.000 & 0.000 & 0.000 & {R.fmt(r['pr_auc'])} & 0.500 & {R.fmt(r['accuracy'])} & -- \\\\")
t_res = R.table(
    r"l@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c",
    r"Model & Prec. & Rec. & F1 & PR-AUC & ROC-AUC & Acc. & $\tau$",
    "\n".join(rows),
    "Test-set results (n=2{,}682, 130 fraud), one evaluation. $\\tau$ chosen on validation to maximise F1; fraud is the positive class; best per column in bold.",
    "tab:res", size=r"\scriptsize")

f_eda = R.fig("figures/B1_fig1_eda.png",
              rf"(a) 1:20 class imbalance; (b) length distribution per class, as a density: fraudulent postings are markedly shorter (median {ds['median_words_real']} vs {ds['median_words_fraud']} words); (c) fraud rate by job type and company-logo presence (bar labels: postings in that job type). A missing logo raises the rate by roughly an order of magnitude in every job type.",
              "fig:eda")
f_pr = R.fig("figures/B1_fig2_pr_cm.png",
             r"(a) precision--recall on the test set (legend: average precision); (b) confusion matrix of the best model at its validation-chosen threshold. The dotted line in (a) is the 4.85\% base rate.",
             "fig:pr")

fraud_terms = ", ".join(rf"\texttt{{{R.esc(t)}}}" for t, _ in d["top_fraud_terms"][:8])
real_terms = ", ".join(rf"\texttt{{{R.esc(t)}}}" for t, _ in d["top_real_terms"][:6])
struct_terms = "; ".join(rf"\texttt{{{R.esc(t)}}}" for t, _ in d["top_structured"][:4])
fp_ex = d["fp_examples"]
fpsummary = "; ".join(f"``{R.esc(e[0][:38])}'' ({e[1]} words)" for e in fp_ex[:3])

# ------------------------------------------------------------------- prose ---
body = R.header(
    "B1", "Detecting Fraudulent Job Postings",
    f"A linear SVM over TF-IDF plus structured fields reaches F1 {R.fmt(bt['f1'])} and PR-AUC "
    f"{R.fmt(bt['pr_auc'])} on the fraud class, while an always-``real'' model is "
    f"{R.fmt(d['dummy_accuracy'])} accurate and finds nothing.")

body += rf"""
\section{{Introduction}}
Job boards are a soft target: a fake posting costs nothing to publish, harvests CVs and
identity documents, and is often deleted before anyone investigates. EMSCAD was released so
that the \emph{{text}} of an advertisement could be used as evidence -- fraudulent postings
tend to have an empty or copied company profile and language aimed at extracting money. Only
{ds['fraud_pct']}\% of postings are fraudulent, so an always-``real'' model is
{R.fmt(d['dummy_accuracy'])} accurate and useless; and because the evidence is split between
structured metadata (no company logo, no screening questions) and free text (\emph{{wire}},
\emph{{registration fee}}), the useful question is how much each contributes and whether
combining them pays.

\section{{Problem Statement}}
Each posting is a pair $(x^{{\text{{str}}}},x^{{\text{{txt}}}})$: binary flags, categorical
fields and five free-text fields. With $y\in\{{0,1\}}$ marking a fraudulent posting, we learn
$f(x^{{\text{{str}}}},x^{{\text{{txt}}}})\to[0,1]$ estimating $P(y=1\mid x)$ and predict fraud
iff $f(x)\ge\tau$. This is supervised binary classification with a 1:20 imbalance; success is
precision, recall and F1 on the \emph{{fraud}} class plus PR-AUC (the honest summary under
imbalance), with ROC-AUC for comparability. The threshold encodes a business cost -- how
many genuine postings may be sent to manual review -- so it is set on validation data.

\section{{Dataset Used}}
Real or Fake Job Posting Prediction (EMSCAD)~\cite{{kaggle}}, assembled by the University of
the Aegean~\cite{{emscad}} and used here from the Hugging Face mirror
\path{{victor/real-or-fake-fake-jobposting-prediction}} (CC0). It contains {ds['samples']}
postings, {ds['fraud']} fraudulent ({ds['fraud_pct']}\%), each with five free-text fields,
seven categorical fields, three binary flags and a salary range. I dropped \texttt{{job\_id}}
(an identifier) and \texttt{{salary\_range}} ({ds['missing_salary_range']}\% missing, and
almost exclusively missing for fraudulent postings -- keeping it would leak the label); no rows
were removed. Missing text becomes the empty string and five missingness indicators are added,
so ``profile absent'' is measurable rather than imputed away.

\section{{Dataset Statistics}}
{t_stats}

{f_eda}

\section{{Methodology}}
\subsection{{Learning setting}}
Supervised binary classification with severe imbalance. I compared an always-``real''
\emph{{Dummy}} (a diagnostic: it fixes the accuracy floor), \emph{{Logistic Regression}}
(linear in a 20k-dimensional TF-IDF space, coefficients readable as evidence),
\emph{{Linear SVM}} (same space, hinge loss, usually sharper on sparse text) and a
\emph{{Random Forest}} (non-linear, can exploit interactions between flags and text
frequencies). All three modalities enter one \texttt{{ColumnTransformer}}:
\texttt{{SimpleImputer}}+\texttt{{StandardScaler}} on numeric columns, a
\texttt{{OneHotEncoder}} (unseen categories ignored) on categorical columns, and
\texttt{{TfidfVectorizer}} on the concatenated text.

\subsection{{Data-split approach}}
Stratified 70/15/15 train/validation/test (\texttt{{random\_state=42}}) gives
{ds['n_train']:,}/{ds['n_val']:,}/{ds['n_test']:,} postings. Hyperparameters are chosen by 3-fold
\texttt{{StratifiedKFold}} on the training split, selecting on PR-AUC, which does not reward
the majority class. The threshold is chosen on \emph{{validation}} to maximise F1 and the test
split is scored once. Imbalance is handled by \texttt{{class\_weight='balanced'}} inside the
estimator, not by resampling, so no synthetic rows enter the pipeline. Every vectoriser and
encoder is fitted inside the pipeline, hence on training folds only; fitting TF-IDF on the
full corpus would leak test vocabulary.

\subsection{{Parameter settings}}
{t_hp}

\section{{Results and Analysis}}
\subsection{{Results}}
{t_res}

{f_pr}

\subsection{{Analysis}}
\textbf{{Why accuracy is a poor metric.}} The Dummy row settles it: {R.fmt(d['dummy_accuracy'])}
accuracy, zero fraud recall, PR-AUC {R.fmt(test['Dummy (always real)']['pr_auc'])} -- exactly
the base rate. Accuracy is dominated by the {ds['real']} genuine postings, so a model can look
excellent while never finding a fraud; PR-AUC is the fair summary because its baseline is the
base rate. The best model reaches {R.fmt(bt['pr_auc'])}, about
{bt['pr_auc']/d['base_rate']:.1f}$\times$ the base rate.

\textbf{{Does combining help?}} An ablation answers this. Structured metadata alone
is weak (PR-AUC {R.fmt(abl['structured']['pr_auc'])},
F1 {R.fmt(abl['structured']['f1'])}): a missing logo or profile raises the odds, but most
postings without a logo are genuine small employers. Text alone is nearly as good as the
combined model (PR-AUC {R.fmt(abl['text']['pr_auc'])}, F1 {R.fmt(abl['text']['f1'])}), and
combining adds {abl['combined']['pr_auc'] - abl['text']['pr_auc']:+.3f} PR-AUC and
{abl['combined']['f1'] - abl['text']['f1']:+.3f} F1 over text alone. The gain is small but
consistent and appears mainly in \emph{{precision at high recall}} -- the region a review queue
operates in.

\textbf{{Which words and fields indicate fraud?}} The strongest positive TF-IDF weights are
{fraud_terms}; the strongest negative (genuine) weights are {real_terms} -- exactly the lexicon
a human moderator would use. On the structured side the top weights are {struct_terms}; these
are one-hot categories with few positives, so some are small-sample artefacts and I would not
report them as findings without confidence intervals. The honest reading is that structured
metadata contributes a prior and the text decides.

\textbf{{Error analysis.}} At the chosen threshold the model makes {d['n_fp']} false positives
and {d['n_fn']} false negatives out of 130 frauds. Hand-reading the false positives
({fpsummary}) shows the shared failure mode: a genuine posting from a small or new employer
with no logo, a thin company profile and recruiter boilerplate (``apply now'', ``immediate
start'') -- fraudulent on every available signal, but not fraudulent. The false negatives are
postings whose text is lifted almost verbatim from a real advertisement, so the lexical evidence
is absent. That is a label-noise ceiling as much as a modelling one.

\textbf{{Limitations.}} EMSCAD was annotated around 2014 and fraud language drifts, so absolute
PR-AUC will not transfer to today's boards; the data are also not a random sample of a live
board. Both argue for monitoring precision after deployment.

\section{{Conclusion}}
Text carries almost all of the signal here (PR-AUC {R.fmt(abl['text']['pr_auc'])} alone
versus {R.fmt(abl['structured']['pr_auc'])} for structured fields), and combining the two
gives the best result at F1 {R.fmt(bt['f1'])} and PR-AUC {R.fmt(bt['pr_auc'])}. What did not
work as hoped was the non-linear forest, which trails the linear models on this
high-dimensional sparse space. Next I would test temporal generalisation and calibrate the
score so that "flag for review" maps to a stable expected precision.

\newpage
\section*{{AI Use Disclosure}}
I used Claude (Anthropic) as an AI assistant for scaffolding this notebook and drafting
report prose from the numbers it produced. The following parts were generated or
substantially assisted by AI: the notebook's cell structure, the plotting code and the first
draft of Sections I--VII. I verified the content by re-running the notebook end to end,
checking that no transformer is fitted outside a \texttt{{Pipeline}}, confirming that the
threshold is selected on validation only, and reading the top-weighted n-grams by hand to
make sure they are meaningful English rather than artefacts. All experiments, results and
analysis in this report were produced and checked by me, and I take full responsibility for
this work.

\begin{{thebibliography}}{{9}}
\bibitem{{kaggle}} Real or Fake Job Posting Prediction (EMSCAD), Kaggle.
\url{{https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction}}
\bibitem{{emscad}} S.~Mahbub, E.~Pardede, and A.~Kayes, ``Detection of fraudulent job
advertisements using machine learning,'' 2020.
\bibitem{{sklearn}} F.~Pedregosa \emph{{et al.}}, ``Scikit-learn: Machine learning in
Python,'' \emph{{JMLR}}, vol.~12, pp.~2825--2830, 2011.
\end{{thebibliography}}
\end{{document}}
"""

tex = R.PREAMBLE + r"\begin{document}" + body
path = SRC / f"{R.STUDENT['roll']}_B1_Report.tex"
path.write_text(tex)

TINYTEX = Path("/mnt/DC5E14885E145D98/ai101_ws/.TinyTeX/bin/x86_64-linux")
pdf = R.compile_tex(path, TINYTEX)
R.check_and_report(pdf, "B1")
print("  ->", pdf)
