# AI-101 — Machine Learning Assignment

**Student:** Rizwan Ali · **Roll No.** 70177926 · **Section:** A
**Course:** Introduction to Artificial Intelligence (AI-101), Department of Intelligent Systems,
The University of Lahore · **Instructor:** Dr. Muhammad Awais

Three tasks, one from each pool. Every number in the three reports is produced by executing the
corresponding notebook — the reports are generated from the notebooks' JSON output, never typed
by hand.

| Task | Pool | Problem | Dataset | Headline result | Report |
|---|---|---|---|---|---|
| **A1** | A (numerical) | Breast-cancer diagnosis, binary classification | Breast Cancer Wisconsin (Diagnostic), 569 × 30 | Logistic regression: F1 0.951, ROC-AUC 0.996 at τ\*=0.505 chosen on out-of-fold predictions | [PDF](A1_Breast_Cancer/report/70177926_A1_Report.pdf) |
| **B1** | B (text) | Fraudulent job-posting detection, 1:20 imbalance | EMSCAD, 17,880 postings | Linear SVM on TF-IDF + structured fields: F1 0.868, PR-AUC 0.945 (base rate 0.048) | [PDF](B1_Fake_Job_Postings/report/70177926_B1_Report.pdf) |
| **C3** | C (images) | Image captioning with a CNN encoder + LSTM decoder | Flickr8k, 8,091 images × 5 captions | Frozen ResNet-50 + attention LSTM, beam $k{=}5$: BLEU-4 0.227 on 1,091 test images (merge decoder 0.202) | [PDF](C3_Image_Captioning/report/70177926_C3_Report.pdf) |

## Repository layout

```
AI101-ML-Assignment-70177926/
├── README.md                     ← this file
├── AI_DISCLOSURE.md              ← AI use, per task (same text as in each PDF)
├── requirements.txt              ← pinned, CPU-only environment
├── .gitignore
├── tools/
│   ├── nb_build.py               ← executes a cell script into a .ipynb (jupytext-style)
│   └── report_lib.py             ← LaTeX helpers; every PDF is built from the JSON
├── A1_Breast_Cancer/
│   ├── A1_cells.py               ← notebook source (plain Python, cell markers)
│   ├── A1_notebook.ipynb         ← executed notebook (all outputs saved)
│   ├── A1_report.py              ← builds the PDF from results/A1_results.json
│   ├── data/README.md            ← how to obtain the dataset
│   ├── figures/                  ← PNGs at print size (300 dpi)
│   ├── results/                  ← JSON + CSVs written by the notebook
│   └── report/70177926_A1_Report.pdf
├── B1_Fake_Job_Postings/         ← same structure
└── C3_Image_Captioning/          ← same structure
```

Datasets, cached CNN features and trained weights are **not** committed (see `.gitignore`);
each `data/README.md` explains how to fetch the data. No file in this repository is larger
than 50 MB.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU wheels of torch/torchvision

# 1. execute a notebook (writes figures/, results/)
python tools/nb_build.py A1_Breast_Cancer/A1_cells.py A1_Breast_Cancer/A1_notebook.ipynb
# 2. rebuild that task's report from the JSON the notebook just wrote
python A1_Breast_Cancer/A1_report.py
```

The task folders must be the working directory (the scripts resolve paths relative to
themselves, but the notebooks resolve `figures/` and `results/` from the current directory).

* **A1** needs no download — scikit-learn ships the dataset.
* **B1** downloads a 50 MB CSV from Hugging Face once (path adopted automatically; see
  `B1_Fake_Job_Postings/data/README.md`).
* **C3** downloads ~1.1 GB of Flickr8k once, then caches a frozen ResNet-50 feature pass
  (~1.6 GB, git-ignored). On CPU the full run takes about 70 minutes: 18 min for feature
  extraction, 9 min for the merge decoder, 34 min for the attention decoder, the rest for
  decoding and controls.

## Reproducibility notes

* Fixed seeds everywhere (`random_state=42`, `torch.manual_seed(0)`); re-running B1 reproduces
  every metric byte-for-byte — only wall-clock fitting times differ.
* Model/threshold choice never touches the test set: A1 selects on cross-validated ROC-AUC and
  sets its threshold on out-of-fold training predictions; B1 selects on CV PR-AUC and sets its
  threshold on the validation split. Each test set is scored once.
* C3 splits **by image**, so the five captions of an image always stay in one split, and builds
  its vocabulary from training captions only. Its BLEU implementation is asserted against
  `nltk.corpus_bleu` in the notebook.
* All three runs were executed on CPU (Intel i5-8265U, 8 threads, no GPU); the C3 report states
  the hardware and wall-clock training time, as the brief requires.

## AI use

Generative AI (Claude, via the OpenCode agent) was used for scaffolding, plotting code, LaTeX
plumbing and first-draft prose, and is disclosed per task in `AI_DISCLOSURE.md` and at the end
of every report, including how each part was verified. All experiments, results and analysis
were produced and checked by me; I take full responsibility for this work.
