# AI Use Disclosure

**Student:** Rizwan Ali · **Roll No.** 70177926 · **Section:** A
**Course:** AI-101, Department of Intelligent Systems, The University of Lahore
**Instructor:** Dr. Muhammad Awais
**Tasks attempted:** A1 (Breast Cancer Diagnosis), B1 (Fake Job Postings), C3 (Image Captioning)

This file contains the same disclosure that appears at the end of each of the three report
PDFs. The disclosure is per task, because the level of assistance differed between tasks.

---

## A1 — Breast Cancer Diagnosis from Cell-Nucleus Measurements

I used **Claude (Anthropic), accessed through the OpenCode coding agent, in November 2026** for
writing the notebook scaffolding, the plotting code, the LaTeX report template and a first draft
of the report prose. The following parts were generated or substantially assisted by AI: the
cell structure of `A1_notebook.ipynb`, the EDA and plotting cells, the `GridSearchCV` /
cross-validation harness, the threshold-study code, the `results/A1_results.json` writer, and the
first draft of Sections I–VII of the report.

I verified this content by re-running the notebook end to end in a clean kernel (no errors, no
stale state), by checking that every scaler and model lives inside a `Pipeline` fitted on the
training split only, by confirming that the decision threshold is selected on out-of-fold
training predictions rather than on the test set, and by reading the fitted coefficients and the
correlation clusters against the course material. All numbers in the report are read
programmatically from `results/A1_results.json`, which is written by the notebook itself, so the
PDF cannot disagree with the notebook. All experiments, results and analysis in this report were
produced and checked by me, and I take full responsibility for this work.

---

## B1 — Detecting Fraudulent Job Postings

I used **Claude (Anthropic), accessed through the OpenCode coding agent, in November 2026** for
writing the notebook scaffolding, the plotting code, the LaTeX report template and a first draft
of the report prose. The following parts were generated or substantially assisted by AI: the cell
structure of `B1_notebook.ipynb`, the `ColumnTransformer` definition, the TF-IDF / ablation
experiments, the plotting cells, and the first draft of Sections I–VII of the report.

I verified this content by re-running the notebook end to end (which reproduced every metric
exactly; only the wall-clock fitting times changed), by confirming that every vectoriser and
encoder is fitted inside a `Pipeline` — i.e. on training folds only — by checking that the
decision threshold is chosen on the validation split and the test set is scored once, and by
reading the highest-weighted n-grams by hand to confirm they are real words and not tokenisation
artefacts. I also re-checked the ablation myself by removing one feature block at a time. All
numbers in the report are read programmatically from `results/B1_results.json`, which is written
by the notebook itself. All experiments, results and analysis in this report were produced and
checked by me, and I take full responsibility for this work.

---

## C3 — Image Captioning with a CNN Encoder and an LSTM Decoder

I used **Claude (Anthropic), accessed through the OpenCode coding agent, in November 2026** for
writing the notebook scaffolding, the PyTorch training loop, the beam-search implementation, the
BLEU evaluation code, the plotting code, the LaTeX report template and a first draft of the report
prose. The following parts were generated or substantially assisted by AI: the cell structure of
`C3_notebook.ipynb`, the frozen-ResNet-50 feature cache, both decoder models, the greedy and beam
decoders, the BLEU/distinct-word evaluation, the sample-caption figure, and the first draft of
Sections I–VII of the report.

I verified this content by re-running the notebook end to end on CPU (hardware and wall-clock
time are reported in the paper), by confirming that the train/validation/test split is made by
**image**, so that none of an image's five captions can leak across splits, by checking that the
vocabulary is built from training captions only, by asserting that every caption is paired with
its own image's cached feature vector (a subset-position/global-row mismatch would silently turn
the captioner into a pure language model), by scoring the validation set a second time with the
image features permuted across images to confirm the decoder actually uses them, by confirming
that all five references are passed to the BLEU scorer, and by reading the generated captions of
the reported examples against their images. The loss curves, the BLEU table and the distinct-word
counts are produced by the notebook and stored in `results/C3_results.json`; the report reads that
file, so no number is typed by hand. All experiments, results and analysis in this report were
produced and checked by me, and I take full responsibility for this work.

---

## What the AI did **not** do

* It did not invent, estimate or "fill in" any metric. Every number in the three PDFs is
  generated by executing the corresponding notebook.
* It did not use any dataset other than the one named in each task. The Kaggle-hosted datasets
  were obtained from Hugging Face mirrors because I have no Kaggle API credentials; this is
  stated in each report's *Dataset Used* section.
* No part of this work was copied from, or shared with, another student.

## Reproducing everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# each task: execute the notebook, then build the report from its JSON
python tools/nb_build.py A1_Breast_Cancer/A1_cells.py A1_Breast_Cancer/A1_notebook.ipynb
python A1_Breast_Cancer/A1_report.py
```

Each task folder also has a `data/README.md` describing how to obtain the dataset.
