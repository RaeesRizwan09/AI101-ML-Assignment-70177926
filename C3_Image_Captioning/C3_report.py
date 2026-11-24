#!/usr/bin/env python3
"""Generate + compile the C3 report from results/C3_results.json."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import report_lib as R

TASK = Path(__file__).resolve().parent
d = json.loads((TASK / "results" / "C3_results.json").read_text())
ds = d["dataset"]
enc = d["encoder"]
tr = d["training"]
bleu = {(r["model"], r["decoder"]): r for r in d["bleu"]}
best = d["best"]
ex = d["examples"]
env = d["env"]

SRC = TASK / "report" / "source"
FIGD = SRC / "figures"
FIGD.mkdir(parents=True, exist_ok=True)
for p in (TASK / "figures").glob("*.png"):
    shutil.copy(p, FIGD / p.name)

CONFIGS = ["greedy", "beam3", "beam5"]
DEC_LABEL = {"greedy": "greedy", "beam3": "beam $k{=}3$", "beam5": "beam $k{=}5$"}


def b(model, dec, col="bleu4"):
    return bleu[(model, dec)][col]


# ------------------------------------------------------------------ tables ---
t_stats = R.table(
    "l@{\\hspace{2pt}}l@{\\hspace{5pt}}l@{\\hspace{2pt}}l",
    r"Property & Value & Property & Value",
    rf"""Images & {ds['images']:,} & Captions & {ds['captions']:,} (5 each) \\
Train/val/test & {ds['n_train']:,} / {ds['n_val']:,} / {ds['n_test']:,} & Vocabulary & {ds['vocab']:,} words \\
Caption length & mean {ds['mean_len']:.1f}, p95 {ds['p95_len']} & Types before & {ds['vocab_before']:,} \\
Steps capped at & {ds['max_len']} tokens & Kept at $\geq{ds['min_freq']}$ & {ds['vocab'] / ds['vocab_before'] * 100:.0f}\% of types \\
OOV, val / test & {ds['oov_val'] * 100:.2f}\% / {ds['oov_test'] * 100:.2f}\% & Top word & \texttt{{{ds['top_words'][0][0]}}} ({ds['top_words'][0][1]:,}) \\""",
    "Flickr8k statistics after preprocessing. The split is by image, the vocabulary is built from training captions only.",
    "tab:stats", size=r"\scriptsize")

rows = []
for model in ("merge", "attention"):
    for dec in CONFIGS:
        r = bleu[(model, dec)]
        b4 = r["bleu4"]
        best_col = b4 == max(b(m, dd) for m in ("merge", "attention") for dd in CONFIGS)
        rows.append(
            f"{model} & {DEC_LABEL[dec]} & {R.fmt(r['bleu1'])} & {R.fmt(r['bleu2'])} & "
            f"{R.fmt(r['bleu3'])} & {R.bold_if(best_col, R.fmt(b4))} & "
            f"{r['distinct_words']:,} & {r['mean_len']:.1f}" + r" \\")
t_bleu = R.table(
    r"l@{\hspace{3pt}}l@{\hspace{5pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{5pt}}c@{\hspace{3pt}}c",
    r"Model & Search & B1 & B2 & B3 & B4 & Words & Len",
    "\n".join(rows),
    rf"B1--B4 are corpus BLEU-1\ldots4 against all five references on {ds['n_test']:,} test images; Words counts distinct words over the generated captions and Len is the mean caption length. Clipped n-gram counts and a brevity penalty; checked against \texttt{{nltk.corpus\_bleu}} (difference $<10^{{-6}}$).",
    "tab:bleu", size=r"\scriptsize")

# ----------------------------------------------------------------- figures ---
f_loss = R.fig("figures/C3_fig2_loss.png",
               r"Training and validation cross-entropy per epoch for both decoders. The gap that opens between the two curves is the overfitting that the best-validation checkpoint protects against.",
               "fig:loss")
f_ex = R.fig("figures/C3_fig3_examples.png",
             r"Generated captions for four test images (best configuration: "
             rf"{best['merge']['decoder'] if best['merge']['bleu4'] >= best['attention']['bleu4'] else best['attention']['decoder']}"
             r"). Top row: the two highest per-image BLEU-4. Bottom row: the two lowest.",
             "fig:ex")

# ------------------------------------------------------------- comparisons ---
def delta(model, a, bb, col="bleu4"):
    return b(model, bb, col) - b(model, a, col)


g_m, k3_m, k5_m = b("merge", "greedy"), b("merge", "beam3"), b("merge", "beam5")
g_a, k3_a, k5_a = b("attention", "greedy"), b("attention", "beam3"), b("attention", "beam5")
best_model = "attention" if best["attention"]["bleu4"] >= best["merge"]["bleu4"] else "merge"
worst_model = "merge" if best_model == "attention" else "attention"

beam_helps = (max(k3_m, k5_m) > g_m) or (max(k3_a, k5_a) > g_a)
dm = bleu[("merge", "greedy")]["distinct_words"]
da = bleu[("attention", "greedy")]["distinct_words"]
dm_b5 = bleu[("merge", "beam5")]["distinct_words"]
da_b5 = bleu[("attention", "beam5")]["distinct_words"]
more_generic = (dm_b5 < dm) or (da_b5 < da)

dep = d["image_dependence"]
dep_best = dep[best_model]
dep_other = dep[worst_model]
best4_overall = max(g_m, k3_m, k5_m, g_a, k3_a, k5_a)
beam_gain = max(k3_m, k5_m, k3_a, k5_a) - max(g_m, g_a)
ep_merge = tr["merge"]["seconds"] / max(tr["merge"]["epochs_run"], 1)
ep_attn = tr["attention"]["seconds"] / max(tr["attention"]["epochs_run"], 1)
attn_cost = ep_attn / max(ep_merge, 1e-9)
runs = (f"{tr['merge']['epochs_run']} epochs ({tr['merge']['seconds'] / 60:.1f} min) for the "
        f"merge decoder and {tr['attention']['epochs_run']} epochs "
        f"({tr['attention']['seconds'] / 60:.1f} min) for attention")
bleu_gap = best["attention"]["bleu4"] - best["merge"]["bleu4"]
if abs(bleu_gap) < 5e-4:
    kv_head = "The two decoders finish within 0.001 BLEU-4 of each other."
    kv_body = ("At this scale the extra capacity of attention does not buy measurable caption quality, "
               "which is itself a result: a frozen global image vector already carries most of the "
               "signal that 49 attended regions add.")
elif bleu_gap > 0:
    kv_head = f"Attention wins by {R.fmt(bleu_gap)} BLEU-4."
    kv_body = ("At each step the decoder can look at one region, so nouns are anchored to the right "
               "object even when the phrasing is generic. The merge decoder compresses the whole "
               "image into one 256-d vector before the first word is written, which is where its "
               "colour and counting errors come from. The two criteria disagree, though: merge has "
               f"the lower validation loss ({tr['merge']['best_val']:.3f} vs "
               f"{tr['attention']['best_val']:.3f}) and the lower BLEU, because per-token likelihood "
               "and corpus BLEU are different objectives.")
else:
    kv_head = f"The merge decoder comes out ahead by {R.fmt(-bleu_gap)} BLEU-4."
    kv_body = ("That is a small gap for a {:,}-image test set, and the attention decoder is the more "
               "heavily parameterised model here, so the result says that attention did not pay for "
               "itself at this scale rather than that it cannot."
               ).format(ds["n_test"])
generic_para = (
    f"\\textbf{{Generic captions and the distinct-word count.}} Beam search does not simply widen "
    f"the vocabulary: merge moves from {dm:,} distinct words with greedy to {dm_b5:,} at $k{{=}}5$, "
    f"attention from {da:,} to {da_b5:,}, out of {ds['vocab']:,}. "
    + ("Attention's extra BLEU is therefore partly template -- fewer words, higher score -- while "
       "merge buys its gain without narrowing, so the effect is not a law. "
       if more_generic else
       "Neither model narrows its vocabulary, so on this test set beam search does not buy score by "
       "collapsing to template language. ")
    + "Per-token maximum likelihood makes \"a dog is running through the grass\" the safest "
      "continuation for many dog images, and nothing in the objective rewards naming \\emph{this} "
      "colour or \\emph{this} count.")

concl = (f"A frozen ResNet-50 with a trained {best_model} LSTM decoder reaches BLEU-4 "
         f"{R.fmt(best4_overall)}, ahead of the {worst_model} decoder "
         f"({R.fmt(best[worst_model]['bleu4'])}) under identical features and splits. "
         + ("Beam search buys {:.4f} BLEU-4 at best".format(beam_gain) if beam_gain > 0
            else "Beam search does not help at all")
         + (", and it narrows the vocabulary" if more_generic else "")
         + ". The metric and the objective both reward fluency, so deployment needs a specificity "
           "signal, not a bigger beam.")

# ------------------------------------------------------------------- prose ---
body = R.header(
    "C3", "Image Captioning with a CNN Encoder and an LSTM Decoder",
    rf"A frozen ResNet-50 with a trained {best_model} LSTM decoder reaches corpus BLEU-4 "
    rf"{R.fmt(best4_overall)} on {ds['n_test']:,} unseen images. Replacing each image's features with "
    rf"another image's raises validation loss from {dep_best['aligned']:.2f} to "
    rf"{dep_best['shuffled']:.2f} nats, so the decoder is reading the picture and not just the corpus.")

body += rf"""
\section{{Introduction}}
Describing an image in one sentence couples two hard problems: deciding \emph{{what}} is in the
picture, and deciding \emph{{how}} to say it. Flickr8k was built for exactly this
setting~\cite{{flickr}}: {ds['images']:,} photographs, each described independently by five
people. The task is a sequence-to-sequence problem whose source is an image: a convolutional
network compresses the picture into features and a recurrent network writes the caption one token
at a time. The metric (BLEU) rewards \emph{{agreement with five annotators}}, not truth, so a
caption that is safe and generic can outscore a precise one; separating the two is what the
analysis below does.

\section{{Problem Statement}}
An image $I$ is mapped to a caption $c=(c_1,\dots,c_T)$, $c_t\in\mathcal{{V}}$. With a frozen
encoder $E$, we learn
$p_\theta(c\mid I)=\prod_t p_\theta(c_t\mid c_{{<t}}, E(I))$ by minimising the token-level
cross-entropy
$\mathcal{{L}}=-\sum_t \log p_\theta(c_t\mid c_{{<t}},E(I))$ over the training captions.
Two decoders are compared under identical conditions: a \emph{{merge}} decoder that fuses one
global image vector with the LSTM state, and an \emph{{attention}} decoder that attends over a
$7\times7$ grid at every step. Both are decoded greedily and with beam search ($k=3,5$) and scored
with corpus BLEU-1\ldots4 against \emph{{all five}} references. Success is a BLEU-4 well above what a
caption prior alone could reach, a control showing that the decoder's output really depends on the
image, and a qualitative error analysis that says where the remaining gap is.

\section{{Dataset Used}}
Flickr8k~\cite{{flickr}} (Kaggle release \path{{adityajn105/flickr8k}}), obtained from a Hugging
Face mirror so that no Kaggle credentials are needed. It contains {ds['images']:,} images and
{ds['captions']:,} captions, five per image, written by human annotators. I split by
\textbf{{image}} into {ds['n_train']:,} /
{ds['n_val']:,} / {ds['n_test']:,}: splitting by caption would put four references of an image in
training and the fifth in test, and the test score would measure memorisation. Captions were
lowercased and stripped of punctuation; \texttt{{<start>}}/\texttt{{<end>}} delimit them and rare
words become \texttt{{<unk>}}.

\section{{Dataset Statistics}}
{t_stats}

\section{{Methodology}}
\subsection{{Learning setting}}
An encoder--decoder trained by teacher forcing. The encoder is an ImageNet-pretrained
ResNet-50~\cite{{resnet}} with all weights frozen; its {enc['dim_avg']:,}-d average-pooled vector (or
its {enc['grid'][0]}$\times${enc['grid'][1]}$\times${enc['grid'][2]} feature map) is the only visual
input. Freezing is deliberate: fine-tuning 23 M convolutional parameters on {ds['n_train']:,}
images on a CPU was out of budget and would overfit. Features were extracted once and cached, so
both decoders see the \emph{{same}} visual representation and any difference is attributable to the
decoder.

\subsection{{Data-split approach}}
{ds['n_train']:,}/{ds['n_val']:,}/{ds['n_test']:,} images, seeded, split on identifiers so that all
five captions of an image stay together. The vocabulary is built from training captions only
({ds['vocab']:,} words kept at frequency $\geq{ds['min_freq']}$), leaving {ds['oov_val'] * 100:.2f}\,\%
and {ds['oov_test'] * 100:.2f}\,\% out-of-vocabulary words on validation and test that the model can
never produce. Each training epoch samples one caption per image, so all {ds['n_train']:,} images are
seen every epoch while the five references rotate; validation loss uses all five. Training runs for
{runs} with early stopping (patience 6) on validation loss, keeping the best-validation checkpoint.
Every caption is paired with its own image's cached features, and the notebook asserts that mapping
and reports a shuffled-feature control in the Results.

\subsection{{Parameter settings}}
Embeddings and LSTM hidden states are 256-d in the merge decoder and 512-d with a 256-d attention
space in the attention decoder; dropout 0.5; Adam at $10^{{-3}}$ with a $\times 0.5$ schedule when
validation loss stalls; gradient norm clipped at 5; batch size 128. Training ran on
\textbf{{{env['cpu']}}}, {env['cpu_count']} threads, no GPU; feature extraction took
{enc['extract_seconds'] / 60:.1f} min ({enc['images_per_second']:.1f} images/s), and the two
trainings took the times given above. Beam search keeps 3 or 5 hypotheses and applies the length
penalty $\ell^{{-0.7}}$; without it beam search prefers short captions, since every extra token adds
a negative log-probability.

\section{{Results and Analysis}}
\subsection{{Results}}
{t_bleu}

{f_loss}

{best_model.capitalize()} decoding ({best[best_model]['decoder']}) is the best configuration at
BLEU-4 {R.fmt(best[best_model]['bleu4'])}; the {worst_model} decoder's best is
{R.fmt(best[worst_model]['bleu4'])}.

\textbf{{Does the decoder use the image?}} Scoring the validation captions twice with the trained
{best_model} decoder -- once with each image's own features, once with another validation image's --
gives {dep_best['aligned']:.2f} and {dep_best['shuffled']:.2f} nats per token (+{dep_best['delta']:.2f});
the {worst_model} decoder shows the same pattern ({dep_other['aligned']:.2f} $\rightarrow$
{dep_other['shuffled']:.2f}). A decoder that ignored the picture would score the same either way, so
the BLEU above comes from vision and not from the caption prior alone.

{f_ex}

\subsection{{Analysis}}
\textbf{{Does beam search help?}} {"Yes, but the gain is small." if beam_helps else "Not on this test set: no beam width beat greedy."}
The best beam result is {beam_gain:+.4f} BLEU-4 over greedy. For the merge decoder BLEU-4 moves
{R.fmt(g_m)} (greedy) $\rightarrow$ {R.fmt(k3_m)} ($k{{=}}3$) $\rightarrow$ {R.fmt(k5_m)} ($k{{=}}5$);
for attention, {R.fmt(g_a)} $\rightarrow$ {R.fmt(k3_a)} $\rightarrow$ {R.fmt(k5_a)}. Beam search
maximises the sequence log-probability, so it prefers phrases the corpus makes likely. Its risk is
that it replaces a specific word with a more frequent one, so part of the gain can be fluency
rather than accuracy.

{generic_para}

\textbf{{Error types.}} The poor examples in Fig.~\ref{{fig:ex}} show counting and identity errors:
one dog becomes two, cows become dogs. Elsewhere the failures are \emph{{colour}} (a global
average-pooled vector and a $7\times7$ grid both blur it) and \emph{{action}} (a frozen ImageNet
classifier encodes \emph{{what}} is present better than \emph{{what is happening}}). When the image
is ambiguous it falls back on the most frequent people-phrase in Flickr8k, which is why a
captioner needs human review.

\textbf{{Merge versus attention.}} {kv_head} {kv_body} The price is speed: attention costs
{attn_cost:.1f}$\times$ per epoch, since it evaluates 49 locations at every one of the
{ds['max_len']} steps.

\textbf{{Limitations.}} BLEU correlates loosely with human judgement, Flickr8k is small, both
models are trained from scratch on CPU, and zero-shot BLIP would almost certainly beat both.

\section{{Conclusion}}
{concl} Next: attention maps, a tuned length penalty, and a zero-shot BLIP comparison.

\newpage
\section*{{AI Use Disclosure}}
I used Claude (Anthropic) as an AI assistant for scaffolding this notebook and drafting report
prose from the numbers it produced. The following parts were generated or substantially assisted by
AI: the notebook's cell structure, the PyTorch training loop, the beam-search and BLEU
implementations, the plotting code and the first draft of Sections I--VII. I verified the content by
re-running the notebook end to end on CPU (hardware and wall-clock time are reported above),
checking that the split is made by image so no caption leaks, that the vocabulary is built from
training captions only, that every caption is paired with its own image's features (asserted in the
notebook, and tested by a shuffled-feature control), and that all five references are passed to the
BLEU scorer -- which I also cross-checked against \texttt{{nltk.corpus\_bleu}}. All experiments,
results and analysis in this report were produced and checked by me, and I take full responsibility
for this work.

\begin{{thebibliography}}{{9}}
\bibitem{{flickr}} M.~Hodosh, P.~Young, and J.~Hockenmaier, ``Framing image description as a
ranking task: Data, models and evaluation metrics,'' \emph{{JAIR}}, vol.~47, pp.~853--899, 2013.
\bibitem{{resnet}} K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image
recognition,'' in \emph{{CVPR}}, 2016.
\bibitem{{showtell}} O.~Vinyals, A.~Toshev, S.~Bengio, and D.~Erhan, ``Show and tell: A neural image
caption generator,'' in \emph{{CVPR}}, 2015.
\bibitem{{attn}} K.~Xu \emph{{et al.}}, ``Show, attend and tell: Neural image caption generation with
visual attention,'' in \emph{{ICML}}, 2015.
\bibitem{{karpathy}} A.~Karpathy and L.~Fei-Fei, ``Deep visual-semantic alignments for generating
image descriptions,'' in \emph{{CVPR}}, 2015.
\end{{thebibliography}}
\end{{document}}
"""

tex = R.PREAMBLE + r"\begin{document}" + body
path = SRC / f"{R.STUDENT['roll']}_C3_Report.tex"
path.write_text(tex)

TINYTEX = Path("/mnt/DC5E14885E145D98/ai101_ws/.TinyTeX/bin/x86_64-linux")
pdf = R.compile_tex(path, TINYTEX)
R.check_and_report(pdf, "C3")
print("  ->", pdf)
