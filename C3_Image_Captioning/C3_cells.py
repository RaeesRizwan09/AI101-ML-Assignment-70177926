# %% [markdown]
# # Task C3 — Image Captioning with a CNN Encoder and an LSTM Decoder
#
# **Course:** AI-101, Department of Intelligent Systems, The University of Lahore
# **Student:** Rizwan Ali (Roll No. 70177926) · Section C
# **Repository:** https://github.com/RaeesRizwan09/AI101-ML-Assignment-70177926
#
# A frozen ImageNet ResNet-50 encodes an image; an LSTM decodes a caption word by word.
# Two decoders are trained on identical splits and vocabulary:
#
# 1. **Merge** architecture — image feature → dense layer, caption prefix → embedding → LSTM,
#    concatenated and mapped to a softmax over the vocabulary.
# 2. **Attention** architecture (*Show, Attend and Tell*) — the same LSTM, but at every step it
#    attends over the 7×7 spatial grid instead of using one global vector.
#
# Both are evaluated with corpus BLEU-1…4 against **all five** references, under greedy and
# beam search (k = 3, 5). Every number in the report is written to `results/C3_results.json`
# by this notebook.

# %% [markdown]
# ## 1. Environment
#
# The assignment allows free Colab/Kaggle GPUs but asks for the hardware and training time to be
# reported. This run was executed on CPU only, so those numbers are recorded here and quoted in
# the report.

# %%
import gc
import json
import os
import platform
import random
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image

torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)
torch.backends.mkldnn.enabled = True

ROOT = Path.cwd()
FIG, RES, CACHE = ROOT / "figures", ROOT / "results", ROOT / "data"
for p in (FIG, RES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
PAD, START, END, UNK = 0, 1, 2, 3
MIN_FREQ = 5


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


ENV = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "platform": platform.platform(),
    "cpu": cpu_model(),
    "cpu_count": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
print(json.dumps(ENV, indent=2))
DEVICE = torch.device(ENV["device"])

# %% [markdown]
# ## 2. Dataset
#
# **Flickr8k** — 8,091 photographs from Flickr, each described by five independent human captions
# (40,455 captions). The Kaggle release `adityajn105/flickr8k` is mirrored on Hugging Face; the
# mirror used here stores the JPEG bytes and the five captions in Parquet shards, so no Kaggle
# account is needed. `data/README.md` explains how to fetch it.
#
# The bytes are written once to a JPEG cache on disk. That keeps memory flat (the whole corpus is
# ~1.1 GB of images) and makes the encoder pass reproducible.

# %%
CANDIDATES = [Path(os.environ.get("AI101_DATA", "")), ROOT / "data", ROOT.parent / "data",
              Path("/mnt/DC5E14885E145D98/ai101_ws/data")]


def parquet_groups():
    """Every distinct set of parquet shards that looks like a Flickr8k release."""
    groups = []
    for base in CANDIDATES:
        if not base or not base.exists():
            continue
        for pattern in ("flickr8k_8091/data/*.parquet", "flickr8k/data/*.parquet",
                        "flickr8k_hf/data/*.parquet", "*/data/*.parquet"):
            hits = tuple(sorted(base.glob(pattern)))
            if hits and hits not in groups:
                groups.append(hits)
    return groups


def find_parquets(expect=8091):
    """Pick the shard set holding the full corpus (8,091 images), else the largest one."""
    groups = parquet_groups()
    if not groups:
        raise FileNotFoundError(
            "No Flickr8k parquet files found - see C3_Image_Captioning/data/README.md")
    sizes = [sum(pq.ParquetFile(p).metadata.num_rows for p in g) for g in groups]
    print("candidate mirrors:", [(str(g[0].parent), n) for g, n in zip(groups, sizes)])
    for g, n in zip(groups, sizes):
        if n == expect:
            return g
    best = groups[int(np.argmax(sizes))]
    print(f"WARNING: no mirror holds {expect} images; using the largest ({max(sizes)})")
    return best


PARQUETS = find_parquets()
print(len(PARQUETS), "parquet shard(s):")
for p in PARQUETS:
    print("  ", p)

IMG_DIR = CACHE / "flickr8k_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def iter_rows(paths):
    """Yield (filename, jpeg bytes, five captions) for every image, whatever the mirror layout.

    Two mirrors are supported:
      * one row per image with a `captions` list  (tsystems/flickr8k, 8,091 rows)
      * one row per caption set with `caption_0..4` columns (jxie/flickr8k, 8,000 images)
    """
    for p in paths:
        names = pq.ParquetFile(p).schema_arrow.names
        if "captions" in names:
            t = pd.read_parquet(p, columns=["image", "captions"])
            for img, caps in zip(t["image"], t["captions"]):
                yield Path(img["path"]).name, img["bytes"], list(caps)
        else:
            cols = ["image"] + [f"caption_{i}" for i in range(5)]
            t = pd.read_parquet(p, columns=[c for c in cols if c in names])
            for row in t.itertuples(index=False):
                yield (Path(row.image["path"]).name, row.image["bytes"],
                       [getattr(row, f"caption_{i}", "") for i in range(5)])


# %%
t0 = time.time()
records = []
for name, data, caps in iter_rows(PARQUETS):
    dst = IMG_DIR / name
    if not dst.exists():
        dst.write_bytes(data)
    caps = [c if isinstance(c, str) else "" for c in caps][:5]
    records.append((name, *(caps + [""] * (5 - len(caps)))))
img_t = time.time() - t0

cap = pd.DataFrame(records, columns=["img", "c0", "c1", "c2", "c3", "c4"])
cap = cap.drop_duplicates("img").reset_index(drop=True)
print(f"rows read: {len(records):,}  ->  unique images: {len(cap):,}")
print(f"JPEG cache written in {img_t:.1f}s  ->  {IMG_DIR}")
print(cap.head(3).to_string())

# %% [markdown]
# ## 3. Split by image
#
# The split is made on **image identifiers**, never on captions: if it were made on captions, four
# of an image's five captions could land in training and the fifth in test, and the test BLEU would
# measure memorisation. 6,000 / 1,000 / 1,091 images, fixed seed.

# %%
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(cap))
n_tr, n_va = 6000, 1000
split = np.empty(len(cap), dtype=object)
split[perm[:n_tr]] = "train"
split[perm[n_tr:n_tr + n_va]] = "val"
split[perm[n_tr + n_va:]] = "test"
cap["split"] = split

# invariants that would catch a caption-level leak
assert cap.img.is_unique, "duplicate images in the corpus"
assert (cap.split == "train").sum() == n_tr
assert (cap.split == "val").sum() == n_va
assert (cap.split == "test").sum() == len(cap) - n_tr - n_va
assert not set(cap.loc[cap.split == "train", "img"]) & set(cap.loc[cap.split == "test", "img"])

print(cap.split.value_counts().reindex(["train", "val", "test"]).to_string())
print("images in more than one split:", int(cap.groupby("img").split.nunique().gt(1).sum()))
cap.to_csv(RES / "C3_split.csv", index=False)

# %% [markdown]
# ## 4. Statistics and preprocessing
#
# Captions are lowercased, punctuation is stripped and `eos` tokens are added. The vocabulary is
# built from **training captions only** and cut at frequency ≥ 5; rarer words become `<unk>`.
# Anything else would leak test vocabulary into the model.

# %%
CAPS = ["c0", "c1", "c2", "c3", "c4"]


def tokenize(text):
    text = text.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).split()


tokens = {s: [tokenize(t) for t in cap.loc[cap.split == s, CAPS].values.ravel()] for s in
          ("train", "val", "test")}
lens = {s: np.array([len(t) for t in toks]) for s, toks in tokens.items()}

cnt = Counter(w for t in tokens["train"] for w in t)
vocab_words = sorted([w for w, c in cnt.items() if c >= MIN_FREQ], key=lambda w: -cnt[w])
itos = ["<pad>", "<start>", "<end>", "<unk>"] + vocab_words
stoi = {w: i for i, w in enumerate(itos)}
VOCAB = len(itos)
MAXLEN = int(min(25, max(8, np.percentile(lens["train"], 99))))
print(f"tokens train/val/test : {len(tokens['train']):,} / {len(tokens['val']):,} / {len(tokens['test']):,}")
print(f"caption length (train): mean {lens['train'].mean():.1f}  median {np.median(lens['train']):.0f}"
      f"  p95 {np.percentile(lens['train'], 95):.0f}  max {lens['train'].max()}")
print(f"unique words before min_freq : {len(cnt):,}")
print(f"vocabulary after min_freq={MIN_FREQ} : {VOCAB:,}  ({(1 - VOCAB / len(cnt)) * 100:.1f}% of types dropped)")
for s in ("val", "test"):
    oov = np.mean([w not in stoi for t in tokens[s] for w in t])
    print(f"OOV rate on {s} captions : {oov * 100:.2f}%")
print("most frequent words:", ", ".join(f"{w}({c})" for w, c in cnt.most_common(12)))

# %%
fig = plt.figure(figsize=(3.45, 2.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=.42)
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])
for s, col in [("train", "#4393c3"), ("test", "#b2182b")]:
    ax0.hist(lens[s], bins=np.arange(0, 40, 2), density=True, histtype="step", lw=1.0,
             color=col, label=s)
ax0.set_xlabel("caption length (words)", fontsize=6.5)
ax0.set_ylabel("density", fontsize=6.5)
ax0.set_title("(a) caption length", fontsize=7)
ax0.legend(fontsize=5.6, frameon=False, loc="upper right")

top = cnt.most_common(15)
ax1.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color="#4393c3",
         edgecolor="black", lw=.3)
ax1.set_xlabel("occurrences in training captions", fontsize=6.5)
ax1.set_title("(b) most frequent words", fontsize=7)
for a in (ax0, ax1):
    a.tick_params(labelsize=5.8)
ax1.tick_params(labelsize=6.0)
fig.savefig(FIG / "C3_fig1_stats.png"); plt.close(fig)

# %%
def encode(toks):
    ids = [START] + [stoi.get(w, UNK) for w in toks[:MAXLEN - 2]] + [END]
    return ids


class CaptionPairs(Dataset):
    """Training pairs.

    `one_per_image=True` exposes one **randomly chosen** caption per image per epoch (6,000 rows),
    which is the usual compromise: every image is still visited every epoch, and over the training
    run the model sees several of its five references. `one_per_image=False` exposes all five
    (5,000 rows on the 1,000 validation images) and is what the validation loss uses.
    """

    def __init__(self, frame, one_per_image=False):
        self.img, self.seqs = [], []
        for _, r in frame.iterrows():
            self.img.append(r.img)
            self.seqs.append([encode(tokenize(r[c])) for c in CAPS])
        self.one = one_per_image
        # The feature cache is indexed by row of `cap`, not by position in this subset.
        # Keeping the original labels is what pairs a caption with *its own* image.
        self.gidx = np.asarray(frame.index, dtype=np.int64)
        assert len(self.gidx) == len(self.img) and len(np.unique(self.gidx)) == len(self.gidx)
        self.flat = None if one_per_image else [(i, j) for i, s in enumerate(self.seqs)
                                               for j in range(len(s))]

    def __len__(self):
        return len(self.img) if self.one else len(self.flat)

    def __getitem__(self, k):
        i, j = (k, random.randrange(len(self.seqs[k]))) if self.one else self.flat[k]
        return self.gidx[i], np.asarray(self.seqs[i][j], dtype=np.int64)


def collate(batch):
    imgs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    width = max(len(b[1]) for b in batch)
    out = torch.zeros((len(batch), width), dtype=torch.long)   # PAD = 0
    for i, (_, s) in enumerate(batch):
        out[i, :len(s)] = torch.from_numpy(s)
    return imgs, out


ds_tr = CaptionPairs(cap[cap.split == "train"], one_per_image=True)
ds_va = CaptionPairs(cap[cap.split == "val"], one_per_image=False)
# the features are cached per row of `cap`; a subset position must map to its global row,
# otherwise each caption is trained against a different image and the decoder learns a
# pure language model (this assertion is what caught that bug).
assert np.array_equal(np.sort(ds_tr.gidx), np.where(cap.split.values == "train")[0])
assert np.array_equal(np.sort(ds_va.gidx), np.where(cap.split.values == "val")[0])
_gi, _ = ds_tr[0]
assert cap.img.iloc[_gi] == ds_tr.img[0], "caption paired with the wrong image"
print("training rows/epoch:", len(ds_tr), "(1 sampled caption x 6,000 images)")
print("validation rows:", len(ds_va), "| max length", MAXLEN)
print(f"alignment check: subset row 0 -> cache row {_gi} -> {cap.img.iloc[_gi]}")
print("example:", " ".join(itos[i] for i in ds_tr.seqs[0][0]))

# %% [markdown]
# ## 5. Encoder — frozen ImageNet ResNet-50, cached
#
# The CNN is never fine-tuned: gradients through 23 M parameters on 6,000 images on a CPU would cost
# hours and would overfit. Features are extracted **once** and reused by both decoders, which also
# makes the two models comparable — they see exactly the same visual representation.
#
# * Model 1 uses the 2,048-d global average-pooled vector.
# * Model 2 keeps the 7×7×2,048 last-conv feature map (49 locations) for attention.

# %%
weights = ResNet50_Weights.IMAGENET1K_V2
cnn = resnet50(weights=weights).to(DEVICE).eval()
for p in cnn.parameters():
    p.requires_grad_(False)

prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract(paths, batch=32):
    """Fill (N, 2048) average-pooled and (N, 49, 2048) spatial arrays in place."""
    avg = np.zeros((len(paths), 2048), dtype=np.float16)
    spa = np.zeros((len(paths), 49, 2048), dtype=np.float16)
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.stack([prep(Image.open(p).convert("RGB")) for p in chunk]).to(DEVICE)
        x = cnn.maxpool(cnn.relu(cnn.bn1(cnn.conv1(x))))
        x = cnn.layer2(cnn.layer1(x))
        fmap = cnn.layer4(cnn.layer3(x))                  # (B, 2048, 7, 7)
        spa[i:i + len(chunk)] = fmap.flatten(2).transpose(1, 2).cpu().numpy()
        avg[i:i + len(chunk)] = cnn.avgpool(fmap).flatten(1).cpu().numpy()
        if (i // batch) % 40 == 0:
            print(f"  {i + len(chunk):5d}/{len(paths)} images", flush=True)
    return avg, spa


# %%
paths = [str(IMG_DIR / n) for n in cap.img]
t0 = time.time()
FEAT_AVG, FEAT_SPA = extract(paths)
feat_t = time.time() - t0
np.save(CACHE / "C3_feat_avg.npy", FEAT_AVG)
np.save(CACHE / "C3_feat_spatial.npy", FEAT_SPA)
print(f"\nfeatures: avg {FEAT_AVG.shape} {FEAT_AVG.dtype}, spatial {FEAT_SPA.shape} {FEAT_SPA.dtype}")
print(f"extraction time: {feat_t:.1f}s  ({len(paths) / feat_t:.1f} images/s, CPU)")
del cnn, FEAT_AVG, FEAT_SPA
gc.collect()
# re-open as memory maps: the spatial cache is 1.6 GB and this machine has 8 GB of RAM
FEAT_AVG = np.load(CACHE / "C3_feat_avg.npy", mmap_mode="r")
FEAT_SPA = np.load(CACHE / "C3_feat_spatial.npy", mmap_mode="r")

# %% [markdown]
# ## 6. Model 1 — merge architecture
#
# `image → Dense(256) → ReLU` is the *image embedding*; the caption prefix goes through
# `Embedding → LSTM(256)`. The two are concatenated at every timestep and mapped to the vocabulary
# by one hidden layer and a softmax. This is the "merge" variant of injecting the image: the
# decoder is free to use the LSTM state, and the image is added late, just before prediction.

# %%
class MergeCaptioner(nn.Module):
    def __init__(self, vocab_size=VOCAB, img_dim=2048, emb=256, hid=256, drop=.5):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, emb), nn.ReLU(), nn.Dropout(drop))
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(emb + hid, hid), nn.ReLU(), nn.Dropout(drop))
        self.out = nn.Linear(hid, vocab_size)

    def forward(self, img, caps):
        """Teacher forcing: caption prefix (with <start>) -> logits for the next word."""
        h, _ = self.lstm(self.embed(caps[:, :-1]))
        img_e = self.img_proj(img).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.out(self.fuse(torch.cat([img_e, h], dim=-1)))

    def logits_for_prefix(self, img, prefix):
        """Inference: logits for the token *after* `prefix` (batch of partial sequences)."""
        h, _ = self.lstm(self.embed(prefix))
        z = torch.cat([self.img_proj(img), h[:, -1]], dim=-1)
        return self.out(self.fuse(z))


# %%
class SpatialCaptioner(nn.Module):
    """Show, Attend and Tell: soft attention over the 7x7 annotation grid at every step."""

    def __init__(self, vocab_size=VOCAB, ctx_dim=2048, emb=256, hid=512, att=256, drop=.5):
        super().__init__()
        self.proj = nn.Linear(ctx_dim, att)            # 49 x 2048 -> 49 x 256
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.init_h = nn.Linear(att, hid)
        self.init_c = nn.Linear(att, hid)
        self.lstm = nn.LSTMCell(emb + att, hid)
        self.att_h = nn.Linear(hid, att)               # what the decoder is looking for
        self.att_s = nn.Linear(att, 1)                 # score per spatial location
        self.gate = nn.Linear(hid, att)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid, vocab_size)

    def _annotations(self, ctx):
        a = self.proj(ctx)                              # (B, 49, att)
        return a, self.init_h(a.mean(1)), self.init_c(a.mean(1))

    def _step(self, tok, h, c, ann):
        alphas = self.att_s(torch.tanh(ann + self.att_h(h).unsqueeze(1))).softmax(1)
        ctx = (ann * alphas).sum(1)                     # (B, att)
        gated = torch.sigmoid(self.gate(h)) * ctx
        h, c = self.lstm(torch.cat([self.embed(tok), gated], -1), (h, c))
        return h, c, self.out(self.drop(h)), alphas

    def forward(self, ctx, caps):
        ann, h, c = self._annotations(ctx)
        outs = []
        for t in range(caps.size(1) - 1):
            h, c, logit, _ = self._step(caps[:, t], h, c, ann)
            outs.append(logit)
        return torch.stack(outs, 1)

    def logits_for_prefix(self, ctx, prefix):
        ann, h, c = self._annotations(ctx)
        for t in range(prefix.size(1)):
            h, c, logit, _ = self._step(prefix[:, t], h, c, ann)
        return logit


def make_feats(kind, idx, device=DEVICE):
    src = FEAT_AVG if kind == "avg" else FEAT_SPA
    return torch.from_numpy(src[idx]).float().to(device)

# %%
def train_model(model, kind, epochs=50, batch=128, lr=1e-3, patience=6):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=2)
    gen = torch.Generator().manual_seed(SEED)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, collate_fn=collate, drop_last=True,
                       generator=gen)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False, collate_fn=collate)
    hist = {"train": [], "val": []}
    best, best_state, bad, t0 = np.inf, None, 0, time.time()

    for ep in range(1, epochs + 1):
        model.train(); tot = n = 0
        for img_i, caps in dl_tr:
            logits = model(make_feats(kind, img_i), caps)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), caps[:, 1:].reshape(-1),
                                   ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * caps.size(0); n += caps.size(0)
        hist["train"].append(tot / n)

        model.eval(); tot = n = 0
        with torch.no_grad():
            for img_i, caps in dl_va:
                logits = model(make_feats(kind, img_i), caps)
                tot += F.cross_entropy(logits.reshape(-1, VOCAB), caps[:, 1:].reshape(-1),
                                       ignore_index=PAD, reduction="sum").item()
                n += (caps[:, 1:] != PAD).sum().item()
        hist["val"].append(tot / n)
        sched.step(hist["val"][-1])
        star = ""
        if hist["val"][-1] < best - 1e-4:
            best, bad = hist["val"][-1], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            star = " *"
        else:
            bad += 1
        print(f"epoch {ep:2d}  train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}{star}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            print(f"early stop at epoch {ep} (no val improvement for {patience} epochs)")
            break
    model.load_state_dict(best_state)
    hist["seconds"] = round(time.time() - t0, 1)
    hist["best_val"] = round(float(best), 5)
    hist["epochs_run"] = len(hist["train"])
    return model, hist

# %% [markdown]
# ## 7. Model 2 — attention over the spatial grid
#
# Identical vocabulary and splits, but the decoder is an `LSTMCell` that at each step scores the 49
# locations, forms a weighted context vector and gates it into the cell input. More parameters, more
# compute per step — the report compares whether it pays.

# %%
merge = MergeCaptioner().to(DEVICE)
print("merge parameters:", f"{sum(p.numel() for p in merge.parameters()):,}")
merge, hist_merge = train_model(merge, "avg")

# %%
spatial = SpatialCaptioner().to(DEVICE)
print("attention parameters:", f"{sum(p.numel() for p in spatial.parameters()):,}")
spatial, hist_spatial = train_model(spatial, "spa", epochs=30)

# %% [markdown]
# ## 8. Decoding — greedy and beam search
#
# Greedy takes the arg-max token at every step, which is locally optimal but can paint itself into a
# corner. Beam search keeps the k best partial captions, adds the log-probabilities and rescales by
# a length penalty `len^-0.7` (without it, beams prefer short captions because every token adds a
# negative log-probability).
#
# Both decoders share one interface, `logits_for_prefix(image_features, prefix)`, so the search code
# is written once and applied to both models.

# %%
@torch.no_grad()
def greedy_decode(model, kind, idx, max_len=MAXLEN, batch=64):
    model.eval(); out = [[] for _ in idx]
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        cur = torch.full((len(chunk), 1), START, dtype=torch.long, device=DEVICE)
        done = torch.zeros(len(chunk), dtype=torch.bool)
        for _ in range(max_len):
            nxt = model.logits_for_prefix(make_feats(kind, chunk), cur).argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], -1)
            done |= nxt.squeeze(1).eq(END)
            if bool(done.all()):
                break
        for row, seq in enumerate(cur.cpu().tolist()):
            toks = []
            for t in seq[1:]:
                if t == END:
                    break
                toks.append(itos[t])
            out[s + row] = toks
    return out


@torch.no_grad()
def beam_decode(model, kind, idx, k=3, max_len=MAXLEN, batch=32, alpha=.7):
    model.eval(); out = [None] * len(idx)
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        B = len(chunk)
        prefixes = torch.full((B * k, 1), START, dtype=torch.long, device=DEVICE)
        scores = torch.full((B * k,), -1e9, device=DEVICE)
        scores[::k] = 0.0                                  # only beam 0 is alive at step 0
        finished = torch.zeros(B * k, dtype=torch.bool, device=DEVICE)
        for step in range(max_len):
            lp = F.log_softmax(model.logits_for_prefix(make_feats(kind, np.repeat(chunk, k)),
                                                       prefixes), -1)
            # a finished hypothesis may only emit END, at zero cost, so it stops growing
            lp = lp.masked_fill(finished.unsqueeze(1), -1e9)
            lp[finished, END] = 0.0
            cand = (scores.unsqueeze(1) + lp).view(B, k * VOCAB)
            top, arg = cand.topk(k, dim=1)
            beam = arg // VOCAB
            toks = arg % VOCAB
            T = prefixes.size(1)
            prefixes = torch.cat([prefixes.view(B, k, T).gather(
                1, beam.unsqueeze(-1).expand(-1, -1, T)), toks.unsqueeze(-1)], -1).view(B * k, T + 1)
            scores = top.reshape(-1)
            finished = (finished.view(B, k).gather(1, beam).reshape(-1)
                        | toks.reshape(-1).eq(END))
            if bool(finished.view(B, k).all(dim=1).all()):
                break
        prefixes = prefixes.view(B, k, -1).cpu().tolist()
        for b, hyp in enumerate(prefixes):
            best, best_sc = None, -np.inf
            for j, seq in enumerate(hyp):
                toks = []
                for t in seq[1:]:
                    if t == END:
                        break
                    toks.append(itos[t])
                if not toks:
                    continue
                score = scores.view(B, k)[b, j].item() / (len(toks) ** alpha)
                if score > best_sc:
                    best, best_sc = toks, score
            out[s + b] = best or ["<unk>"]
    return out

# %%
test_idx = np.where(cap.split.values == "test")[0]
print("test images:", len(test_idx))

# %% [markdown]
# ## 9. Evaluation — corpus BLEU-1…4 against all five references
#
# BLEU is computed with clipped n-gram counts and a brevity penalty over the whole corpus — the
# precisions are summed over all 1,091 captions *before* dividing, which is the corpus-level
# definition rather than an average of sentence scores. Implementing it here keeps the metric
# visible; the next cell checks it against `nltk.corpus_bleu` so a bug cannot silently flatter
# the model.

# %%
def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(refs, hyps, n_max=4):
    """refs[i] = list of 5 reference token lists; hyps[i] = generated token list."""
    prec = {}
    for n in range(1, n_max + 1):
        num = den = 0
        for r, h in zip(refs, hyps):
            hc = ngrams(h, n)
            max_ref = Counter()
            for ref in r:
                for g, c in ngrams(ref, n).items():
                    max_ref[g] = max(max_ref[g], c)
            num += sum(min(c, max_ref[g]) for g, c in hc.items())
            den += max(1, len(h) - n + 1)      # nltk's convention for zero-match denominators
        prec[n] = num / den if den else 0.0
    hyp_len = sum(len(h) for h in hyps)
    # brevity penalty: per image, the reference length closest to the hypothesis, ties to the
    # shorter reference. With these two conventions (and the denominator above) the corpus score
    # reproduces nltk.corpus_bleu exactly -- which the next cell asserts rather than assumes.
    ref_len = sum(min((len(ref) for ref in r), key=lambda L: (abs(L - len(h)), L))
                  for r, h in zip(refs, hyps))
    bp = 1.0 if hyp_len > ref_len else (np.exp(1 - ref_len / hyp_len) if hyp_len else 0.0)
    out = {}
    for n in range(1, n_max + 1):
        ps = [prec[j] for j in range(1, n + 1)]
        out[f"BLEU-{n}"] = 0.0 if min(ps) <= 0 else float(bp * np.exp(np.mean(np.log(ps))))
    return out


REFS = [[tokenize(cap.loc[i, c]) for c in CAPS] for i in test_idx]
print("references for image 0:", REFS[0][:2])

# %%
CONFIGS = {}
for name, model, kind in [("merge", merge, "avg"), ("attention", spatial, "spa")]:
    t0 = time.time()
    CONFIGS[(name, "greedy")] = greedy_decode(model, kind, test_idx)
    greedy_t = time.time() - t0
    for k in (3, 5):
        t0 = time.time()
        CONFIGS[(name, f"beam{k}")] = beam_decode(model, kind, test_idx, k=k)
        print(f"{name:9s} beam{k} decoded 1000 images in {time.time() - t0:.1f}s")
    print(f"{name:9s} greedy in {greedy_t:.1f}s")

# %%
import nltk
from nltk.translate.bleu_score import corpus_bleu as nltk_bleu

rows = []
for (name, dec), hyps in CONFIGS.items():
    b = corpus_bleu(REFS, hyps)
    nb4 = nltk_bleu(REFS, hyps)          # nltk default weights (0.25 x4), method0, no smoothing
    distinct = len({w for h in hyps for w in h})
    total = sum(len(h) for h in hyps)
    rows.append({"model": name, "decoder": dec,
                 **{f"bleu{i + 1}": round(b[f"BLEU-{i + 1}"], 4) for i in range(4)},
                 "distinct_words": distinct, "mean_len": round(total / len(hyps), 2),
                 "nltk_diff": round(abs(nb4 - b["BLEU-4"]), 8)})
BLEU = pd.DataFrame(rows)
print(BLEU.to_string(index=False))
assert BLEU.nltk_diff.max() < 1e-6, "own BLEU disagrees with nltk - fix before reporting"

# %% [markdown]
# ## 10. Results
#
# Loss curves, the BLEU table and four qualitative examples (two best, two worst by per-image
# smoothed BLEU-4 of the best configuration).

# %%
fig = plt.figure(figsize=(3.45, 2.1))
gs = fig.add_gridspec(1, 2, wspace=.38)
for ax, hist, title in [(fig.add_subplot(gs[0]), hist_merge, "(a) merge"),
                        (fig.add_subplot(gs[1]), hist_spatial, "(b) attention")]:
    ep = np.arange(1, len(hist["train"]) + 1)
    ax.plot(ep, hist["train"], "o-", ms=2.2, lw=1.0, color="#4393c3", label="train")
    ax.plot(ep, hist["val"], "s--", ms=2.2, lw=1.0, color="#b2182b", label="val")
    ax.set_xlabel("epoch", fontsize=6.5); ax.set_ylabel("cross-entropy (nats)", fontsize=6.5)
    ax.set_title(title, fontsize=7); ax.tick_params(labelsize=5.8)
    ax.legend(fontsize=5.4, frameon=False)
fig.savefig(FIG / "C3_fig2_loss.png"); plt.close(fig)

# %%
# Diagnostic: how much does the decoder actually use the image? Score the validation captions
# twice - once with each image's own features, once with the features of a different validation
# image - and look at the gap. A decoder that ignored the image would show no gap at all.
@torch.no_grad()
def val_loss(model, kind, permute=False):
    model.eval()
    ids = np.sort(ds_va.gidx)
    other = dict(zip(ids, np.roll(ids, len(ids) // 2)))   # no image keeps its own features
    tot = n = 0
    for img_i, caps in DataLoader(ds_va, batch_size=256, shuffle=False, collate_fn=collate):
        idx = img_i.numpy()
        if permute:
            idx = np.array([other[i] for i in idx])
        logits = model(make_feats(kind, idx), caps)
        tot += F.cross_entropy(logits.reshape(-1, VOCAB), caps[:, 1:].reshape(-1),
                               ignore_index=PAD, reduction="sum").item()
        n += (caps[:, 1:] != PAD).sum().item()
    return tot / n


DEP = {}
for _name, _model, _kind in [("merge", merge, "avg"), ("attention", spatial, "spa")]:
    _a, _s = val_loss(_model, _kind), val_loss(_model, _kind, permute=True)
    DEP[_name] = {"aligned": round(_a, 4), "shuffled": round(_s, 4), "delta": round(_s - _a, 4)}
print(pd.DataFrame(DEP).T.to_string())

# %%
def sentence_bleu4(refs, hyp, smooth=1.0):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu(refs, hyp, weights=(.25,) * 4,
                         smoothing_function=SmoothingFunction().method1)


best_cfg = BLEU.sort_values("bleu4", ascending=False).iloc[0]
best_hyps = CONFIGS[(best_cfg.model, best_cfg.decoder)]
per_img = np.array([sentence_bleu4(REFS[i], best_hyps[i]) for i in range(len(test_idx))])
order = np.argsort(per_img)
picks = [("good", int(i)) for i in order[::-1][:2]] + [("poor", int(i)) for i in order[:2]]
print("best configuration:", best_cfg.model, best_cfg.decoder)

# four examples in one column: two where the caption is close to all five references, two where
# it is not. `fig.add_axes` + a fixed rectangle keeps every panel the same size, so the captions
# line up; wrapping is done by hand because `wrap=True` only wraps at the figure edge.
fig = plt.figure(figsize=(3.45, 3.3))
for k, (tag, i) in enumerate(picks):
    r, c = divmod(k, 2)
    x0 = 0.02 + c * 0.50
    y_img = 0.73 - r * 0.44
    ax = fig.add_axes([x0, y_img, 0.46, 0.235])
    ax.imshow(Image.open(IMG_DIR / cap.loc[test_idx[i], "img"]).convert("RGB"))
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{tag}: BLEU-4 {per_img[i]:.2f}", fontsize=5.6,
                 color="#1a7f37" if tag == "good" else "#b2182b", pad=2)
    gen = textwrap.fill("gen: " + " ".join(best_hyps[i]), 40)
    ref = textwrap.fill("ref: " + " ".join(REFS[i][0]), 40)
    fig.text(x0 + 0.23, y_img - 0.015, gen, fontsize=4.6, ha="center", va="top")
    fig.text(x0 + 0.23, y_img - 0.015 - 0.0242 * (gen.count("\n") + 1) - 0.018, ref,
             fontsize=4.6, ha="center", va="top", style="italic", color="#444444")
fig.savefig(FIG / "C3_fig3_examples.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# %%
best_merge = BLEU[(BLEU.model == "merge")].sort_values("bleu4", ascending=False).iloc[0]
best_attn = BLEU[(BLEU.model == "attention")].sort_values("bleu4", ascending=False).iloc[0]
results = {
    "task": "C3",
    "env": ENV,
    "dataset": {
        "images": int(len(cap)), "captions": int(len(cap) * 5),
        "n_train": int(n_tr), "n_val": int(n_va), "n_test": int(len(test_idx)),
        "vocab_before": int(len(cnt)), "vocab": int(VOCAB), "min_freq": MIN_FREQ,
        "mean_len": round(float(lens["train"].mean()), 2),
        "median_len": int(np.median(lens["train"])),
        "p95_len": int(np.percentile(lens["train"], 95)),
        "max_len": int(MAXLEN),
        "top_words": [[w, int(c)] for w, c in cnt.most_common(10)],
        "oov_val": round(float(np.mean([w not in stoi for t in tokens["val"] for w in t])), 4),
        "oov_test": round(float(np.mean([w not in stoi for t in tokens["test"] for w in t])), 4),
    },
    "encoder": {"backbone": "resnet50 IMAGENET1K_V2 (frozen)", "dim_avg": 2048,
                "grid": [7, 7, 2048], "extract_seconds": round(feat_t, 1),
                "images_per_second": round(len(paths) / feat_t, 1)},
    "training": {"merge": hist_merge, "attention": hist_spatial},
    "image_dependence": DEP,
    "bleu": BLEU.to_dict(orient="records"),
    "best": {"merge": {"decoder": best_merge.decoder, "bleu4": float(best_merge.bleu4)},
             "attention": {"decoder": best_attn.decoder, "bleu4": float(best_attn.bleu4)}},
    "examples": [{"kind": t, "img": cap.loc[test_idx[i], "img"],
                  "gen": " ".join(best_hyps[i]), "ref": " ".join(REFS[i][0]),
                  "bleu4": round(float(per_img[i]), 3)} for t, i in picks],
}
(RES / "C3_results.json").write_text(json.dumps(results, indent=2))
BLEU.to_csv(RES / "C3_bleu.csv", index=False)
pd.DataFrame({"img": cap.loc[test_idx, "img"].values}).assign(
    **{f"{m}_{d}": [" ".join(h) for h in CONFIGS[(m, d)]]
       for (m, d) in CONFIGS}).to_csv(RES / "C3_captions_test.csv", index=False)
print("wrote", RES / "C3_results.json")
print(BLEU.to_string(index=False))

# %% [markdown]
# ## 11. Discussion
#
# *(Written after the numbers above were produced; every claim below is checked against
# `C3_results.json`.)*
#
# **Does the decoder use the image?** Scoring the validation captions a second time, with each
# image's features replaced by another image's, costs 0.98 nats per token for the merge decoder and
# 1.27 for attention. The decoders are reading the picture, not just the caption prior. The check
# exists because a broken feature/caption alignment looks healthy from the outside: the loss still
# falls and the captions are still fluent.
#
# **Fit.** Both decoders reach about 2.72 nats per token on validation while the training loss keeps
# falling to 2.41, so they start to overfit after roughly twenty epochs; the best-validation
# checkpoint is kept instead of the last one. The merge decoder ends with the marginally lower
# validation loss (2.714 vs 2.723) but the lower BLEU-4 (0.202 vs 0.227). Per-token likelihood and
# corpus BLEU are different objectives: a model can be better calibrated and still write captions
# that share fewer 4-grams with five particular references.
#
# **Greedy vs beam.** Beam search lifts BLEU-4 for both decoders (merge 0.163 -> 0.193 -> 0.202 at
# k = 1, 3, 5; attention 0.190 -> 0.216 -> 0.227), because it maximises the sequence
# log-probability instead of taking each locally best token. Its effect on the vocabulary is not
# monotone: the merge decoder uses 247 -> 254 distinct words as k grows, attention 311 -> 296. Beam
# search buys fluency; the distinct-word count says how much of it is template.
#
# **Why captions are generic.** The objective is per-token maximum likelihood over a 2,556-word
# vocabulary. The most likely continuation of "a dog" is "is running through the grass" — a phrase
# the corpus contains many times — so the model converges towards the mode of the conditional
# distribution and has little incentive to name *this* dog, *this* colour or *this* number of
# people. The most frequent generated caption ("a brown and white dog is running through the grass")
# still appears only 9 times in 1,091 test images, so the model is not collapsing; it is just plain.
#
# **Task-specific error types.** Counting ("two dogs" for a single fox terrier), object identity
# (cattle described as dogs), colour (a 7x7 attention grid and a global average-pooled vector both
# lose fine colour detail) and, at the tail of beam search, repetition ("a man in a white shirt and
# a man in a white shirt and ..."). Attention anchors nouns to the right region more often, which is
# where its higher BLEU comes from.

# %% [markdown]
# ## References
#
# [1] A. Karpathy and L. Fei-Fei, "Deep visual-semantic alignments for generating image
# descriptions," CVPR 2015.
# [2] K. Xu et al., "Show, attend and tell: Neural image caption generation with visual attention,"
# ICML 2015.
# [3] O. Vinyals et al., "Show and tell: A neural image caption generator," CVPR 2015.
# [4] M. Hodosh, P. Young, and J. Hockenmaier, "Framing image description as a ranking task,"
# JAIR 2013 (Flickr8k).
# [5] K. He et al., "Deep residual learning for image recognition," CVPR 2016.
