# Data — C3: Image Captioning (Flickr8k)

## What the dataset is

**Flickr8k**: 8,091 photographs from Flickr, each described by five independent human captions
(40,455 captions). Released by Hodosh, Young and Hockenmaier (JAIR 2013).

* Kaggle release used in the assignment brief:
  <https://www.kaggle.com/datasets/adityajn105/flickr8k>
* Hugging Face mirrors used here (no Kaggle account needed):
  * <https://huggingface.co/datasets/jxie/flickr8k> — the standard 6,000 / 1,000 / 1,000 split,
    8,000 images (the 91 images that Flickr8k does not assign to any split are absent).
  * <https://huggingface.co/datasets/tsystems/flickr8k> — a single 8,091-image split, which is
    what this notebook uses so that the 6,000 / 1,000 / **1,091** split can be made exactly as
    the brief specifies.

Both store the JPEG bytes in a Parquet `image` struct. They differ in layout, and the notebook
handles both: `tsystems/flickr8k` has one row per image with a `captions` list, while `jxie/flickr8k`
has one row per image with five `caption_0` … `caption_4` columns. No images need to be unpacked
from a zip by hand.

## How to obtain it

```bash
pip install huggingface_hub pyarrow
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("tsystems/flickr8k", "jxie/flickr8k"):
    try:
        p = snapshot_download(repo, repo_type="dataset",
                              allow_patterns=["data/*.parquet"],
                              local_dir=f"data/{repo.split('/')[0]}_flickr8k")
        print("ok", repo, p)
        break
    except Exception as exc:          # fall back to the other mirror
        print("failed", repo, exc)
PY
```

The notebook searches for `*.parquet` files in this order and uses the first group that holds the
full 8,091-image corpus (otherwise the largest group, with a warning):

1. `$AI101_DATA` (a directory of your choice)
2. `C3_Image_Captioning/data/`
3. `<repo root>/data/`
4. `/mnt/DC5E14885E145D98/ai101_ws/data/` — where this run was executed

If nothing is found the cell raises
`FileNotFoundError: No Flickr8k parquet files found - see C3_Image_Captioning/data/README.md`.

## What the notebook caches (all git-ignored)

| path | what | size |
|---|---|---|
| `data/flickr8k_images/*.jpg` | the 8,091 JPEGs, written out once (~11 min the first time) | ~1.1 GB |
| `data/C3_feat_avg.npy` | frozen ResNet-50 global features, `(8091, 2048)` float16 | 33 MB |
| `data/C3_feat_spatial.npy` | frozen ResNet-50 7×7 feature maps, `(8091, 49, 2048)` float16 | 1.6 GB |

Delete `C3_Image_Captioning/data/` to force a clean re-extraction. The spatial cache is opened as
a NumPy memory map during training because this machine has 8 GB of RAM.

## Verification

* 8,091 unique image filenames, 5 captions each → 40,455 captions.
* Split: 6,000 / 1,000 / 1,091 images; an assertion in the notebook fails if any image appears in
  two splits.
* Duplicate filenames are dropped (`drop_duplicates("img")`); the notebook prints the resulting
  image count, so a different mirror is visible in the output rather than silently changing the
  experiment.
