# Foliar disease classification on Niayes Crops Disease V2

Code and artifacts for the manuscript submitted to *AgriEngineering* (agriengineering-4675423).

The repository covers three steps: a duplicate-detection pass over the dataset, training of six
architectures on the resulting split, and the evaluation that produced the tables and figures in the
paper. Trained weights are hosted on Zenodo rather than here, because GitHub rejects files above
100 MB and two of the checkpoints are around 198 MB.

TODO: add the paper title and the DOI once the revised version is accepted.

## Layout

```
notebooks/
  00_deduplication.ipynb   duplicate detection, writes the frozen split
  01_training.ipynb        one notebook, six runs (see MODEL_VARIANT)
  02_evaluation.ipynb      metrics, significance tests, figures
configs/                   one config.json per trained model
splits/
  keep_indices_train.npy   indices of the training images kept
  clean_test_mask.npy      test images with no near-duplicate in training
results/
  dedup_report.json        all duplicate-detection counts
  tables/                  CSV versions of the tables in the paper
```

## Data

The dataset is public: https://www.kaggle.com/datasets/ibrahimagabardiop/niayes-crops-disease-v2
(36 classes; 59,152 training, 12,666 validation and 12,705 test images). We did not modify or
redistribute it.

We did find that a large share of it is duplicated across the official splits. Using DINOv2
embeddings at a cosine similarity of 0.98, with the eight dihedral orientations of each training
image embedded so that rotated and mirrored copies are caught, 8,626 of the 12,705 test images
(67.9%) have a near-identical counterpart in the training set, and 8,607 of 12,666 validation images
do as well. The figure stays at 63.2% even at a cosine of 0.999, so it is not an artefact of the
threshold. The discrepancy with PlantVillage class counts (152 healthy potato images there against
2,432 here) suggests the dataset was augmented before being partitioned.

We therefore removed from training every image with a duplicate in validation or test (23,532) and
every internal duplicate (14,116), leaving 21,504 training images. The test and validation splits
were left untouched, so results stay comparable with other work; the paper reports metrics both on
the full test set and on the 4,079 test images with no detected duplicate in training.

`00_deduplication.ipynb` reproduces all of this. It also runs the dHash method as a diagnostic, which
is how we found that dHash with a Hamming distance of 3 produces a large number of false positives on
uniform-background leaf photographs — 41.9% of the pairs it flags are not supported by the embedding
detector, and 55.6% of those pair images from different classes. Exact canonical dHash matching does
not have this problem, but it misses copies that embeddings catch.

### Checking you have the same split

Every training run recorded a fingerprint of the split it used. All six should print the same value:

```python
import hashlib, numpy as np
idx = np.load("splits/keep_indices_train.npy")
print(hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:16])
# 918155abacc0323d
```

If you get something else, the deduplication was run against a different copy of the dataset.

## Environment

Everything ran on Kaggle notebooks with a single NVIDIA Tesla T4. Exact versions are recorded in each
`configs/*.json` under `entorno`, since that is what the runs actually used:

| | |
| --- | --- |
| GPU | NVIDIA Tesla T4, 15.6 GB |
| Peak GPU memory in training | 7.43 GB (varies by architecture; see each config) |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| timm | 1.0.30 |
| CUDA | 13.0 |
| Random seed | 42 |

Training one model takes between 69 and 111 minutes on a T4, depending on the architecture.

## Reproducing

Run the notebooks in order. They are written for Kaggle, so paths point at `/kaggle/input`; adjust the
`CFG` block at the top if you run them elsewhere. All three need a GPU, and `00` and `01` need network
access to download pretrained weights.

**00 — deduplication.** Attach the dataset and run it. Takes about 45 minutes, most of it spent
embedding 59,152 training images in eight orientations. It writes `keep_indices_train.npy` and
`clean_test_mask.npy`, which the other two notebooks read. Before using the output, look at
`fig_pares_limite.png`: it shows the lowest-similarity pairs the detector accepted, and if those are
not actually copies then `EMB_COSINE_MIN` is set too low.

**01 — training.** The same notebook trains all six models; change one line:

```python
MODEL_VARIANT = "convnext_small_cbam"
# convnext_small_plain | resnet50 | efficientnet_b3 | swin_tiny | mobilenetv3_large
```

Every run uses the frozen split, the same seed, the same augmentation and the same optimiser
settings. The only thing that differs between the two ConvNeXt variants is the CBAM block, which
is 80k parameters out of 49.5M. Each run writes `best_model.pth`, the test and validation logits as
`.npz`, a `config.json` and a `history.json`. The logits are what the significance tests in `02` need,
so keep them.

**02 — evaluation.** Attach the six training outputs plus the split. It works from the saved logits,
so it finishes in a few minutes; only the Grad-CAM section runs the model again. If a run is missing
it skips that variant and says so.

Training hyperparameters are in the paper and in full in each `config.json`. In short: AdamW with
1e-4 for the backbone and 1e-3 for the head, weight decay 0.05 (not applied to normalisation and bias
parameters), one epoch of linear warmup followed by cosine decay stepped per batch, batch size 64, up
to 30 epochs with early stopping on validation macro-F1, MixUp and CutMix disabled for the last five
epochs, and an EMA of the weights for evaluation.

## Results

Test set, 12,705 images. "Dedup" is the 4,079 test images with no detected duplicate in training.

| Model | Params | GFLOPs | Accuracy | Accuracy (dedup) | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-S + CBAM | 49.56M | 17.37 | 0.9866 | 0.9755 | 0.9776 |
| ConvNeXt-S | 49.48M | 17.37 | 0.9879 | 0.9774 | 0.9764 |
| Swin-T | 27.55M | 8.98 | 0.9876 | 0.9774 | 0.9742 |
| EfficientNet-B3 | 10.75M | 1.92 | 0.9832 | 0.9708 | 0.9700 |
| MobileNetV3-L | 4.25M | 0.43 | 0.9776 | 0.9645 | 0.9605 |
| ResNet-50 | 23.59M | 8.17 | 0.9708 | 0.9620 | 0.9553 |

Two things worth pointing out. CBAM does not help: McNemar on the paired predictions gives p = 0.081
for accuracy and a paired bootstrap gives p = 0.465 for macro-F1, and the sign of the difference
depends on which metric you look at. And the six architectures land within 1.7 points of each other
despite a 40x spread in compute, which suggests the choice of backbone matters less on this dataset
than the quality of the data.

## Weights

On Zenodo: https://doi.org/10.5281/zenodo.24222910

Six checkpoints, one per architecture, as PyTorch `state_dict` files. Load them with the model
definition in `01_training.ipynb` — the classifier head is a `LayerNorm`, a `Dropout` and a `Linear`
layer on top of the timm backbone, and the CBAM variant inserts the attention block before global
pooling.

## Known limitations

The models were only evaluated on Niayes Crops Disease V2. We did not collect field photographs from
urban gardens in Lima, which is the intended deployment setting, so nothing here says how the models
behave on images taken with a phone in a real garden.

The duplicate detection finds near-duplicates, not all duplicates. Calling the filtered subset
"leakage-free" would be too strong; it is a subset with detectable duplicates removed.

Three citrus classes have fewer than 35 test images each, and their confidence intervals are wide.
Citrus Healthy has eight test images and an F1 of 0.875 with a bootstrap interval of [0.615, 1.000].
Per-class numbers for small classes should be read with that in mind.

Each configuration was trained once, with a single seed. Differences of a few tenths of a point
between architectures are within what a different seed could produce.

## Citation

TODO: add the BibTeX entry once the paper is published.

## License

Code in this repository: MIT (see `LICENSE`). The dataset keeps its own license from Kaggle, and the
pretrained backbones keep theirs from `timm`.