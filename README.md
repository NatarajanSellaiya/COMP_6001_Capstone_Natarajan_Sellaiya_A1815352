# Multimodal Precision Agriculture via Spatio Temporal Cross Modal Attention

**Course:** COMP 6001 Capstone
**Student:** Natarajan Sellaiya (A1815352)
**Model:** Spatio-Temporal Cross-Modal Gated Network (ST-CGN)

## Overview

ST-CGN classifies a crop canopy into five states (Healthy, Nitrogen Deficiency, Drought Stress, Weed Infestation, Waterway Encroachment) from two aligned inputs:

- a multispectral tile, shape `(4, H, W)`, channels red, green, blue, and near-infrared
- a meteorological token sequence, shape `(T,)`, with token `0` reserved for padding

The network is written from `nn.Conv2d`, `nn.Linear`, and `nn.LSTM`. It is compared with four baselines trained on the same stratified folds and the same optimizer: a unimodal CNN, a unimodal BiLSTM, early fusion, and late fusion. The primary endpoint is final-epoch macro F1.

This file is written by the notebook into the output directory at the start of a run. The numbers below follow the configuration that produced the file. `DEBUG_MODE` for this run is `False`.

## System requirements

- Google Colab Pro+ with an NVIDIA A100 is the graded runtime. A CPU run is only for `DEBUG_MODE = True`.
- Python 3.10 or newer.
- Packages pinned in `requirements.txt` in this same directory: `torch`, `torchvision`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `numpy`, and `thop`.
- `torchvision` and `thop` are not used as pretrained model factories. `thop` is an optional cross-check. The authoritative FLOP count is the notebook's own forward-hook profiler.
- Disk space for `outputs/`: figures, CSV tables, `oof_predictions.npz`, and `summary.json`.
- Do not upgrade an existing CUDA build of PyTorch. The setup cell installs a package only when its import fails, and it does not reinstall torch when torch already imports.

## Reproduce the graded run in Google Colab Pro+

1. Upload `ST_CGN_Precision_Agriculture.ipynb` to Google Drive or open it directly in Colab.
2. Menu path: **Runtime → Change runtime type → GPU → A100**.
3. In the configuration cell, leave `DEBUG_MODE = False`. That profile uses 2,000 images, 128×128 tiles, sequence length 30, and 15 epochs. Set `DEBUG_MODE = True` only for a short smoke test.
4. Optional real data, already mounted, in `/content/drive/MyDrive/COMP6001_STCGN/`:
   - `images.npy`, float array `(N, 4, H, W)` or `(N, H, W, 4)`, values in `[0, 1]` or `[0, 255]`
   - `tokens.npy`, int array `(N, T)`, ids in `[0, 11]`, with `0` meaning pad
   - `labels.npy`, int array `(N,)`, values in `{0, 1, 2, 3, 4}`
   - `climate.csv`, optional, columns `precip`, `temp`, `precip_deficit`, `thermal_spike`, `humidity`, `soil_moisture`
5. Do not call `drive.mount` inside **Run all**. An OAuth prompt blocks the run. Mount Drive yourself first if the files live there.
6. Menu path: **Runtime → Run all**. Cells are ordered. Skipping the training cell leaves the statistical tests without fold scores.
7. Collect artifacts from `outputs/`, or from `/content/drive/MyDrive/COMP6001_STCGN_outputs/` when Drive is already mounted.

If the three `.npy` files are absent, the notebook builds a controlled synthetic benchmark and prints that fact. The label in that benchmark is the spectral pattern inside the quadrant named by the order of the weather tokens. It is a mechanism test. It is not a field survey.

## Computational complexity

Let `n` be the image side, `t` the number of daily tokens, `d` the attention width, and `R` the pooled grid. A cross-modal score between every spatial site and every day is `O(H · W · T)` before any constant factors from the channel width. On a raw 128×128 tile and a 30-day log that product is `128² × 30` scores per sample, which is the memory term that exhausts a GPU.

The residual backbone uses stride 4, so the native map of a 128×128 tile is 32×32. Attention on that map, still before pooling, forms

`32² × 30 = 30,720` scores per sample.

Adaptive average pooling to a 4×4 grid is applied before `Q K^T` is formed. The mask then costs

`4² × 30 = 480` scores per sample.

That is a 64-fold reduction of the spatial-temporal product (`30,720` to `480`), and each image quadrant remains a 2×2 block on the grid. The Big-O term falls from `O(H' W' T d)` with `H' = W' = 32` to `O(R² T d)` with `R = 4`. Memory for the mask falls the same way: `O(B · 32 · 32 · T)` becomes `O(B · 4 · 4 · T)`.

This run's live settings are image side `128`, sequence length `30`, attention grid `4`. The native map is `32×32` (`30,720` scores) and the pooled mask is `4×4` (`480` scores). Parameter counts and measured FLOPs are written to `complexity.csv` and `07_complexity.png` by Section 3. The receptive field of the residual stack is 23 input pixels, which is local on a 128-pixel tile and is why a deeper 3×3 tower is not used.

Early fusion is `O(n² + t)` because it keeps only channel moments and a mean embedding. Late fusion pays for the CNN and the BiLSTM and does not build the `R² t` mask. ST-CGN pays for both encoders and for that small mask. The mask is the only term that can select a site.

## Edge cases and limitations

- **Softmax stability.** Scores are divided by `sqrt(d)` and clamped to `[-50, 50]`. Padded days are filled with `-50` before the spatial softmax. An all-pad row makes a temporal softmax undefined; those outputs are replaced with zeros and the visual readout is set to 0, so a missing log cannot inject NaN.
- **CUDA out of memory.** The dangerous activation is the attention map `B × R² × T` together with the convolutional maps, not the parameter count. The training loop clears the cache and raises an error that names `batch_size` and `image_size` as the knobs to lower. `DEBUG_MODE = True` is the safe fallback.
- **Missing weather logs.** Token `0` is padding. A row of pads is kept, not dropped. The temporal gate and the masked mean ignore those steps. Internal pads are left-aligned, which keeps a usable sequence and shifts absolute time; a field logger should emit an explicit missing-day token instead of a gap.
- **Class imbalance.** Healthy is the majority class by construction in the synthetic prior. The loss uses inverse-frequency weights fit on the training fold only and rescaled to mean 1. The reported endpoint is macro F1, so the minority classes keep an equal vote. Accuracy is reported and is not the decision metric.
- **BatchNorm.** The training loader drops a trailing batch only when the fold contains at least two full batches, so a batch of one does not update BatchNorm. Evaluation still scores every validation sample.
- **What the synthetic ceiling does not mean.** A high ST-CGN score on the fallback generator shows that the mask can recover a quadrant-scale interaction that pooling cannot represent. It does not estimate accuracy on a particular farm, season, or sensor.

## Artifacts

| File | Contents |
| --- | --- |
| `01_class_distribution.png` … `06_interaction_map.png` | Exploratory figures |
| `07_complexity.png`, `complexity.csv` | Parameters, FLOPs, Big-O |
| `08_convergence.png` … `16_paired_folds.png` | Training curves, metrics, ROC/PR, attention, gates, paired folds |
| `fold_metrics.csv`, `summary_metrics.csv`, `per_class_f1.csv` | Fold and aggregate scores |
| `linear_probes.csv` | Orderless ceilings and the privileged oracle |
| `statistical_tests.csv` | Paired t-test and Wilcoxon results |
| `oof_predictions.npz`, `summary.json`, `environment.txt` | Predictions and the run record |
| `requirements.txt`, `README.md` | This directory's pins and this guide |
