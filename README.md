
# Project Title — Code & Evaluation Logs

This repository contains the exact scripts executed for our project results, along with the raw training/evaluation logs (`.jsonl`) and generated figures (`.png`) referenced in the presentation slides.

## Repository Structure

* `Scripts/` — Final data extraction, model architecture, training, and plotting scripts used to generate the reported numbers.
  * `extract_v2.py` — Data preprocessing and feature extraction pipeline.
  * `model2.py` — Final model architecture definition.
  * `train3.py` — Main training and evaluation loop producing the slide metrics.
  * `fleet_stats.py` — Summary statistics and fleet-level evaluation.
  * `fig2.py`, `fig3.py` — Scripts used to generate the evaluation plots.
* `Processed/` & `Scripts/Processed/` — Raw evaluation and training logs (`.jsonl`) corresponding to the runs reported on the slides. *(Large intermediate `.npz` arrays and raw datasets are excluded via `.gitignore`.)*
* `Graphs/` — Output figures (`.png`) generated directly from the logged runs (`ocv/`, `sun/`, `sun2/`, `tues/`).

## Mapping Logs & Figures to Slides

| Slide / Figure | Script Executed | Corresponding Log / Output |
| :--- | :--- | :--- |
| **Slide [X] — Main Results** | `Scripts/train3.py` | `Processed/fleet/.../*.jsonl` |
| **Slide [Y] — Figure 2** | `Scripts/fig2.py` | `Graphs/.../*.png` |
| **Slide [Z] — Figure 3** | `Scripts/fig3.py` | `Graphs/.../*.png` |

## Reproducing Results

1. Ensure Python 3.11+ and required packages (`numpy`, `matplotlib`, `torch`/`scikit-learn`) are installed.
2. Run feature extraction:
   ```bash
   python Scripts/extract_v2.py