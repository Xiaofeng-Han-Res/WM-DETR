# WM-DETR: Dual-Branch Wavelet-Mamba and Sparse Attention for Robust Underwater Object Detection

This repository contains the code, experiments, and paper resources for **WM-DETR**, an end-to-end framework for robust underwater object detection.  
WM-DETR integrates **frequency-domain wavelet decomposition**, **state-space modeling**, and **adaptive sparse interaction** to improve robustness under underwater degradations such as scattering noise, color distortion, low contrast, and small-object ambiguity.

---

## Highlights

- **DBWM (Dual-Branch Wavelet-Mamba)**  
  Decomposes features into low- and high-frequency sub-bands.  
  A global-Mamba branch models long-range semantic dependencies, while a detail-Mamba branch enhances fine-grained structures and object boundaries.

- **SAIM (Sparse-Adaptive Interaction Module)**  
  Combines adaptive sparse attention with spatially aware feedforward operations to suppress redundancy and preserve informative small-object cues.

- **Robust Underwater Detection**  
  Designed for challenging underwater conditions, including occlusion, blurring, color distortion, low-light environments, and cluttered backgrounds.

---

## Repository Structure

```text
WM-DETR-paper/
├── README.md
├── .gitignore
├── main.tex
├── refs.bib
├── sections/
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   └── conclusion.tex
├── figures/
│   ├── challenge.pdf
│   ├── framework.pdf
│   ├── feature.pdf
│   └── results/
├── tables/
├── supplementary/
├── response/
└── src/
