<div align="center">

# WM-DETR: Dual-Branch Wavelet-Mamba and Sparse Attention for Robust Underwater Object Detection

[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Last Commit](https://img.shields.io/badge/last%20commit-March%202026-yellowgreen)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-orange)]()
[![Forks](https://img.shields.io/github/forks/Xiaofeng-Han-Res/WM-DETR?style=social)]()

**Xiaofeng Han · Shunpeng Chen · Zhenghuang Fu · Zhe Feng · Lue Fan · Dong An · Zhangwei Wang · Li Guo · Weiliang Meng\* · Xiaopeng Zhang · Rongtao Xu\* · Shibiao Xu**

</div>

---

This repository presents **WM-DETR**, a robust underwater object detection framework that integrates **wavelet-based frequency decomposition**, **state-space modeling (Mamba)**, and **sparse adaptive interaction**.

Underwater detection is severely affected by **light attenuation, scattering noise, color distortion, low contrast, and small objects**. WM-DETR addresses these challenges by decomposing visual features into frequency sub-bands and modeling them using dual-branch state-space representations.

Our framework improves detection robustness by **suppressing background noise while preserving fine structural details**, achieving strong performance on challenging underwater benchmarks.

---

# Highlights

- **Dual-Branch Wavelet-Mamba (DBWM)**  
  Decomposes features into low-frequency and high-frequency components and applies Mamba-based modeling to capture global semantics and fine-grained details.

- **Sparse-Adaptive Interaction Module (SAIM)**  
  Combines adaptive sparse attention with spatially-aware feedforward operations to reduce redundancy while preserving small-object features.

- **Robust Underwater Perception**  
  Designed for challenging underwater environments with occlusion, blur, color distortion, and cluttered backgrounds.

---

# Method Overview

WM-DETR consists of two key modules:

### 1. DBWM (Dual-Branch Wavelet-Mamba)

- Applies **Haar wavelet decomposition** to split features into frequency sub-bands
- **Global-Mamba branch** models long-range semantic dependencies
- **Detail-Mamba branch** enhances fine structures and object boundaries
- Adaptive fusion balances **noise suppression and detail preservation**

### 2. SAIM (Sparse-Adaptive Interaction Module)

- Introduces **adaptive sparse attention**
- Integrates **spatially-aware feedforward operations**
- Improves feature interaction efficiency while preserving weak target cues

---

# Visualization

WM-DETR produces **more compact and target-aligned activations** than RT-DETR.

Example comparisons include:

- Attention heatmaps
- Internal feature visualization
- Haar frequency sub-band decomposition

These visualizations demonstrate that WM-DETR **suppresses noisy background activations while enhancing true object responses**.

---

# Datasets

Experiments are conducted on two challenging underwater detection datasets:

### DUO
A widely used underwater object detection benchmark.

### RUOD
A challenging dataset containing multiple underwater object categories and complex environments.

Dataset structure:
