<div align="center">

# WM-DETR: Dual-Branch Wavelet-Mamba and Sparse Attention for Robust Underwater Object Detection

[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Last Commit](https://img.shields.io/badge/last%20commit-March%202026-yellowgreen)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-orange)]()
[![Forks](https://img.shields.io/github/forks/Xiaofeng-Han-Res/WM-DETR?style=social)]()


</div>

# Introduction

This repository presents **WM-DETR**, a robust underwater object detection framework that integrates **wavelet-based frequency decomposition**, state-space modeling (Mamba), and sparse adaptive interaction.Underwater detection is severely affected by light attenuation, scattering noise, color distortion, low contrast, and small objects. WM-DETR addresses these challenges by decomposing visual features into frequency sub-bands and modeling them using dual-branch state-space representations.Our framework improves detection robustness by suppressing background noise while preserving fine structural details, achieving strong performance on challenging underwater benchmarks.

![Alt Text](illustrates.png)
---

# Quick Start

## 1. Dataset Preparation

### Download Dataset

**（1）DUO** (<a href="https://github.com/chongweiliu/DUO">GitHub</a>)
- Additional benchmark: **RUOD**

### Dataset Structure

Please download and organize the datasets with the following structure:

```text
datasets/
├── DUO/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
└── RUOD/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

