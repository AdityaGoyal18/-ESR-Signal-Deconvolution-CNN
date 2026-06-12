# Deep Learning Assisted ESR Deconvolution for Resolving Overlapping Radical Spectra

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Research](https://img.shields.io/badge/Research-ESR-green)
![Conference](https://img.shields.io/badge/Presented-ICESR%202026-purple)

A deep learning framework for separating overlapping Electron Spin Resonance (ESR) signals into individual radical components using a 1D Convolutional Neural Network (CNN).

Presented at the International Conference on Electron Spin Resonance (ICESR 2026), Indian Institute of Science (IISc), Bengaluru.

---

## Research Poster

<p align="center">
  <img src="assets/poster.png" width="800">
</p><img width="768" height="1024" alt="ICESR Poster-Deep Learning Assisted ESR Deconvolution for Resolving Overlapping Radical Spectra" src="https://github.com/user-attachments/assets/969fc9c1-a90a-4fdc-9897-ac79c8fde037" />


---

## Overview

Electron Spin Resonance (ESR) spectroscopy is widely used for studying materials containing unpaired electrons (radicals). In many practical applications, signals originating from multiple radicals overlap, making spectral interpretation difficult.

Traditional spectral fitting techniques often struggle when:

- Multiple radical components are present
- Signal overlap is high
- Noise levels increase
- Manual parameter tuning is required

This project introduces a Deep Learning based ESR spectral deconvolution framework capable of automatically separating composite ESR spectra into their individual radical components.

---

## Problem Statement

Overlapping ESR spectra from multiple radicals create complex composite signals that are difficult to analyze using conventional methods.

### Objective

Given a composite ESR signal:

```
Composite ESR Signal
        ↓
Neural Network
        ↓
Radical 1 + Radical 2
```

The goal is to accurately reconstruct the individual radical spectra from the mixed ESR signal.

---

## Methodology

### Synthetic Dataset Generation

A synthetic ESR dataset was generated using Lorentzian derivative line-shape modeling.

Each sample consists of:

- Two individual radical spectra
- One composite ESR spectrum
- Controlled overlap conditions
- Various peak positions and linewidths

---

### Model Architecture

A 1D Convolutional Neural Network (CNN) is employed for ESR spectral deconvolution.

#### Input

```
Composite ESR Spectrum
Shape: (1 × 256)
```

#### Output

```
Radical 1 Spectrum
Radical 2 Spectrum

Shape: (2 × 256)
```

### CNN Pipeline

```
Input ESR Signal
        ↓
Conv1D Layer
        ↓
ReLU Activation
        ↓
Feature Extraction
        ↓
Flatten
        ↓
Fully Connected Layers
        ↓
Separate Radical Spectra
```

### Training Details

| Parameter | Value |
|------------|--------|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Input Size | 256 |
| Output Size | 512 |
| Network Type | 1D CNN |

---

## Repository Structure

```
ESR-Signal-Deconvolution-CNN
│
├── data/
├── dataset/
├── models/
├── training/
├── evaluation/
│
├── assets/
│   ├── poster.png
│   ├── architecture.png
│   ├── workflow.png
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   └── fig4.png
│
├── test.py
├── esr_model.pth
└── README.md
```

---

## Results

## ESR Signal Deconvolution Results

<img width="1920" height="1020" alt="results" src="https://github.com/user-attachments/assets/df71473c-2144-4d2a-9c17-515044433016" />

---

## Performance Evaluation

### Quantitative Results

| Metric | Radical 1 | Radical 2 |
|----------|----------|----------|
| RMSE | 0.0932 | 0.0955 |
| MAE | 0.0317 | 0.0320 |
| R² Score | 0.4810 | 0.6371 |
| Correlation | 0.6972 | 0.8269 |
| SNR (dB) | 2.85 | 4.40 |

### Baseline vs CNN

| Method | RMSE |
|----------|----------|
| Baseline | 0.1777 |
| CNN | 0.1373 |

The proposed CNN-based ESR deconvolution framework achieves approximately **22.8% lower RMSE** compared to the baseline approach, demonstrating improved signal separation and reconstruction capability.

### Key Findings

- Accurate ESR spectral deconvolution
- Reliable reconstruction under strong spectral overlap
- Improved RMSE compared to conventional baseline methods
- Effective learning of ESR line shapes and peak positions
- Automated separation of radical spectra without manual fitting
- Strong agreement between predicted and ground-truth signals
---

## Applications

This framework can be applied to:

- ESR Spectroscopy
- Material Science
- Radical Detection
- Chemical Analysis
- Biomedical Signal Processing
- Spectral Deconvolution Problems
- Scientific Machine Learning

---

## Future Work

Future improvements include:

- Application to real experimental ESR datasets
- Multi-radical spectral separation
- Transformer-based architectures
- Physics-informed neural networks
- Noise-robust ESR analysis
- Real-time ESR spectral processing

---

## Conference Presentation

This research work was presented at:

**International Conference on Electron Spin Resonance (ICESR 2026)**

 Indian Institute of Science (IISc), Bengaluru, India

March 21–24, 2026

Our abstract titled:

**"Deep Learning Assisted ESR Deconvolution for Resolving Overlapping Radical Spectra"**

was accepted and published in the official ICESR 2026 Abstract Booklet.

[Official ICESR Abstract Booklet](https://share.google/g74VZMShYfz6wrjyS)

Page: 70
---

## Authors

### Palak Vastrakar
B.Tech CSE  
Dr. Shayma Prasad Mukherjee International Institute of Information Technology Naya Raipur

### Aditya Goyal

### Srishti Tripathi

### Dr. Punya Prasanna Paltani

---

## Citation

If you use this work, please cite:

```bibtex
@conference{vastrakar2026esr,
  title={Deep Learning Assisted ESR Deconvolution for Resolving Overlapping Radical Spectra},
  author={Vastrakar, Palak and Goyal, Aditya and Tripathi, Srishti and Paltani, Punya Prasanna},
  booktitle={International Conference on Electron Spin Resonance (ICESR)},
  year={2026}
}
```

---

## Acknowledgements

The authors thank:

- IIIT Naya Raipur
- Research mentors and faculty members
- ICESR 2026 organizers
- Indian Institute of Science (IISc), Bengaluru

for their support and guidance throughout this research work.

---

⭐ If you found this project useful, please consider giving the repository a star.
