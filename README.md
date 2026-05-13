HARA-Seismic-Compression
Attention-Enhanced Heavy Advanced Residual Autoencoder for Seismic Waveform Compression
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
---
Overview
This repository contains the source code for the paper submitted to Computers & Geosciences. It provides a deep learning framework for compressing 1D seismic waveforms at compression ratios (CR) from 2× to 100×, evaluated against wavelet-based baselines using SNR, PSNR, SSIM, MSE, MAE, and Pearson Correlation.
---
Repository Structure
```
HARA-Seismic-Compression/
├── notebooks/
│   ├── Final_Enhanced_Autoencoder_with_Wavelet_comparisons.ipynb
│   └── Ablation_study.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```
---
Dataset
Experiments use the STEAD dataset (Mousavi et al., 2019), filtered to local earthquakes within 60 km and magnitude > 3. Download at: https://github.com/smousavi05/STEAD
Update the data paths in each notebook before running:
```python
csv_file  = "path/to/merged.csv"
file_name = "path/to/merged.hdf5"
```
---
Installation
```bash
git clone https://github.com/Emad-helal/HARA-Seismic-Compression.git
cd HARA-Seismic-Compression
pip install -r requirements.txt
```
---
Usage
Notebook	Description
`Final_Enhanced_Autoencoder_with_Wavelet_comparisons.ipynb`	Trains and evaluates all models across CRs 2–100; compares against wavelet baselines (db4, sym8, coif3)
`Ablation_study.ipynb`	Evaluates five architectural variants: Base, No-Attention, Plain-Conv, Light, and Heavy capacity
---
Models
Model	Type
`HeavyAdvancedResidualAutoencoder`	Proposed — residual blocks + channel attention
`AdvancedResidualAutoencoder`	Base ablation variant
`GeneralizedAutoencoder`	Neural baseline
`AE_PureConcat`	Neural baseline
Wavelet (db4 / sym8 / coif3)	Classical baseline
---
Citation
```bibtex
@article{helal2025hara,
  title   = {Attention-Enhanced Heavy Advanced Residual Autoencoder for Seismic Waveform Compression},
  author  = {Helal, Emad B. and others},
  journal = {Computers \& Geosciences},
  year    = {2025},
  doi     = {}
}
```
---
License
MIT License — see LICENSE for details.
