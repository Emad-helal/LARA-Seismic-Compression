HARA-Seismic-Compression
Seismic Waveform Compression Using Attention-Enhanced Autoencoder
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
---
Abstract
Efficient storage and transmission of seismic waveforms are fundamental requirements in modern seismological monitoring. Existing compression methods, however, struggle to preserve critical signal characteristics at moderate-to-high compression ratios. Transform-based approaches lack the adaptability required for seismic signals, whose frequency content and energy distribution vary significantly over time. Standard deep learning architectures tend to degrade high-frequency components and phase fidelity during reconstruction. These limitations motivate the development of more robust compression frameworks.
This paper proposes the Attention-Enhanced Heavy Advanced Residual Autoencoder (HARA), a novel end-to-end deep learning model for high-fidelity seismic signal compression. HARA combines hierarchical residual learning with channel-wise attention mechanisms that dynamically prioritise seismologically significant features while limiting information loss across the compression pipeline. The model is evaluated on the Stanford Earthquake Dataset (STEAD) across compression ratios from 2:1 to 100:1. HARA consistently outperforms wavelet-based methods and conventional autoencoder baselines, particularly at ratios of 10:1 and above. The 20:1 ratio represents the optimal operating point, achieving an SNR of 35.41 dB and an SSIM of 0.992. At the extreme 100:1 ratio, HARA maintains a 6.35 dB SNR advantage over the strongest baseline, with a correlation coefficient approximately 30% higher under the same conditions.

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
  title   = {Seismic Waveform Compression Using Attention-Enhanced Autoencoder},
  author  = {Helal, Emad B. and others},
  journal = {Computers \& Geosciences},
  year    = {2025},
  doi     = {}
}
```
---
License
MIT License — see LICENSE for details.
