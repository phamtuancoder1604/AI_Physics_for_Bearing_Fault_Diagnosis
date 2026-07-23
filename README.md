Python
readme_content = """# AI-Guided Resonance Localization and Physics-Based Verification for Bearing Fault Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, physics-guided hybrid AI framework for industrial rolling element bearing fault diagnosis. The system combines the spatial noise-filtering capabilities of an unsupervised **Convolutional Autoencoder (CAE)** with deterministic **Physics-Based Kinematic Verification** to achieve highly transparent, robust, and real-time diagnostic performance.

---

##  Key Highlights

- **Domain-Separation Strategy (1D → 2D → 1D):** Converts raw 1D vibration signals into 2D magnitude spectrograms via Short-Time Fourier Transform (STFT), intentionally discarding the phase angle component to overcome 1D wave cancellation caused by micro-phase shifts.
- **Zero-Fault Unsupervised Paradigm:** Trained exclusively on normal healthy baseline data (`latent_dim = 128`), forcing the Autoencoder to act as an aggressive nonlinear low-pass filter that fails to reconstruct unfamiliar fault transients, resulting in clean 2D residual energy matrices.
- **Automated Resonance Localization:** Employs Spectral Kurtosis across the frequency axis of the 2D residual matrix to automatically pinpoint the optimal resonance band $[f_{\\text{low}}, f_{\\text{high}}]$, eliminating reliance on manual engineer tuning.
- **Deterministic Kinematic Verification:** Reverts back to the original phase-preserved 1D signal for Butterworth bandpass filtering and Hilbert Envelope Analysis. The extracted peak frequency ($f_{\\text{peak}}$) is mathematically cross-referenced against mechanical kinematic formulas (BPFO, BPFI) within a strict $\\pm 2\\%$ tolerance margin.
- **Extreme Noise Robustness:** Achieves **100% accuracy** on clean baselines and maintains **95% accuracy** under severe Additive White Gaussian Noise (**-4 dB SNR**).
- **Edge AI Ready:** Ultra-lightweight footprint (< 2 million parameters) with an end-to-end inference latency of **16.65 ms per sample** (~60 Hz), suitable for IIoT sensor node deployment.

---

##  System Architecture


README.md created successfully!

[Raw 1D Vibration Signal x(t)]
│
▼
[Step 1: STFT Transformation] ──► (Discards Phase Angle → 2D Magnitude Spectrogram S)
│
▼
[Step 2: Hourglass CAE]       ──► Trained on Healthy States → Reconstructs Baseline (Ŝ)
│
▼
[Step 3: Residual Matrix R]   ──► R = ReLU(S - Ŝ) → Spectral Kurtosis Analysis
│
▼
[Optimal Bandwidth Coordinates] ──► Extract [f_low, f_high]
│
▼
[Step 4: Revert to Raw 1D]    ──► Bandpass Filter [f_low, f_high] → Hilbert Envelope → FFT
│
▼
[Kinematic Matching]          ──► Compare f_peak with Physical Constants (BPFO / BPFI)
│
▼
[Diagnostic Decision]         ──► Match within ±2% margin? ──► [FAULT ALERT / NORMAL]


---

##  Tech Stack

| Domain | Technology / Library | Usage |
| :--- | :--- | :--- |
| **Language** | `Python 3.8+` | Core development |
| **Deep Learning** | `PyTorch` | Unsupervised Hourglass CAE model implementation & training |
| **Signal Processing** | `SciPy` (`scipy.signal`) | STFT, Butterworth Bandpass Filter, Hilbert Transform |
| **Mathematics & Stats** | `NumPy`, `Scikit-learn` | Matrix ops, FFT, Spectral Kurtosis, Evaluation metrics |
| **Data I/O & Vis** | `SciPy.io`, `Matplotlib`, `Seaborn` | Loading CWRU `.mat` files, Spectrogram & Envelope plotting |

---

##  Experimental Results

Evaluated on the **Case Western Reserve University (CWRU)** Bearing Benchmark Dataset (Drive End, 1797 RPM, 0 HP load, 48 kHz sampling rate):

| Metric | Clean Baseline | 10 dB SNR | 4 dB SNR | 0 dB SNR | **-4 dB SNR** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **100.0%** | **100.0%** | **98.57%** | **97.14%** | **95.00%** |
| **AUC-ROC** | **99.93%** | - | - | - | - |
| **Kinematic Error**| **0.269%** | - | - | - | - |
| **Inference Time** | **16.65 ms** | - | - | - | - |

---

##  Repository Structure
├── configs/ 
├── data/                      # Data folder (CWRU .mat files)
├── evaluation/                # Results
├── model.py                   # PyTorch model definitions (CAE architecture)
├── signal_preprocessing.py    # signal preprocessing
├── main.py                    # Full pipeline
├── evaluate.py                # End-to-end testing script with noise injection
├── diagnosis.py               # Main diagnostic pipeline execution
├── prepare_dataset.py         # Conver signal to image
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation


---

##  Quick Start

### 1. Installation

Clone the repository and install required Python packages:

```bash
git clone [https://github.com/your-username/bearing-fault-diagnosis-hybrid.git](https://github.com/your-username/bearing-fault-diagnosis-hybrid.git)
cd bearing-fault-diagnosis-hybrid
pip install -r requirements.txt
2. Training
Train the unsupervised Convolutional Autoencoder on healthy baseline spectrograms:

Bash
python train.py --config config.yaml
3. Evaluation & Inference
Run the complete 4-step AI-Physics pipeline on test samples with noise simulation:

Bash
python evaluate.py --snr -4
 Citation
If you use this work or architecture in your research, please cite our manuscript:

Đoạn mã
@inproceedings{pham2026aiguided,
  title={AI-Guided Resonance Localization and Physics-Based Verification for Bearing Fault Diagnosis},
  author={Pham, Van Hoang Anh Tu and Pham, Van Hoang Anh Tuan},
  organization={FPT University},
  year={2026}
}
 License
This project is released under the MIT License.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md created successfully!")

Your Markdown file is ready
[file-tag: code-generated-file-0-1784799072060078501]

