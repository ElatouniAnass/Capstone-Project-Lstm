# ✈️ Predictive Maintenance for Turbofan Engines with LSTM

> Predicting whether an aircraft engine will fail within the next 30 cycles using deep learning on NASA's C-MAPSS sensor data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**Capstone project** | [Anass Elatouni] | [SCU / SWE] | [2025] 

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Model Architecture](#model-architecture)
6. [Results in Detail](#results-in-detail)
7. [Getting Started](#getting-started)
8. [Project Structure](#project-structure)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [Acknowledgments](#acknowledgments)
11. [Contact](#contact)

---

## Overview

Unplanned engine failures are costly and dangerous. Predictive maintenance aims to flag an engine **before** it fails so that maintenance can be scheduled proactively rather than reactively.

This project frames the problem as a **binary classification task**:

> Given the last 50 operating cycles of sensor readings from an engine, will it fail within the next **30 cycles**?

A stacked **LSTM** network is trained on multivariate time-series data from the NASA C-MAPSS FD001 dataset. The 30-cycle window is configurable (`w1`), so the model can be tuned to be more or less proactive depending on the maintenance strategy.

## Key Results

| Metric (Test Set) | Score |
|---|---|
| Accuracy | **96.77%** |
| Precision | **92.31%** |
| Recall | **96.00%** |
| F1-Score | **94.12%** |

The model correctly identified 24 of 25 engines that were close to failure, missing only one. High recall matters most here, since a missed failure is far more costly than a false alarm.

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation), subset **FD001**.

| Property | Value |
|---|---|
| Training rows | 20,631 |
| Test rows | 13,096 |
| Engines | 100 train / 100 test |
| Operating conditions | 1 (sea level) |
| Fault mode | 1 (HPC degradation) |
| Features | 3 operational settings + 21 sensor readings |

Each engine runs from a healthy state until failure in the training set. In the test set, trajectories stop before failure, and a separate file (`RUL_FD001.txt`) provides the true remaining useful life.

**Files used:** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`

Download: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Methodology

```
Raw sensor data
      │
      ▼
Clean & label  ──►  Compute RUL  ──►  failure_within_w1 = (RUL ≤ 30)
      │
      ▼
MinMax normalization (fit on train, applied to test)
      │
      ▼
Sliding windows (50 cycles × 25 features)
      │
      ▼
Stacked LSTM  ──►  Sigmoid output  ──►  Fail / No Fail
```

### 1. Data cleaning
- Dropped empty columns caused by trailing whitespace in the raw `.txt` files.
- Assigned column names: `id`, `cycle`, 3 operational settings, and sensors `s1` to `s21`.

### 2. Label generation
- **Training:** RUL = (max cycle of the engine) − (current cycle).
- **Test:** RUL = (last observed cycle + ground-truth additional RUL) − (current cycle).
- Target: `failure_within_w1 = 1` if `RUL <= 30`, else `0`.

### 3. Normalization
- Min-Max scaling to [0, 1] on settings, sensors, and a normalized cycle feature (`cycle_norm`).
- The scaler is **fit on training data only** and applied to the test set to avoid data leakage.

### 4. Sequence generation
- Sliding windows of **50 timesteps** are built per engine, so sequences never cross engine boundaries.
- Training tensor shape: `(15631, 50, 25)`.

### 5. Exploratory analysis
Interactive Plotly charts show clear degradation trends. For example, some sensors (such as `s2`) drift upward as an engine ages, while others (such as `s6`) drift downward, which confirms there is a learnable signal.

## Model Architecture

| Layer | Output Shape | Params |
|---|---|---|
| LSTM (80 units) | (None, 50, 80) | 33,920 |
| Dropout (0.2) | (None, 50, 80) | 0 |
| LSTM (40 units) | (None, 40) | 19,360 |
| Dropout (0.2) | (None, 40) | 0 |
| Dense (1, sigmoid) | (None, 1) | 41 |

**Total parameters:** 53,321 (~208 KB)

**Training configuration**

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Batch size | 200 |
| Max epochs | 200 |
| Validation split | 5% |
| Early stopping | Monitor `val_loss`, patience 10 |
| Checkpointing | Best model by `val_loss` |

Early stopping ended training at epoch 21 (about 23 seconds on a Colab GPU).

## Results in Detail

### Training set

| Metric | Score |
|---|---|
| Accuracy | 96.52% |
| Precision | 85.26% |
| Recall | 99.68% |
| F1-Score | 91.91% |

### Test set (last 50-cycle window of each engine)

Engines with fewer than 50 cycles of history were excluded, leaving 93 evaluation engines.

|  | Predicted No Fail | Predicted Fail |
|---|---|---|
| **Actual No Fail** | 66 | 2 |
| **Actual Fail** | 1 | 24 |

| Metric | Score |
|---|---|
| Accuracy | 96.77% |
| Precision | 92.31% |
| Recall | 96.00% |
| F1-Score | 94.12% |

The model generalizes well, with test performance comparable to or better than training performance. The recall-oriented behavior (very few missed failures) fits the safety-critical nature of the task.

## Getting Started

### Prerequisites

- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment

### Installation

```bash
git clone https://github.com/[ElatouniAnass]/[Capstone-Project-Lstm].git
cd [Capstone-Project-Lstm]
pip install tensorflow pandas numpy matplotlib seaborn plotly scikit-learn
```

### Running the project

1. Download the C-MAPSS dataset and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in a `CMAPSSData/` folder.
2. If using Colab, upload the folder to Google Drive. The notebook expects `/content/drive/MyDrive/CMAPSSData/`.
3. Open `[PdM_with_LSTM (1)].ipynb` and run all cells.
4. If running locally, remove the `google.colab` import and the `drive.mount(...)` call, and set `drive_path` to your local data folder.

### Adjusting the prediction window

Change `w1` in the preprocessing section to control how early the model flags failures:

```python
w1 = 30   # try 15, 30, 45 ...
```

A larger window gives earlier warnings but makes the task harder and may lower precision.

## Project Structure

```
├── CMAPSSData/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── [notebook-name].ipynb     # Full pipeline: preprocessing, EDA, training, evaluation
├── binary_model.keras        # Best saved model
├── binary_submit_train.csv   # Training predictions
├── binary_submit_test.csv    # Test predictions
└── README.md
```

## Limitations and Future Work

**Current limitations**
- Evaluated on a single subset (FD001) with one operating condition and one fault mode.
- The test set is small (93 engines), so metrics have wide confidence intervals.
- Only the final window of each test engine is evaluated, not the full degradation trajectory.
- Constant or near-constant sensors are kept as inputs rather than removed.

**Ideas for improvement**
- Test on FD002 to FD004 (multiple operating conditions and fault modes).
- Add feature selection to remove uninformative sensors.
- Compare against baselines (Random Forest, GRU, 1D-CNN, Transformer).
- Extend from classification to direct **RUL regression**.
- Tune hyperparameters (sequence length, units, dropout, learning rate).
- Use cross-validation grouped by engine ID for more robust estimates.
- Deploy as an API or dashboard for real-time monitoring.

## Acknowledgments

- Dataset: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.
- Built with TensorFlow/Keras, scikit-learn, Plotly, and Seaborn.
- [Supervisor / advisor name, if applicable]

## Contact

**[Anass Elatouni]**
- GitHub: [@Elatounianass](https://github.com/ElatouniAnass)
- Email: elatounianass@gmail.com

---

*If you found this project useful, consider giving it a ⭐*
