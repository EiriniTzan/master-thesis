# Machine Learning-Based Concept Drift Detection

This repository contains the code and experimental notebooks developed for an MSc thesis on concept drift detection in data streams.

The thesis examines two drift detection approaches:

1. A DNN-AE latent-space drift detection approach, based on:

> **Concept Drift Detection Based on Deep Neural Networks and Autoencoders**  
> Lisha Hu, Yaru Lu and Yuehua Feng

2. A STUDD-based drift detection approach, based on:

> **STUDD: A Student-Teacher Method for Unsupervised Concept Drift Detection**  
> Vitor Cerqueira, Heitor Murilo Gomes, Albert Bifet and Luis Torgo

The DNN-AE approach is evaluated in a controlled synthetic SEA setting, while the STUDD approach is investigated through detector-comparison experiments across multiple real datasets.

Synthetic data streams are generated with [CapyMOA](https://capymoa.org/), and the
STUDD baseline is also evaluated using CapyMOA's built-in implementation.

---

## Method overview

### DNN-AE latent-space drift detection

The first part of the thesis investigates a DNN-AE latent-space drift detection approach.

A base DNN classifier is trained on reference data. The latent representations produced by this classifier are then used to train an autoencoder. During stream processing, incoming samples are passed through the classifier, and their latent representations are reconstructed by the autoencoder.

Drift is signalled when the reconstruction error exceeds a threshold estimated from the reference data.

This approach is evaluated in:

- `notebooks/test_sea_dnn_ae.ipynb`

### STUDD-based drift detection

The second part of the thesis investigates STUDD, a student-teacher framework for unsupervised drift detection.

A teacher model is trained on an initial labeled batch, while a student model learns online to mimic the teacher's predictions. Changes in the teacher-student relationship are used as a signal for drift detection.

In addition to the paper-style STUDD setup, this repository evaluates different internal drift detectors inside the STUDD framework, including:

- ADWIN
- Page-Hinkley
- HDDMAverage
- HDDMWeighted
- CUSUM

The STUDD detector-comparison experiments are evaluated in:

- `notebooks/test_abrupt_insects.ipynb`
- `notebooks/test_detectors_bike.ipynb`
- `notebooks/test_detectors_covtype.ipynb`
- `notebooks/test_detectors_electricity.ipynb`
- `notebooks/test_detectors_posture.ipynb`

---

## Repository layout

```
tscls/
  models/          # DNNClassifier, StreamDNN, Autoencoder, builder helpers
  training/        # offline and online training loops
  detection/       # AEDriftDetector (capymoa BaseDriftDetector), threshold utils
  pipeline/        # PipelineConfig, AEDriftDetector façade, StreamMonitor
  core/            # shared result types and sample helpers

notebooks/
  detection_metrics_summary.ipynb    # computed classic detection metrics for STUDD detector comparison
  helpers.py                         # shared plotting and stream utilities
  studd_utils.py                     # helper functions for the STUDD experiments
  test_sea_dnn_ae.ipynb              # DNN+AE detector evaluated on the SEA benchmark
  test_abrupt_insects.ipynb          # STUDD detector comparison on the AbruptInsects dataset
  test_detectors_bike.ipynb          # STUDD detector comparison on the Bike dataset
  test_detectors_covtype.ipynb       # STUDD detector comparison on the Covtype dataset
  test_detectors_electricity.ipynb   # STUDD detector comparison on the Electricity dataset
  test_detectors_posture.ipynb       # STUDD detector comparison on the Posture dataset
  archive/                           # older exploratory and reference notebooks
  figures/                           # exported PDF figures (auto-created on notebook run)
```

---

## Data streams

### SEA synthetic stream

The DNN-AE experiment uses a synthetic SEA stream generated with CapyMOA.

The SEA setup uses capymoa-based data generation:

- **Reference set**: collected from a `CapySEA` generator (concept f4, θ = 9.5)
- **Stream**: a `DriftStream` that switches from f4 to f3 (θ = 7.0) at sample 5 000
- The stream is pre-collected into NumPy arrays and wrapped in `NumpyStream` for
  fast online evaluation (avoids per-sample JVM overhead)

The SEA experiment provides a controlled setting with a known drift point, allowing the detector behaviour to be inspected directly.

### Real datasets

The STUDD detector-comparison notebooks use real datasets from the STUDD paper.

The aggregate detection metrics are computed for the real-dataset detector-comparison experiments.

---

## Notebooks

### DNN-AE experiment

- `test_sea_dnn_ae.ipynb`

End-to-end experiment with the DNN + AE detector:

1. Data generation and visualisation
2. Model configuration and offline training (Models 1 and 3)
3. Online stream monitoring with `StreamMonitor`
4. Plots: reconstruction error, online training loss, rolling accuracy,
   per-concept error distributions
5. All figures exported as PDF to `figures/dnn_ae/`

### STUDD detector comparison

The following notebooks compare different internal drift detectors inside the STUDD framework:

- `test_abrupt_insects.ipynb`
- `test_detectors_bike.ipynb`
- `test_detectors_covtype.ipynb`
- `test_detectors_electricity.ipynb`
- `test_detectors_posture.ipynb`

For each dataset, the notebooks record:

- STUDD alarms
- supervised reference alarms
- teacher accuracy
- teacher-student agreement/disagreement
- detector-level alarm behaviour

### Summary metrics

- `detection_metrics_summary.ipynb`

This notebook contains classic alarm-based drift detection metrics for the real-dataset STUDD detector-comparison experiments.

The metrics include:

- Missed Detection Ratio (MDR)
- False Alarm Ratio (FAR)
- Mean Time to Detection (MTD)
- Mean Time Between False Alarms (MTFA)

The DNN-AE SEA experiment is reported separately and is not included in the aggregate STUDD detector-comparison metrics.

### Archive

The `archive/` folder contains older exploratory and reference notebooks that are kept for context but are not part of the main reported results.

This includes early STUDD experiments, older exploratory notebooks and reference notebooks used during development.

---

## Setup

```bash
# install dependencies (Python 3.11 required)
poetry install

# LaTeX rendering in figures requires:
sudo apt install texlive-latex-base texlive-fonts-recommended \
                 texlive-latex-extra cm-super dvipng
```

Start JupyterLab from the repository root:

```bash
poetry run jupyter lab
```

Open the notebooks from the `notebooks/` directory.