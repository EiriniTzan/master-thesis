# Concept Drift Detection with Deep Neural Networks and Autoencoders

MSc thesis implementation of the algorithm proposed in:

> **Concept Drift Detection Based on Deep Neural Networks and Autoencoders**
> Lisha Hu, Yaru Lu, and Yuehua Feng

Synthetic data streams are generated with [CapyMOA](https://capymoa.org/), and the
STUDD baseline is also evaluated using CapyMOA's built-in implementation.

---

## Algorithm overview

The detector maintains three models that collaborate online:

| Model | Role |
|-------|------|
| **Model 1** — `DNNClassifier` | Offline classifier trained on reference data; frozen during the stream |
| **Model 2** — `StreamDNN` | Online classifier that continuously adapts to incoming samples |
| **Model 3** — `Autoencoder` | Reconstructs the latent space of Model 1; drift is signalled when reconstruction error exceeds a calibrated threshold |

The threshold is set from the distribution of reconstruction errors on the reference
set. During the stream, Model 2 also adapts the autoencoder online so that gradual
concept shift can be tracked before a hard alarm is raised.

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
  helpers.py                  # shared plotting and stream utilities
  test_sea_dnn_ae.ipynb       # DNN+AE detector evaluated on the SEA benchmark
  test_sea_studd.ipynb        # STUDD baseline evaluated on the same SEA stream
  figures/                    # exported PDF figures (auto-created on notebook run)
```

---

## Data streams

Both notebooks use identical capymoa-based data generation:

- **Reference set**: collected from a `CapySEA` generator (concept f4, θ = 9.5)
- **Stream**: a `DriftStream` that switches from f4 to f3 (θ = 7.0) at sample 5 000
- The stream is pre-collected into NumPy arrays and wrapped in `NumpyStream` for
  fast online evaluation (avoids per-sample JVM overhead)

---

## Notebooks

### `test_sea_dnn_ae.ipynb`

End-to-end experiment with the DNN + AE detector:

1. Data generation and visualisation
2. Model configuration and offline training (Models 1 and 3)
3. Online stream monitoring with `StreamMonitor`
4. Plots: reconstruction error, online training loss, rolling accuracy,
   per-concept error distributions
5. All figures exported as PDF to `figures/dnn_ae/`

### `test_sea_studd.ipynb`

Baseline experiment using CapyMOA's `STUDD` (Student–Teacher Uncertainty-based
Drift Detection):

1. Same SEA stream as the DNN+AE notebook
2. `HoeffdingTree` student / `ARF` teacher configuration
3. Plots: teacher–student agreement, rolling accuracy

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
