# Changelog

## 2026-05-06

### Naming aligned with paper (Model 1 / Model 2 / Model 3)

**Class and module rename**

`DNNBase` (in `tscls/models/base_dnn.py`) renamed to `DNNClassifier` (in
`tscls/models/dnn_classifier.py`). `StreamDNN` and `Autoencoder` keep their
descriptive names. All imports updated accordingly.

**Config field rename (`OptimizationConfig`, `TrainingConfig`)**

| Old | New |
|-----|-----|
| `base_loss_name` / `base_optimizer_name` / `base_learning_rate` | `model1_loss_name` / `model1_optimizer_name` / `model1_learning_rate` |
| `stream_loss_name` / `stream_optimizer_name` / `stream_learning_rate` | `model2_loss_name` / `model2_optimizer_name` / `model2_learning_rate` |
| `ae_loss_name` / `ae_optimizer_name` / `ae_learning_rate` | `model3_loss_name` / `model3_optimizer_name` / `model3_learning_rate` |
| `base_epochs` / `ae_epochs` | `model1_epochs` / `model3_epochs` |

**`Detector` attribute rename**

| Old | New |
|-----|-----|
| `self.base_model` | `self.model1` (`DNNClassifier`) |
| `self.stream_model` | `self.model2` (`StreamDNN`) |
| `self.autoencoder` | `self.model3` (`Autoencoder`) |
| `self.base_train_losses` / `self.ae_train_losses` | `self.model1_train_losses` / `self.model3_train_losses` |

`PipelineResult` fields updated to match (`model1_train_losses`,
`model3_train_losses`).

**Autoencoder architecture (`encoder_sizes`)**

Changed from `[64, 32, 8]` to `[64, 8]` to match paper Table 5 (L\_A = 8,
single hidden layer bottleneck).

**SEA stream length**

Changed from 5 000 to 10 000 stream samples with drift at index 5 000,
matching paper Table 1 (SEA\_a configuration).

---

## 2026-04-28

### Architectural refactoring — `Detector` and `StreamMonitor`

`DNNAEDDPipeline` was replaced by two focused classes.

**`tscls/pipeline/detector.py` — `Detector`**
- `fit(X, y)` — offline phase: trains base DNN, trains AE on latents,
  calibrates threshold, clones stream DNN, builds all optimisers.
- `detect(sample)` — one online step: scale → stream DNN forward → AE error →
  threshold check → last-hidden-layer update → `StepResult`.

**`tscls/pipeline/monitor.py` — `StreamMonitor`**
- `run(stream)` — iterates a `StreamSimulator`, calls `detector.detect()` per
  sample, aggregates into `PipelineResult`.

Deleted: `dnn_ae_dd_pipeline.py`, `tscls/core/state.py` (unused `PipelineState`),
old simulation scripts.

---

### Builder pattern (`tscls/models/builders.py`)

- **`FeedforwardBuilder`** — builds an `nn.Sequential` from a `layer_sizes`
  list; Xavier-uniform init on all linear layers.
- **`AutoencoderBuilder`** — composes two `FeedforwardBuilder`s (encoder +
  mirrored decoder), returns `(encoder, decoder)`.

`DNNClassifier` and `Autoencoder` now delegate construction to these builders.
Configuration uses a single `layer_sizes` list instead of separate
`input_dim / hidden_dims / output_dim` fields.

---

### `DNNClassifier` — three named attributes

```python
self.body               # nn.Sequential — first hidden layers + activations
self.last_hidden_layer  # nn.Sequential — last hidden linear + its activation
self.output_layer       # nn.Linear     — final projection, no activation
```

The latent representation passed to the AE is the output of `last_hidden_layer`.

---

### `StreamDNN` layer freezing (Algorithm 3)

Per Algorithm 3 of the paper, only `last_hidden_layer` (W(L)\_F, b(L)\_F) is
updated during streaming. `body` and `output_layer` are frozen.

`StreamDNN` references `stream_model.last_hidden_layer` by name for
freezing/unfreezing.

> **Note:** Drift detection on SEA data is still under investigation.
> False positives appear before the true drift point. The root cause
> (threshold calibration vs. algorithm behaviour) has not yet been resolved.

---

### Other changes

- `DriftDetector` renamed to `AEDetector` in `tscls/detection/ae_detector.py`.
- `StreamConfig` removed from `pipeline_config.py` (no longer needed).
- New `tscls/datasets/` module with a `SEA` dataset class.
- New `scripts/test_sea.ipynb` for single-drift experiments with SEA concepts.
