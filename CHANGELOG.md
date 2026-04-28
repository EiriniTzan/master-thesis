# Changelog

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

`DNNBase` and `Autoencoder` now delegate construction to these builders.
Configuration uses a single `layer_sizes` list instead of separate
`input_dim / hidden_dims / output_dim` fields.

---

### `DNNBase` — three named attributes

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
- New `scripts/test_sea.py` and `scripts/test_sea.ipynb` for single-drift
  experiments with SEA concepts.
