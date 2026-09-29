# Detector Adapter Interface

HDR-SC leaves the detector architecture, feature extraction, image-score
aggregation and map postprocessing unchanged. An adapter exposes:

```python
class DetectorAdapter:
    name: str

    def prepare(self, pool_paths):
        ...  # Cache support-independent normal-pool features only.

    def fit(self, support_indices):
        ...  # Replace the support-conditioned detector state.

    def score(self, evaluation_indices):
        ...  # Return one native image score per requested normal image.

    def anomaly_map(self, evaluation_index):
        ...  # Return ONE finite nonnegative H x W NumPy map.
```

The current fit state applies to both score and anomaly_map. Map inputs are
always from the same leave-support-out complement used for image risk. Return
the detector's native map, not a color heatmap or a test-normalized response.
Do not substitute map maxima for the detector's native image score.

## Callable Wrapper

```python
from hdr_sc import CallableDetectorAdapter

adapter = CallableDetectorAdapter(
    name="my-detector",
    prepare_pool=extract_pool_features,   # (paths) -> pool_state
    fit_support=build_normal_model,      # (pool_state, indices) -> model
    score_normal=score_normal_images,    # (pool_state, model, indices) -> scores
    map_normal=map_normal_image,         # (pool_state, model, index) -> H x W
)
```

The map callback is required by the updated API. Old image-score-only adapters
must add it; fabricating a map from the image score is not a valid integration.

## Streaming and GPU Safety

Perform inference under `torch.inference_mode()` or the detector's equivalent.
Transfer only the current map to CPU, detach it, and release its GPU tensors
before returning. Never retain maps in the adapter. The selector reduces and
deletes one map before requesting the next; it retains only per-image scalars.
No candidate x image map cache is created. Native features may remain cached.
The core is NumPy-only; GPU inference is the external adapter's responsibility.

Image scoring may use small batches but must return scalar scores, not all
normal maps. Refit must release old support-dependent detector state. If a
retained set has only one candidate, no spatial maps are requested.

## Detector-Specific Responsibilities

- SubspaceAD: construct the support-conditioned normal subspace; expose native
  reconstruction image scores and anomaly maps.
- PatchCore: construct its support-conditioned feature memory; expose its
  native image-score aggregation and postprocessed anomaly maps.
- AnomalyDINO: construct its normal-reference memory; expose its native image
  scores and spatial responses.

These are integration contracts, not bundled or validated detector adapters.
Record external code versions and preserve preprocessing, feature layers,
memory compression, resolution, map interpolation and smoothing in each run.
Keep support and validation images disjoint. Never access test paths, anomaly
labels or defect masks during support construction. Use detector-native caches
only when detector settings, ordered pool and support identity all match.
