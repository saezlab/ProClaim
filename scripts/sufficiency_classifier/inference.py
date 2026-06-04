"""Runtime helpers for loading and scoring the sufficiency MLP."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from sufficiency_classifier.train_mlp_classifier import SufficiencyMLP


def flatten_features(agg: Mapping[str, object]) -> dict[str, float]:
    """Flatten nested aggregated features into a flat numeric mapping."""
    flat: dict[str, float] = {}
    for section in ("metadata_aggregation", "nlp_aggregation", "cross_features"):
        values = agg.get(section, {})
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if isinstance(value, (int, float)):
                flat[key] = float(value)
            else:
                flat[key] = 0.0
    return flat


def mlp_predict(
    model: torch.nn.Module,
    flat_features: Mapping[str, float],
    expected_features: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Run a single sufficiency prediction on pre-aggregated features."""
    vector = np.array(
        [float(flat_features.get(name, 0.0)) for name in expected_features],
        dtype=np.float32,
    )
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    safe_scale = np.where(scale == 0, np.float32(1.0), scale)
    scaled = ((vector - mean) / safe_scale).astype(np.float32, copy=False)

    inputs = torch.from_numpy(scaled).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probability = torch.sigmoid(logits).item()

    return float(probability), vector