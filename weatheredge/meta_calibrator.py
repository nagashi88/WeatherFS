"""Meta-model calibration pipeline — WeatherEdge PRD v3.1 Phase 3.

Replaces ad-hoc blending weights with a two-stage calibration pipeline:
  Stage 1: LogisticRegression maps raw model features to a raw probability.
  Stage 2: IsotonicRegression maps raw probability to a calibrated probability.

Preferred feature vector (6 features):
  [gfs_prob, ecmwf_prob, hrrr_prob, timesfm_prob, lead_hours, month]

Extended feature vector (9 features, only when 500+ training samples):
  [gfs_prob, ecmwf_prob, hrrr_prob, timesfm_prob, lead_hours, month,
   hour_of_day, spread, volume]

Minimum training samples: max(300, n_features * 50).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)

# Feature counts
_PREFERRED_N_FEATURES = 6
_EXTENDED_N_FEATURES = 9
_EXTENDED_THRESHOLD = 500


class MetaCalibrator:
    """Two-stage meta-model calibrator (logistic regression + isotonic regression).

    Usage::

        calibrator = MetaCalibrator()
        calibrator.train(features, outcomes)
        prob = calibrator.predict(features_row)
    """

    def __init__(self) -> None:
        self.model: Optional["LogisticRegression"] = None
        self.isotonic: Optional["IsotonicRegression"] = None

    def train(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit the two-stage calibration pipeline.

        Args:
            features: 2-D array of shape (n_samples, n_features).
                      6 preferred features or 9 extended features.
            outcomes: 1-D binary array of shape (n_samples,) with values in {0, 1}.

        Raises:
            RuntimeError: If sklearn is not installed.
            ValueError: If the sample count is below the required minimum.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required for MetaCalibrator.train(). "
                "Install it with: pip install scikit-learn"
            )

        features = np.asarray(features)
        outcomes = np.asarray(outcomes)

        n_samples, n_features = features.shape

        # Determine active feature count based on sample size
        if n_samples >= _EXTENDED_THRESHOLD and n_features == _EXTENDED_N_FEATURES:
            active_n_features = _EXTENDED_N_FEATURES
        else:
            active_n_features = _PREFERRED_N_FEATURES

        min_samples = max(300, active_n_features * 50)
        if n_samples < min_samples:
            raise ValueError(
                f"Insufficient training data: {n_samples} samples provided, "
                f"minimum required is {min_samples} "
                f"(max(300, {active_n_features} features * 50))."
            )

        # Stage 1: logistic regression on raw features
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(features, outcomes)
        raw_probs = lr.predict_proba(features)[:, 1]

        # Stage 2: isotonic regression to calibrate raw probabilities
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, outcomes)

        self.model = lr
        self.isotonic = iso

        log.info(
            "MetaCalibrator trained: %d samples, %d features, "
            "extended_features=%s",
            n_samples,
            active_n_features,
            active_n_features == _EXTENDED_N_FEATURES,
        )

    def predict(self, features: np.ndarray) -> Optional[float]:
        """Return a calibrated probability for a single feature row.

        Args:
            features: 1-D or 2-D array matching the feature count used at training.

        Returns:
            Calibrated probability in [0, 1], or None if the model has not been
            trained yet (caller should fall back to blended_probability).
        """
        if not self.is_ready():
            return None

        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        raw_prob = self.model.predict_proba(features)[:, 1]
        calibrated = self.isotonic.transform(raw_prob)
        return float(calibrated[0])

    def is_ready(self) -> bool:
        """Return True if the model has been trained and is ready for inference."""
        return self.model is not None and self.isotonic is not None
