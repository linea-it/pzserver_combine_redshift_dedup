import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from deduplication import validate_spatial_safety  # noqa: E402


def test_spatial_safety_accepts_radius_smaller_than_margin():
    validate_spatial_safety(0.5, 5.0, 0.8)


@pytest.mark.parametrize(
    ("radius", "margin", "warning_fraction", "message"),
    [
        (0.0, 5.0, 0.8, "crossmatch_radius_arcsec must be positive"),
        (0.5, 0.0, 0.8, "margin_threshold_arcsec must be positive"),
        (0.5, 5.0, 0.0, "margin_warning_fraction"),
        (0.5, 5.0, 1.01, "margin_warning_fraction"),
        (5.0, 5.0, 0.8, "must be smaller"),
        (6.0, 5.0, 0.8, "must be smaller"),
    ],
)
def test_spatial_safety_rejects_unsafe_configuration(
    radius, margin, warning_fraction, message
):
    with pytest.raises(ValueError, match=message):
        validate_spatial_safety(radius, margin, warning_fraction)
