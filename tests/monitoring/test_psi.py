import pandas as pd
import pytest

from project35.monitoring.psi import PSIError, PSIProfile


@pytest.fixture
def reference_frame():
    return pd.DataFrame({
        "balance": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "segment": ["A", "B", "A", "B", "A", "B"],
    })


def test_same_distribution_is_stable_and_reuses_frozen_bins(reference_frame):
    profile = PSIProfile.fit(reference_frame, bins=4)
    report = profile.calculate(reference_frame.copy(), current_window="same-period")

    assert report["status"] == "stable"
    assert report["alert"] is False
    assert report["features"]["balance"]["psi"] == pytest.approx(0.0)
    assert report["features"]["balance"]["bin_definition"]["internal_edges"] == profile.features["balance"].edges
    assert report["reference_sha256"] == profile.reference_sha256
    assert len(report["current_sha256"]) == 64


def test_shifted_numeric_and_new_categories_trigger_drift(reference_frame):
    profile = PSIProfile.fit(reference_frame, bins=4)
    current = pd.DataFrame({"balance": [500.0, 600.0, 700.0], "segment": ["NEW"] * 3})

    report = profile.calculate(current)

    assert report["status"] == "drift"
    assert report["alert"] is True
    assert report["features"]["balance"]["status"] == "drift"
    assert report["features"]["segment"]["current_proportions"]["special:other"] > 0.99


def test_missing_values_are_a_separate_monitoring_bin(reference_frame):
    profile = PSIProfile.fit(reference_frame, bins=3)
    report = profile.calculate(pd.DataFrame({"balance": [None, None], "segment": [None, None]}))

    assert report["features"]["balance"]["current_proportions"]["special:missing"] > 0.99
    assert report["features"]["segment"]["current_proportions"]["special:missing"] > 0.99


@pytest.mark.parametrize(
    "current",
    [
        pd.DataFrame({"balance": [10.0]}),
        pd.DataFrame({"balance": ["not-numeric"], "segment": ["A"]}),
        pd.DataFrame({"balance": [float("inf")], "segment": ["A"]}),
    ],
)
def test_invalid_window_is_rejected(reference_frame, current):
    with pytest.raises(PSIError):
        PSIProfile.fit(reference_frame).calculate(current)


def test_invalid_reference_settings_are_rejected(reference_frame):
    with pytest.raises(PSIError, match="bins"):
        PSIProfile.fit(reference_frame, bins=1)
