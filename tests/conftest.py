import pandas as pd
import pytest


@pytest.fixture
def sample_records():
    # Synthetic records are test fixtures only; they are not the Project 35 training dataset.
    return [
        {"age": 24, "segment": "A", "target": 0},
        {"age": 31, "segment": "B", "target": 1},
        {"age": 42, "segment": "A", "target": 0},
        {"age": 53, "segment": "C", "target": 1},
    ]


@pytest.fixture
def sample_frame(sample_records):
    return pd.DataFrame.from_records(sample_records)
