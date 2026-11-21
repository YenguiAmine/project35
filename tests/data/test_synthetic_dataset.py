import pandas as pd

from scripts.generate_synthetic_dataset import create_dataset


def test_synthetic_generator_is_reproducible_and_has_disjoint_populations(tmp_path):
    first = create_dataset(tmp_path / "first", train_rows=500, evaluation_rows=300, seed=91)
    second = create_dataset(tmp_path / "second", train_rows=500, evaluation_rows=300, seed=91)
    train = pd.read_csv(first["fit_population"]["path"])
    evaluation = pd.read_csv(first["gate_evaluation_population"]["path"])
    repeated = pd.read_csv(second["fit_population"]["path"])

    assert train.equals(repeated)
    assert len(train) == 500
    assert len(evaluation) == 300
    assert train["churn"].nunique() == evaluation["churn"].nunique() == 2
    train_rows = set(train.astype(str).agg("|".join, axis=1))
    evaluation_rows = set(evaluation.astype(str).agg("|".join, axis=1))
    assert train_rows.isdisjoint(evaluation_rows)
    assert first["synthetic"] is True
