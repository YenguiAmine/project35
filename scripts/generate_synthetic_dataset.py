from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_population(rows: int, *, seed: int) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("Chaque population doit contenir au moins 100 lignes.")
    rng = np.random.default_rng(seed)
    contract_type = rng.choice(["monthly", "annual", "multi_year"], rows, p=[0.52, 0.33, 0.15])
    tenure_months = rng.integers(1, 121, rows)
    company_size = np.clip(rng.lognormal(mean=3.4, sigma=0.9, size=rows).round(), 2, 5000).astype(int)
    unit_price = np.select(
        [contract_type == "monthly", contract_type == "annual"],
        [15.0, 12.5],
        default=10.0,
    )
    monthly_recurring_revenue = np.maximum(
        50.0,
        company_size * unit_price + rng.normal(0, 90, rows),
    ).round(2)
    support_tickets_90d = rng.poisson(
        0.7 + 0.45 * (contract_type == "monthly") + 0.12 * (tenure_months < 12), rows
    )
    payment_delay_days_90d = np.clip(
        rng.normal(2.5 + 4.2 * (contract_type == "monthly"), 6.0, rows).round(), 0, 45
    ).astype(int)
    product_usage_ratio_30d = rng.beta(2.4, 2.0, rows).clip(0, 1).round(4)
    cs_touches_90d = rng.poisson(1.0 + 0.7 * (contract_type != "monthly"), rows)
    auto_renew = rng.random(rows) < np.select(
        [contract_type == "monthly", contract_type == "annual"], [0.38, 0.76], default=0.92
    )

    # Synthetic labels depend probabilistically on observed behaviors/contract terms.
    # They are generated independently of row order and have no customer-level meaning.
    log_odds = (
        -2.25
        + 0.85 * (contract_type == "monthly")
        + 0.3 * (contract_type == "annual")
        + 0.85 * (tenure_months < 12)
        + 1.7 * (0.45 - product_usage_ratio_30d)
        + 0.075 * payment_delay_days_90d
        + 0.22 * support_tickets_90d
        - 0.13 * cs_touches_90d
        - 0.5 * auto_renew
    )
    churn_probability = 1.0 / (1.0 + np.exp(-log_odds))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "company_size": company_size,
            "monthly_recurring_revenue": monthly_recurring_revenue,
            "support_tickets_90d": support_tickets_90d,
            "payment_delay_days_90d": payment_delay_days_90d,
            "product_usage_ratio_30d": product_usage_ratio_30d,
            "cs_touches_90d": cs_touches_90d,
            "contract_type": contract_type,
            "industry": rng.choice(
                ["manufacturing", "technology", "professional_services", "retail", "healthcare"], rows
            ),
            "region": rng.choice(["north_america", "europe", "asia_pacific"], rows),
            "payment_method": rng.choice(["bank_transfer", "card", "invoice"], rows, p=[0.38, 0.34, 0.28]),
            "auto_renew": auto_renew,
            "churn": churn,
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_dataset(
    output_dir: str | Path = "data/synthetic",
    *,
    train_rows: int = 4000,
    evaluation_rows: int = 1000,
    seed: int = 3501,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = generate_population(train_rows, seed=seed)
    # A separate RNG stream creates an independently sampled gate holdout.
    evaluation = generate_population(evaluation_rows, seed=seed + 1_000_003)
    train_path = output / "b2b_churn_train.csv"
    evaluation_path = output / "b2b_churn_gate_evaluation.csv"
    train.to_csv(train_path, index=False, encoding="utf-8", float_format="%.6f")
    evaluation.to_csv(evaluation_path, index=False, encoding="utf-8", float_format="%.6f")
    manifest = {
        "dataset": "Project 35 synthetic B2B churn development dataset",
        "synthetic": True,
        "generator": "scripts/generate_synthetic_dataset.py",
        "seed": seed,
        "target_column": "churn",
        "positive_class": 1,
        "populations_are_independently_generated": True,
        "fit_population": {
            "path": str(train_path),
            "rows": int(len(train)),
            "class_counts": {str(k): int(v) for k, v in train["churn"].value_counts().sort_index().items()},
            "sha256": sha256(train_path),
        },
        "gate_evaluation_population": {
            "path": str(evaluation_path),
            "rows": int(len(evaluation)),
            "class_counts": {str(k): int(v) for k, v in evaluation["churn"].value_counts().sort_index().items()},
            "sha256": sha256(evaluation_path),
        },
        "limitations": [
            "Generated data are illustrative and contain no real customer observations.",
            "The target is simulated from a hand-authored probability rule.",
            "Metrics and explanations from this data are software demonstrations only, not business evidence.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Génère des populations synthétiques de démonstration, sans données réelles.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--evaluation-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3501)
    args = parser.parse_args()
    manifest = create_dataset(
        args.output_dir,
        train_rows=args.train_rows,
        evaluation_rows=args.evaluation_rows,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
