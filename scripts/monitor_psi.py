from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project35.monitoring.psi import PSIError, PSIProfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare une fenêtre courante à une référence PSI figée.")
    parser.add_argument("--reference", required=True, help="CSV de référence stable")
    parser.add_argument("--current", required=True, help="CSV de la fenêtre à surveiller")
    parser.add_argument("--output", default="reports/monitoring/psi_report.json")
    parser.add_argument("--window", default=None, help="Identifiant lisible de la fenêtre courante")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--watch-threshold", type=float, default=0.10)
    parser.add_argument("--drift-threshold", type=float, default=0.25)
    args = parser.parse_args()

    try:
        reference = pd.read_csv(args.reference)
        current = pd.read_csv(args.current)
        profile = PSIProfile.fit(
            reference,
            bins=args.bins,
            epsilon=args.epsilon,
            watch_threshold=args.watch_threshold,
            drift_threshold=args.drift_threshold,
        )
        # Keep the CLI's provenance hash byte-for-byte consistent with the API loader.
        profile.reference_sha256 = hashlib.sha256(Path(args.reference).read_bytes()).hexdigest()
        report = profile.calculate(
            current,
            current_window=args.window,
            observed_at=datetime.now(timezone.utc).isoformat(),
        )
    except (OSError, ValueError, PSIError) as exc:
        parser.error(str(exc))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
