#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from metrics_aggregator import compute_harness_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute harness metrics for an existing run directory"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to harness/results/<run_name> directory containing eval_results.json",
    )
    return parser.parse_args()


def validate_agent_entry(agent_name: str, payload: object) -> tuple[str, dict[str, bool]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload for {agent_name}: expected object")

    output_dir = payload.get("output_dir")
    instance_results = payload.get("instance_results")

    if not isinstance(output_dir, str):
        raise ValueError(f"Invalid output_dir for {agent_name}: {output_dir!r}")
    if not isinstance(instance_results, dict):
        raise ValueError(f"Invalid instance_results for {agent_name}: {type(instance_results)!r}")

    normalized_results: dict[str, bool] = {}
    for instance_id, passed in instance_results.items():
        normalized_results[str(instance_id)] = bool(passed)

    return output_dir, normalized_results


def reorder_payload_with_metrics(
    payload: dict[str, object], harness_metrics: dict[str, int | float]
) -> dict[str, object]:
    reordered: dict[str, object] = {}
    inserted = False

    for key, value in payload.items():
        if key == "harness_metrics":
            continue
        if key == "instance_results" and not inserted:
            reordered["harness_metrics"] = harness_metrics
            inserted = True
        reordered[key] = value

    if not inserted:
        reordered["harness_metrics"] = harness_metrics

    return reordered


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_results_path = run_dir / "eval_results.json"

    if not eval_results_path.is_file():
        raise FileNotFoundError(f"eval_results.json not found: {eval_results_path}")

    with eval_results_path.open("r") as f:
        eval_results = json.load(f)

    if not isinstance(eval_results, dict):
        raise ValueError("eval_results.json must be a JSON object")

    updated = 0
    for agent_name, payload in eval_results.items():
        output_dir, instance_results = validate_agent_entry(agent_name, payload)
        harness_metrics = compute_harness_metrics(output_dir, instance_results)
        eval_results[agent_name] = reorder_payload_with_metrics(payload, harness_metrics)
        updated += 1

    with eval_results_path.open("w") as f:
        json.dump(eval_results, f, indent=2)
        f.write("\n")

    print(f"Updated harness_metrics for {updated} agent entries")
    print(f"Wrote: {eval_results_path}")


if __name__ == "__main__":
    main()
