import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TypeVar
from rich.progress import track

_INSTANCE_ID_RE = re.compile(r"instance_id\s*=\s*(\S+)")
_STEP_RE = re.compile(r"query llm: step\s+(\d+)")


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def parse_system_log(system_log_path: Path) -> tuple[str | None, float]:
    instance_id: str | None = None
    key_usage = 0.0

    try:
        with system_log_path.open("r", errors="replace") as f:
            for line in f:
                if instance_id is None and "instance_id" in line:
                    match = _INSTANCE_ID_RE.search(line)
                    if match:
                        instance_id = match.group(1)
                        continue

                if line.startswith("llm usage:"):
                    payload = line.split("llm usage:", 1)[1].strip()
                    try:
                        usage = ast.literal_eval(payload)
                    except (ValueError, SyntaxError):
                        continue

                    if isinstance(usage, Mapping):
                        key_usage = _to_float(usage.get("key_usage", 0.0))
    except FileNotFoundError:
        return None, 0.0

    return instance_id, key_usage


def parse_agent_steps(agent_log_path: Path) -> int:
    max_step = -1

    try:
        with agent_log_path.open("r", errors="replace") as f:
            for line in f:
                match = _STEP_RE.search(line)
                if not match:
                    continue
                step = _to_int(match.group(1))
                if step > max_step:
                    max_step = step
    except FileNotFoundError:
        return 0

    return max_step + 1 if max_step >= 0 else 0


def _iter_with_optional_progress(
    items: Iterable[Path], description: str, show_progress: bool
) -> Iterable[Path]:
    if not show_progress:
        return items
    return track(items, description=description)


def resolve_instance_cost(item: Mapping[str, Any]) -> float:
    if bool(item.get("has_traj_cost", False)):
        return _to_float(item.get("traj_cost", 0.0))
    return _to_float(item.get("key_usage", 0.0))


def build_instance_metric_rows(
    instance_results: Mapping[str, bool],
    metrics_by_instance: Mapping[str, Mapping[str, int | float | bool]],
) -> list[dict[str, int | float | bool | str]]:
    rows: list[dict[str, int | float | bool | str]] = []

    for instance_id, passed in instance_results.items():
        item = metrics_by_instance.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "passed": bool(passed),
                "steps": _to_int(item.get("steps", 0)),
                "cost": resolve_instance_cost(item),
                "input_tokens": _to_int(item.get("input_tokens", 0)),
                "output_tokens": _to_int(item.get("output_tokens", 0)),
                "cache_tokens": _to_int(item.get("cache_tokens", 0)),
                "has_traj_cost": bool(item.get("has_traj_cost", False)),
                "key_usage": _to_float(item.get("key_usage", 0.0)),
                "traj_cost": _to_float(item.get("traj_cost", 0.0)),
            }
        )

    return rows


def parse_traj_dir(traj_dir: Path, show_progress: bool = True) -> tuple[int, int, int, float, bool]:
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_tokens = 0
    total_cost = 0.0
    has_cost = False

    if not traj_dir.is_dir():
        return 0, 0, 0, 0.0, False

    traj_paths = sorted(traj_dir.glob("*.json"))
    for traj_path in _iter_with_optional_progress(
        traj_paths,
        description=f"Parsing traj files ({traj_dir.parent.parent.name})",
        show_progress=show_progress,
    ):
        try:
            with traj_path.open("r") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        usage = payload.get("usage_object")
        if not isinstance(usage, Mapping):
            usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            continue

        total_input_tokens += _to_int(usage.get("prompt_tokens", 0))
        total_output_tokens += _to_int(usage.get("completion_tokens", 0))

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, Mapping):
            total_cache_tokens += _to_int(prompt_details.get("cached_tokens", 0))

        cost_breakdown = payload.get("cost_breakdown")
        if isinstance(cost_breakdown, Mapping) and "total_cost" in cost_breakdown:
            total_cost += _to_float(cost_breakdown.get("total_cost", 0.0))
            has_cost = True

    return total_input_tokens, total_output_tokens, total_cache_tokens, total_cost, has_cost


def collect_instance_metrics(
    output_dir: str | Path, show_progress: bool = True
) -> dict[str, dict[str, int | float | bool]]:
    root = Path(output_dir)
    metrics_by_instance: dict[str, dict[str, int | float]] = {}

    instance_dirs = sorted(root.glob("p??i??"))
    for instance_dir in _iter_with_optional_progress(
        instance_dirs,
        description="Collecting instance metrics",
        show_progress=show_progress,
    ):
        harness_dir = instance_dir / "_harness"
        instance_id, key_usage = parse_system_log(harness_dir / "system.log")
        if not instance_id:
            continue

        steps = parse_agent_steps(harness_dir / "agent.log")
        input_tokens, output_tokens, cache_tokens, traj_cost, has_traj_cost = parse_traj_dir(
            harness_dir / "traj", show_progress=show_progress
        )

        metrics_by_instance[instance_id] = {
            "steps": steps,
            "key_usage": key_usage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_tokens": cache_tokens,
            "traj_cost": traj_cost,
            "has_traj_cost": has_traj_cost,
        }

    return metrics_by_instance


def aggregate_harness_metrics(
    instance_results: Mapping[str, bool],
    metrics_by_instance: Mapping[str, Mapping[str, int | float | bool]],
) -> dict[str, int | float]:
    instance_ids = list(instance_results.keys())
    passed_instance_ids = [
        instance_id for instance_id, passed in instance_results.items() if passed
    ]
    failed_instance_ids = [
        instance_id for instance_id, passed in instance_results.items() if not passed
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_tokens = 0
    total_cost = 0.0
    total_cost_from_traj = 0.0
    total_cost_from_key_usage = 0.0
    total_steps = 0
    total_steps_passed = 0
    total_steps_failed = 0
    total_cost_passed = 0.0
    total_cost_failed = 0.0
    instances_with_metrics = 0
    instances_with_traj_cost = 0
    instances_using_key_usage_fallback = 0

    for instance_id in instance_ids:
        item = metrics_by_instance.get(instance_id, {})
        if instance_id in metrics_by_instance:
            instances_with_metrics += 1

        steps = _to_int(item.get("steps", 0))
        total_input_tokens += _to_int(item.get("input_tokens", 0))
        total_output_tokens += _to_int(item.get("output_tokens", 0))
        total_cache_tokens += _to_int(item.get("cache_tokens", 0))
        key_usage = _to_float(item.get("key_usage", 0.0))
        total_cost_from_key_usage += key_usage

        if bool(item.get("has_traj_cost", False)):
            traj_cost = _to_float(item.get("traj_cost", 0.0))
            instance_cost = traj_cost
            total_cost += instance_cost
            total_cost_from_traj += traj_cost
            instances_with_traj_cost += 1
        else:
            instance_cost = key_usage
            total_cost += instance_cost
            instances_using_key_usage_fallback += 1

        total_steps += steps
        if instance_results.get(instance_id, False):
            total_steps_passed += steps
            total_cost_passed += instance_cost
        else:
            total_steps_failed += steps
            total_cost_failed += instance_cost

    denominator_instances = len(instance_ids)
    denominator_passed_instances = len(passed_instance_ids)
    denominator_failed_instances = len(failed_instance_ids)

    average_cost = total_cost / denominator_instances if denominator_instances else 0.0
    average_steps = total_steps / denominator_instances if denominator_instances else 0.0
    average_steps_passed = (
        total_steps_passed / denominator_passed_instances
        if denominator_passed_instances
        else 0.0
    )
    average_steps_failed = (
        total_steps_failed / denominator_failed_instances
        if denominator_failed_instances
        else 0.0
    )
    average_cost_passed = (
        total_cost_passed / denominator_passed_instances
        if denominator_passed_instances
        else 0.0
    )
    average_cost_failed = (
        total_cost_failed / denominator_failed_instances
        if denominator_failed_instances
        else 0.0
    )

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_tokens": total_cache_tokens,
        "total_cost": total_cost,
        "total_cost_passed": total_cost_passed,
        "total_cost_failed": total_cost_failed,
        "total_cost_from_traj": total_cost_from_traj,
        "total_cost_from_key_usage": total_cost_from_key_usage,
        "average_cost": average_cost,
        "average_cost_passed": average_cost_passed,
        "average_cost_failed": average_cost_failed,
        "average_steps": average_steps,
        "average_steps_passed": average_steps_passed,
        "average_steps_failed": average_steps_failed,
        "instances_with_metrics": instances_with_metrics,
        "instances_with_traj_cost": instances_with_traj_cost,
        "instances_using_key_usage_fallback": instances_using_key_usage_fallback,
        "denominator_instances": denominator_instances,
        "denominator_passed_instances": denominator_passed_instances,
        "denominator_failed_instances": denominator_failed_instances,
    }


def aggregate_instance_metrics(
    instance_results: Mapping[str, bool],
    metrics_by_instance: Mapping[str, Mapping[str, int | float | bool]],
) -> dict[str, dict[str, int | float | bool]]:
    rows = build_instance_metric_rows(instance_results, metrics_by_instance)
    instance_metrics: dict[str, dict[str, int | float | bool]] = {}

    for row in rows:
        instance_id = str(row["instance_id"])
        instance_metrics[instance_id] = {
            "passed": bool(row["passed"]),
            "steps": _to_int(row["steps"]),
            "cost": _to_float(row["cost"]),
            "input_tokens": _to_int(row["input_tokens"]),
            "output_tokens": _to_int(row["output_tokens"]),
            "cache_tokens": _to_int(row["cache_tokens"]),
            "has_traj_cost": bool(row["has_traj_cost"]),
        }

    return instance_metrics


def compute_metrics_bundle(
    output_dir: str | Path,
    instance_results: Mapping[str, bool],
    show_progress: bool = True,
) -> tuple[dict[str, int | float], dict[str, dict[str, int | float | bool]]]:
    metrics_by_instance = collect_instance_metrics(output_dir, show_progress=show_progress)
    harness_metrics = aggregate_harness_metrics(instance_results, metrics_by_instance)
    instance_metrics = aggregate_instance_metrics(instance_results, metrics_by_instance)
    return harness_metrics, instance_metrics


def compute_harness_metrics(
    output_dir: str | Path,
    instance_results: Mapping[str, bool],
    show_progress: bool = True,
) -> dict[str, int | float]:
    harness_metrics, _ = compute_metrics_bundle(
        output_dir, instance_results, show_progress=show_progress
    )
    return harness_metrics
