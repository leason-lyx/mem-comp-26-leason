import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import sqlite3
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


FACT_FIELDS = [
    "issue_description",
    "issue_type",
    "tags",
    "outcome",
    "repository_context",
    "problem_signature",
    "relevant_artifacts",
    "evidence",
    "root_cause",
    "codebase_constraints",
    "fix_anchor",
    "verification",
]

EXPERIENCE_FIELDS = [
    "perspective",
    "localization_strategy",
    "modification_strategy",
    "decision_rationale",
    "reusable_procedure",
    "pitfalls",
    "failure_reflection",
    "transfer_rule",
]

ALL_MEMORY_FIELDS = FACT_FIELDS + EXPERIENCE_FIELDS
COMMON_MEMORY_FIELDS = ["issue_description", "issue_type", "tags", "outcome"]
CONTRIBUTION_LABELS = {
    "helpful",
    "partially_helpful",
    "not_used",
    "misleading",
    "insufficient_evidence",
    "not_available",
}
USAGE_MODES = {
    "problem_framing",
    "file_localization",
    "constraint_check",
    "solution_planning",
    "validation",
    "course_correction",
    "verification_focus",
    "misdirection",
}
MEMORY_INTERACTIONS = {
    "fact_only",
    "experience_only",
    "both_synergistic",
    "both_independent",
    "neither",
}
OVERALL_HELPFULNESS = {"high", "medium", "low", "none", "unclear"}
SUMMARY_PROMPT_PREFIX = "You summarize a coding-agent trajectory"
INJECTED_MEMORY_MARKER = "External memory retrieved"
MAX_EVIDENCE_ITEMS = 40
MAX_EVIDENCE_CHARS = 20_000
MAX_INSTANCE_EVIDENCE = 5
REQUEST_TIMEOUT = 180
DEFAULT_BASE_URL = os.getenv("QWEN_BASE_URL", "http://100.124.83.129:46400/v1")
DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "api_key")
DEFAULT_MODEL = os.getenv("QWEN_MODEL", "Qwen3-Coder-Next")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch trajectory memory attribution analyzer."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an external-mem-* run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to analysis/out/<run-dir-name>.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI-compatible model name. Defaults to QWEN_MODEL or built-in Qwen value.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL. Defaults to QWEN_BASE_URL or built-in endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="OpenAI-compatible API key. Defaults to QWEN_API_KEY or built-in key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N instances.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent instance analyses. Defaults to 1.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip instances with existing outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing instance outputs.",
    )
    return parser.parse_args()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def detect_run_mode(run_dir: Path) -> str:
    run_key = str(run_dir).lower()
    if "onlyfact" in run_key:
        return "onlyfact"
    if "onlyexp" in run_key:
        return "onlyexp"
    return "full"


def validate_cli_args(args: argparse.Namespace) -> None:
    if not args.run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {args.run_dir}")
    if not args.model:
        raise ValueError("--model or ANALYZE_MODEL is required")
    if not args.base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")


def list_instance_dirs(run_dir: Path) -> list[Path]:
    instance_dirs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not re.fullmatch(r"p\d{2}i\d{2}", child.name):
            continue
        if (child / "instance.json").exists() and (child / "_harness" / "agent.log").exists():
            instance_dirs.append(child)
    return instance_dirs


def is_summary_traj(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    first = messages[0]
    content = first.get("content")
    if first.get("role") != "system" or not isinstance(content, str):
        return False
    return content.startswith(SUMMARY_PROMPT_PREFIX)


def select_traj_files(traj_dir: Path) -> tuple[Path | None, Path | None]:
    history_path = None
    summary_path = None
    for path in sorted(traj_dir.glob("*.json")):
        data = read_json(path)
        messages = data.get("messages") or []
        if is_summary_traj(messages):
            summary_path = path
        else:
            history_path = path
    return history_path, summary_path


def parse_startup_fields(agent_log_text: str) -> list[str]:
    match = re.search(r"\[memory\] startup fields=(\[[^\n]+\])", agent_log_text)
    if not match:
        return []
    loaded = safe_json_loads(match.group(1))
    return loaded if isinstance(loaded, list) else []


def parse_agent_log(agent_log_path: Path) -> dict[str, Any]:
    text = agent_log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    injected_events: list[dict[str, Any]] = []
    saved_memories: list[dict[str, Any]] = []
    current_event: dict[str, Any] | None = None
    final_status = None

    injected_pattern = re.compile(
        r"\[memory\] injected stage=(?P<stage>\S+) step=(?P<step>\d+) count=(?P<count>\d+)"
    )
    injected_id_pattern = re.compile(r"\[memory\]\s+#\d+\s+id=(?P<id>\d+)")
    saved_pattern = re.compile(
        r"\[memory\] saved id=(?P<id>\d+) exit_status=(?P<status>\S+) task_hash=(?P<task_hash>\S+)"
    )
    done_pattern = re.compile(r"done!\s+status:\s+(?P<status>.+)$")

    for idx, line in enumerate(lines):
        injected_match = injected_pattern.search(line)
        if injected_match:
            current_event = {
                "stage": injected_match.group("stage"),
                "step": int(injected_match.group("step")),
                "count": int(injected_match.group("count")),
                "ids": [],
                "line_index": idx,
            }
            injected_events.append(current_event)
            continue

        id_match = injected_id_pattern.search(line)
        if id_match and current_event is not None:
            current_event["ids"].append(int(id_match.group("id")))
            continue

        saved_match = saved_pattern.search(line)
        if saved_match:
            saved_memories.append(
                {
                    "id": int(saved_match.group("id")),
                    "exit_status": saved_match.group("status"),
                    "task_hash": saved_match.group("task_hash"),
                    "line_index": idx,
                }
            )
            current_event = None
            continue

        done_match = done_pattern.search(line)
        if done_match:
            final_status = done_match.group("status").strip()
            current_event = None
            continue

        if "[memory]" not in line:
            current_event = None

    if final_status is None and saved_memories:
        final_status = saved_memories[-1]["exit_status"]

    return {
        "text": text,
        "lines": lines,
        "startup_fields": parse_startup_fields(text),
        "injected_events": injected_events,
        "saved_memories": saved_memories,
        "final_status": final_status or "UNKNOWN",
    }


def load_memories(memory_db_path: Path) -> dict[int, dict[str, Any]]:
    if not memory_db_path.exists():
        return {}
    conn = sqlite3.connect(memory_db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("select * from memories order by id").fetchall()
    finally:
        conn.close()
    result = {}
    for row in rows:
        result[int(row["id"])] = {key: row[key] for key in row.keys()}
    return result


def get_task_title(problem_statement: str) -> str:
    for line in problem_statement.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Title"):
            continue
        if stripped:
            return stripped
    return ""


def parse_artifact_list(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    loaded = safe_json_loads(value)
    if isinstance(loaded, list):
        return [str(item) for item in loaded if str(item).strip()]
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def extract_fix_anchor_paths(value: str) -> list[str]:
    if not value:
        return []
    matches = re.findall(r"[\w./-]+\.[A-Za-z0-9_]+", value)
    return list(dict.fromkeys(matches))


def collect_keyword_terms(injected_memories: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for memory in injected_memories:
        for artifact in parse_artifact_list(str(memory.get("relevant_artifacts", ""))):
            terms.append(artifact)
            terms.append(Path(artifact).name)
        for anchor in extract_fix_anchor_paths(str(memory.get("fix_anchor", ""))):
            terms.append(anchor)
            terms.append(Path(anchor).name)
        tags = str(memory.get("tags", ""))
        for tag in tags.split(","):
            tag = tag.strip()
            if len(tag) >= 3:
                terms.append(tag)
        issue_type = str(memory.get("issue_type", "")).strip()
        if issue_type:
            terms.append(issue_type)
    deduped = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key in seen or len(key) < 3:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def add_candidate(
    candidate_map: dict[int, dict[str, Any]],
    messages: list[dict[str, Any]],
    idx: int,
    priority: int,
    source_type: str,
    reason: str,
) -> None:
    if idx < 0 or idx >= len(messages):
        return
    message = messages[idx]
    content = flatten_message_content(message.get("content"))
    snippet = truncate_text(content, 1200)
    if not snippet:
        return
    current = candidate_map.get(idx)
    if current is None:
        candidate_map[idx] = {
            "message_index": idx,
            "role": message.get("role", "unknown"),
            "source_type": source_type,
            "priority": priority,
            "reasons": [reason],
            "snippet": snippet,
        }
        return
    previous_priority = current["priority"]
    current["priority"] = max(current["priority"], priority)
    if reason not in current["reasons"]:
        current["reasons"].append(reason)
    if source_type != current["source_type"] and priority > previous_priority:
        current["source_type"] = source_type


def extract_evidence_bundle(
    messages: list[dict[str, Any]], injected_memories: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_map: dict[int, dict[str, Any]] = {}
    injection_indexes = []
    keywords = [term.lower() for term in collect_keyword_terms(injected_memories)]

    for idx, message in enumerate(messages):
        content = flatten_message_content(message.get("content"))
        lowered = content.lower()
        if INJECTED_MEMORY_MARKER.lower() in lowered:
            injection_indexes.append(idx)
            add_candidate(
                candidate_map,
                messages,
                idx,
                100,
                "memory_reference",
                "显式注入 memory 文本",
            )
        if "memory" in lowered and message.get("role") in {"user", "assistant"}:
            add_candidate(
                candidate_map,
                messages,
                idx,
                95,
                "memory_reference",
                "显式提到 memory",
            )

    if injection_indexes:
        first_injection = injection_indexes[0]
        for idx in range(first_injection + 1, min(len(messages), first_injection + 7)):
            add_candidate(
                candidate_map,
                messages,
                idx,
                92,
                "post_injection",
                "注入后前 6 条消息",
            )

    for idx, message in enumerate(messages):
        content = flatten_message_content(message.get("content"))
        lowered = content.lower()
        if keywords and any(term in lowered for term in keywords):
            for neighbor in (idx - 1, idx, idx + 1):
                add_candidate(
                    candidate_map,
                    messages,
                    neighbor,
                    85,
                    "command_context",
                    "命中文件/标签/锚点关键词",
                )

    tail_start = max(0, len(messages) - 8)
    for idx in range(tail_start, len(messages)):
        message = messages[idx]
        content = flatten_message_content(message.get("content"))
        lowered = content.lower()
        priority = 70
        if "git diff" in lowered or "complete_task_and_submit_final_output" in lowered:
            priority = 80
        if "verify" in lowered or "validation" in lowered or "py_compile" in lowered:
            priority = 82
        add_candidate(
            candidate_map,
            messages,
            idx,
            priority,
            "final_verification",
            "收尾验证/提交阶段",
        )

    prioritized = sorted(
        candidate_map.values(),
        key=lambda item: (-item["priority"], item["message_index"]),
    )

    selected: list[dict[str, Any]] = []
    used_chars = 0
    for item in prioritized:
        snippet_len = len(item["snippet"])
        if len(selected) >= MAX_EVIDENCE_ITEMS:
            break
        if selected and used_chars + snippet_len > MAX_EVIDENCE_CHARS:
            continue
        selected.append(item)
        used_chars += snippet_len

    selected.sort(key=lambda item: item["message_index"])
    for item in selected:
        item["reason"] = "；".join(item.pop("reasons"))
    return selected


def availability_for_group(run_mode: str, group: str) -> bool:
    if group == "fact":
        return run_mode != "onlyexp"
    if group == "experience":
        return run_mode != "onlyfact"
    raise ValueError(f"unknown group: {group}")


def summarize_memory(memory: dict[str, Any], run_mode: str) -> dict[str, Any]:
    fact_available = availability_for_group(run_mode, "fact")
    exp_available = availability_for_group(run_mode, "experience")
    fact_present = [field for field in FACT_FIELDS if str(memory.get(field, "")).strip()]
    exp_present = [field for field in EXPERIENCE_FIELDS if str(memory.get(field, "")).strip()]
    families = []
    if fact_available and fact_present:
        families.append("fact")
    if exp_available and exp_present:
        families.append("experience")
    return {
        "id": memory["id"],
        "issue_description_excerpt": truncate_text(
            str(memory.get("issue_description", "")), 240
        ),
        "tags": str(memory.get("tags", "")),
        "available_field_families": families,
        "fact_fields_present": fact_present if fact_available else [],
        "experience_fields_present": exp_present if exp_available else [],
    }


def build_instance_context(instance_dir: Path, run_mode: str) -> dict[str, Any]:
    instance = read_json(instance_dir / "instance.json")
    harness_dir = instance_dir / "_harness"
    agent_log = parse_agent_log(harness_dir / "agent.log")
    history_traj_path, summary_traj_path = select_traj_files(harness_dir / "traj")
    if history_traj_path is None:
        raise FileNotFoundError(f"no non-summary trajectory found in {instance_dir}")
    history_traj = read_json(history_traj_path)
    history_messages = history_traj.get("messages") or []
    memories = load_memories(harness_dir / "memory" / "memory.db")

    injected_ids = []
    for event in agent_log["injected_events"]:
        injected_ids.extend(event["ids"])
    injected_ids = list(dict.fromkeys(injected_ids))
    saved_ids = [item["id"] for item in agent_log["saved_memories"]]

    injected_memories = []
    missing_memory_ids = []
    for memory_id in injected_ids:
        memory = memories.get(memory_id)
        if memory is None:
            missing_memory_ids.append(memory_id)
            continue
        injected_memories.append(memory)

    evidence_bundle = extract_evidence_bundle(history_messages, injected_memories)
    task_title = get_task_title(str(instance.get("problem_statement", "")))

    return {
        "instance_dir": str(instance_dir),
        "instance_name": instance_dir.name,
        "repo": instance.get("repo", ""),
        "repo_language": instance.get("repo_language", ""),
        "task_title": task_title,
        "problem_statement": instance.get("problem_statement", ""),
        "requirements": instance.get("requirements", ""),
        "interface": instance.get("interface", ""),
        "agent_status": agent_log["final_status"],
        "run_mode": run_mode,
        "startup_fields": agent_log["startup_fields"],
        "injected_events": agent_log["injected_events"],
        "injected_memory_ids": injected_ids,
        "saved_memory_ids": saved_ids,
        "missing_memory_ids": missing_memory_ids,
        "memories_by_id": memories,
        "injected_memories": injected_memories,
        "injected_memory_summaries": [
            summarize_memory(memory, run_mode) for memory in injected_memories
        ],
        "history_traj_path": str(history_traj_path),
        "summary_traj_path": str(summary_traj_path) if summary_traj_path else None,
        "history_message_count": len(history_messages),
        "summary_traj_exists": summary_traj_path is not None,
        "evidence_bundle": evidence_bundle,
    }


def build_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    fact_available = availability_for_group(context["run_mode"], "fact")
    exp_available = availability_for_group(context["run_mode"], "experience")
    schema = {
        "instance_id": "string",
        "repo": "string",
        "task_title": "string",
        "agent_status": "string",
        "injected_memories": [
            {
                "id": 1,
                "available_field_families": ["fact", "experience"],
                "issue_description_excerpt": "string",
                "tags": "string",
            }
        ],
        "fact_contribution": {
            "label": sorted(CONTRIBUTION_LABELS),
            "confidence": 0.0,
            "summary_zh": "string",
            "used_fields": FACT_FIELDS,
            "usage_modes": sorted(USAGE_MODES),
            "evidence": [
                {
                    "memory_field": FACT_FIELDS,
                    "memory_content": "string",
                    "step": 123,
                    "source_type": "memory_reference|command_context|post_injection|final_verification",
                    "helping_mode": sorted(USAGE_MODES),
                    "agent_action_zh": "string",
                    "snippet": "string",
                    "why_zh": "string",
                }
            ],
        },
        "experience_contribution": {
            "label": sorted(CONTRIBUTION_LABELS),
            "confidence": 0.0,
            "summary_zh": "string",
            "used_fields": EXPERIENCE_FIELDS,
            "usage_modes": sorted(USAGE_MODES),
            "evidence": [
                {
                    "memory_field": EXPERIENCE_FIELDS,
                    "memory_content": "string",
                    "step": 123,
                    "source_type": "memory_reference|command_context|post_injection|final_verification",
                    "helping_mode": sorted(USAGE_MODES),
                    "agent_action_zh": "string",
                    "snippet": "string",
                    "why_zh": "string",
                }
            ],
        },
        "memory_interaction": sorted(MEMORY_INTERACTIONS),
        "counterfactuals": {
            "without_fact_zh": "string",
            "without_experience_zh": "string",
        },
        "final_assessment": {
            "overall_helpfulness": sorted(OVERALL_HELPFULNESS),
            "summary_zh": "string",
            "key_takeaway_zh": "string",
            "representative_memory_ids": [1],
        },
    }

    user_payload = {
        "task": {
            "instance_id": context["instance_name"],
            "repo": context["repo"],
            "task_title": context["task_title"],
            "agent_status": context["agent_status"],
            "run_mode": context["run_mode"],
            "fact_memory_available": fact_available,
            "experience_memory_available": exp_available,
            "history_message_count": context["history_message_count"],
            "startup_fields": context["startup_fields"],
            "problem_statement_excerpt": truncate_text(context["problem_statement"], 1200),
            "requirements_excerpt": truncate_text(context["requirements"], 1200),
        },
        "injected_memories": context["injected_memory_summaries"],
        "evidence_bundle": context["evidence_bundle"],
        "output_schema": schema,
        "rules": [
            "输出必须是严格 JSON，对象顶层不能有额外文字。",
            "证据必须只引用 evidence_bundle 里的片段，不要杜撰。",
            "如果某类记忆在该 run 中不可用，则对应 contribution.label 必须是 not_available，used_fields 与 evidence 为空数组。",
            "不要写泛泛而谈的帮助描述。只有当你能明确指出“哪段 memory 内容，在哪一步，通过什么方式影响了 agent”时，才能判定 helpful 或 partially_helpful。",
            "如果拿不出上述具体案例，就返回 not_used；只有轨迹证据本身严重不足时才用 insufficient_evidence。",
            "used_fields 只能填写对应字段组内的字段名。",
            "usage_modes 只能从给定枚举中选择。",
            "每条 evidence 都必须填写 memory_field、memory_content、step、helping_mode、agent_action_zh、snippet、why_zh。",
            "step 必须使用 evidence_bundle 里的 message_index。",
            "evidence 最多 5 条，每条 snippet 要短，why_zh 要明确解释其如何支持判断。",
        ],
    }

    system = textwrap.dedent(
        """
        You analyze whether injected memories helped a coding agent solve a task.
        You must separate fact-memory contribution from experience-memory contribution.
        Return strict JSON only. Do not wrap in markdown fences.
        """
    ).strip()
    user = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_response_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices in LLM response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    raise ValueError("missing message content in LLM response")


class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key or "EMPTY"
        self.session = requests.Session()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        response = self.session.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return parse_response_content(response.json())


def normalize_evidence_items(items: Any, allowed_fields: list[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items[:MAX_INSTANCE_EVIDENCE]:
        if not isinstance(item, dict):
            continue
        memory_field = str(item.get("memory_field", "")).strip()
        if memory_field not in allowed_fields:
            continue
        memory_content = truncate_text(str(item.get("memory_content", "")), 240)
        try:
            step = int(item.get("step"))
        except (TypeError, ValueError):
            continue
        snippet = truncate_text(str(item.get("snippet", "")), 500)
        why = truncate_text(str(item.get("why_zh", "")), 200)
        source_type = str(item.get("source_type", "")).strip() or "unknown"
        helping_mode = str(item.get("helping_mode", "")).strip()
        if helping_mode not in USAGE_MODES:
            continue
        agent_action = truncate_text(str(item.get("agent_action_zh", "")), 200)
        if not snippet or not memory_content or not agent_action:
            continue
        normalized.append(
            {
                "memory_field": memory_field,
                "memory_content": memory_content,
                "step": step,
                "source_type": source_type,
                "helping_mode": helping_mode,
                "agent_action_zh": agent_action,
                "snippet": snippet,
                "why_zh": why,
            }
        )
    return normalized


def normalize_contribution(
    payload: Any,
    allowed_fields: list[str],
    available: bool,
) -> dict[str, Any]:
    default_label = "insufficient_evidence" if available else "not_available"
    if not isinstance(payload, dict):
        payload = {}

    label = str(payload.get("label", default_label)).strip()
    if label not in CONTRIBUTION_LABELS:
        label = default_label
    if not available:
        label = "not_available"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    used_fields = payload.get("used_fields")
    if not isinstance(used_fields, list):
        used_fields = []
    used_fields = [field for field in used_fields if field in allowed_fields]
    if not available:
        used_fields = []

    usage_modes = payload.get("usage_modes")
    if not isinstance(usage_modes, list):
        usage_modes = []
    usage_modes = [mode for mode in usage_modes if mode in USAGE_MODES]
    if not available:
        usage_modes = []

    evidence = normalize_evidence_items(payload.get("evidence"), allowed_fields)
    if not available:
        evidence = []

    if available and label in {"helpful", "partially_helpful", "misleading"} and not evidence:
        label = "not_used"
        used_fields = []
        usage_modes = []

    return {
        "label": label,
        "confidence": confidence,
        "summary_zh": str(payload.get("summary_zh", "")).strip(),
        "used_fields": used_fields,
        "usage_modes": usage_modes,
        "evidence": evidence,
    }


def make_stub_analysis(context: dict[str, Any], analysis_error: str) -> dict[str, Any]:
    fact_available = availability_for_group(context["run_mode"], "fact")
    exp_available = availability_for_group(context["run_mode"], "experience")
    interaction = "neither"
    if fact_available and not exp_available:
        interaction = "fact_only"
    elif exp_available and not fact_available:
        interaction = "experience_only"
    return {
        "instance_id": context["instance_name"],
        "repo": context["repo"],
        "task_title": context["task_title"],
        "agent_status": context["agent_status"],
        "injected_memories": context["injected_memory_summaries"],
        "fact_contribution": normalize_contribution({}, FACT_FIELDS, fact_available),
        "experience_contribution": normalize_contribution(
            {}, EXPERIENCE_FIELDS, exp_available
        ),
        "memory_interaction": interaction,
        "counterfactuals": {
            "without_fact_zh": "LLM 分析失败，无法稳定判断缺少 fact 记忆后的影响。",
            "without_experience_zh": "LLM 分析失败，无法稳定判断缺少 experience 记忆后的影响。",
        },
        "final_assessment": {
            "overall_helpfulness": "unclear",
            "summary_zh": "LLM 分析失败，当前结果仅保留预处理后的上下文与占位结论。",
            "key_takeaway_zh": "需要重新运行或检查模型配置。",
            "representative_memory_ids": context["injected_memory_ids"][:3],
        },
        "analysis_error": analysis_error,
    }


def normalize_analysis(
    raw_payload: Any,
    context: dict[str, Any],
    analysis_error: str | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return make_stub_analysis(context, analysis_error or "response is not a JSON object")

    fact_available = availability_for_group(context["run_mode"], "fact")
    exp_available = availability_for_group(context["run_mode"], "experience")

    interaction = str(raw_payload.get("memory_interaction", "")).strip()
    if interaction not in MEMORY_INTERACTIONS:
        interaction = "neither"
    if fact_available and not exp_available:
        interaction = "fact_only"
    elif exp_available and not fact_available:
        interaction = "experience_only"

    final_assessment = raw_payload.get("final_assessment")
    if not isinstance(final_assessment, dict):
        final_assessment = {}
    overall_helpfulness = str(final_assessment.get("overall_helpfulness", "unclear")).strip()
    if overall_helpfulness not in OVERALL_HELPFULNESS:
        overall_helpfulness = "unclear"
    representative_memory_ids = final_assessment.get("representative_memory_ids")
    if not isinstance(representative_memory_ids, list):
        representative_memory_ids = []
    representative_memory_ids = [
        memory_id for memory_id in representative_memory_ids if memory_id in context["injected_memory_ids"]
    ]

    counterfactuals = raw_payload.get("counterfactuals")
    if not isinstance(counterfactuals, dict):
        counterfactuals = {}

    normalized = {
        "instance_id": context["instance_name"],
        "repo": context["repo"],
        "task_title": context["task_title"],
        "agent_status": context["agent_status"],
        "injected_memories": context["injected_memory_summaries"],
        "fact_contribution": normalize_contribution(
            raw_payload.get("fact_contribution"), FACT_FIELDS, fact_available
        ),
        "experience_contribution": normalize_contribution(
            raw_payload.get("experience_contribution"),
            EXPERIENCE_FIELDS,
            exp_available,
        ),
        "memory_interaction": interaction,
        "counterfactuals": {
            "without_fact_zh": str(counterfactuals.get("without_fact_zh", "")).strip(),
            "without_experience_zh": str(
                counterfactuals.get("without_experience_zh", "")
            ).strip(),
        },
        "final_assessment": {
            "overall_helpfulness": overall_helpfulness,
            "summary_zh": str(final_assessment.get("summary_zh", "")).strip(),
            "key_takeaway_zh": str(final_assessment.get("key_takeaway_zh", "")).strip(),
            "representative_memory_ids": representative_memory_ids,
        },
        "analysis_error": analysis_error,
    }
    return normalized


def validate_analysis_shape(analysis: dict[str, Any]) -> None:
    required_keys = {
        "instance_id",
        "repo",
        "task_title",
        "agent_status",
        "injected_memories",
        "fact_contribution",
        "experience_contribution",
        "memory_interaction",
        "counterfactuals",
        "final_assessment",
        "analysis_error",
    }
    missing = required_keys - set(analysis.keys())
    if missing:
        raise ValueError(f"normalized analysis missing keys: {sorted(missing)}")


def analyze_with_llm(
    client: OpenAICompatibleClient,
    context: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    prompt_messages = build_prompt(context)
    raw_attempts: list[str] = []
    last_error = ""

    for _ in range(3):
        raw_response = client.chat(prompt_messages, temperature=0.0)
        raw_attempts.append(raw_response)
        parsed = safe_json_loads(raw_response)
        if isinstance(parsed, dict):
            normalized = normalize_analysis(parsed, context)
            validate_analysis_shape(normalized)
            return normalized, "\n\n==== ATTEMPT ====\n\n".join(raw_attempts)
        last_error = "response is not valid JSON"

    repair_system = (
        "You repair a malformed model response into strict JSON. "
        "Return only a single valid JSON object matching the requested schema."
    )
    repair_user = json.dumps(
        {
            "original_prompt": build_prompt(context)[1]["content"],
            "bad_response": raw_attempts[-1] if raw_attempts else "",
        },
        ensure_ascii=False,
        indent=2,
    )
    repair_response = client.chat(
        [{"role": "system", "content": repair_system}, {"role": "user", "content": repair_user}],
        temperature=0.0,
    )
    raw_attempts.append(repair_response)
    parsed = safe_json_loads(repair_response)
    if isinstance(parsed, dict):
        normalized = normalize_analysis(parsed, context)
        validate_analysis_shape(normalized)
        return normalized, "\n\n==== ATTEMPT ====\n\n".join(raw_attempts)

    stub = make_stub_analysis(context, last_error or "repair response is not valid JSON")
    return stub, "\n\n==== ATTEMPT ====\n\n".join(raw_attempts)


def render_instance_markdown(
    result: dict[str, Any],
    context: dict[str, Any],
) -> str:
    fact = result["fact_contribution"]
    exp = result["experience_contribution"]
    lines = [
        f"# {result['instance_id']} 记忆归因分析",
        "",
        f"- Repo: `{result['repo']}`",
        f"- Task: {result['task_title'] or 'N/A'}",
        f"- Agent 状态: `{result['agent_status']}`",
        f"- Run 模式: `{context['run_mode']}`",
        f"- History trajectory: `{context['history_traj_path']}`",
        "",
        "## 注入记忆",
        "",
    ]
    if result["injected_memories"]:
        for memory in result["injected_memories"]:
            families = ", ".join(memory["available_field_families"]) or "none"
            lines.extend(
                [
                    f"- Memory #{memory['id']} | families: `{families}` | tags: `{memory['tags']}`",
                    f"  {memory['issue_description_excerpt']}",
                ]
            )
    else:
        lines.append("- 未解析到注入记忆。")

    lines.extend(
        [
            "",
            "## Fact 记忆作用",
            "",
            f"- Label: `{fact['label']}`",
            f"- Confidence: `{fact['confidence']:.2f}`",
            f"- Summary: {fact['summary_zh'] or 'N/A'}",
            f"- Used fields: {', '.join(fact['used_fields']) or 'none'}",
            f"- Usage modes: {', '.join(fact['usage_modes']) or 'none'}",
            "",
            "## Experience 记忆作用",
            "",
            f"- Label: `{exp['label']}`",
            f"- Confidence: `{exp['confidence']:.2f}`",
            f"- Summary: {exp['summary_zh'] or 'N/A'}",
            f"- Used fields: {', '.join(exp['used_fields']) or 'none'}",
            f"- Usage modes: {', '.join(exp['usage_modes']) or 'none'}",
            "",
            "## 联合作用",
            "",
            f"- Interaction: `{result['memory_interaction']}`",
            f"- Overall helpfulness: `{result['final_assessment']['overall_helpfulness']}`",
            f"- Summary: {result['final_assessment']['summary_zh'] or 'N/A'}",
            f"- Key takeaway: {result['final_assessment']['key_takeaway_zh'] or 'N/A'}",
            "",
            "## Counterfactual",
            "",
            f"- 无 fact 记忆: {result['counterfactuals']['without_fact_zh'] or 'N/A'}",
            f"- 无 experience 记忆: {result['counterfactuals']['without_experience_zh'] or 'N/A'}",
            "",
            "## 证据",
            "",
        ]
    )

    for title, contribution in [("Fact", fact), ("Experience", exp)]:
        lines.append(f"### {title}")
        if contribution["evidence"]:
            for evidence in contribution["evidence"]:
                lines.extend(
                    [
                        f"- step `{evidence['step']}` | field=`{evidence['memory_field']}` | mode=`{evidence['helping_mode']}`",
                        f"  memory 内容: {evidence['memory_content']}",
                        f"  agent 动作: {evidence['agent_action_zh']}",
                        f"  作用说明: {evidence['why_zh']}",
                        f"  轨迹片段: `{evidence['snippet']}`",
                    ]
                )
        else:
            lines.append("- 无")
        lines.append("")

    if result.get("analysis_error"):
        lines.extend(["## 分析错误", "", f"- {result['analysis_error']}", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_run_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Run 记忆归因报告: {summary['run_name']}",
        "",
        f"- Run 路径: `{summary['run_dir']}`",
        f"- Run 模式: `{summary['run_mode']}`",
        f"- 模型: `{summary['model']}`",
        f"- 实例数: `{summary['instance_count']}`",
        f"- 生成时间: `{summary['generated_at']}`",
        "",
        "## 整体结论",
        "",
        f"- Fact 标签分布: {json.dumps(summary['fact_label_counts'], ensure_ascii=False)}",
        f"- Experience 标签分布: {json.dumps(summary['experience_label_counts'], ensure_ascii=False)}",
        f"- 联合作用分布: {json.dumps(summary['memory_interaction_counts'], ensure_ascii=False)}",
        "",
        "## Repo 差异",
        "",
    ]
    for repo, repo_summary in summary["by_repo"].items():
        lines.append(
            f"- `{repo}`: {repo_summary['instance_count']} 个实例, "
            f"fact={json.dumps(repo_summary['fact_label_counts'], ensure_ascii=False)}, "
            f"experience={json.dumps(repo_summary['experience_label_counts'], ensure_ascii=False)}"
        )

    lines.extend(["", "## 常见字段", ""])
    lines.append(
        f"- Fact 常见字段: {json.dumps(summary['top_fact_used_fields'], ensure_ascii=False)}"
    )
    lines.append(
        f"- Experience 常见字段: {json.dumps(summary['top_experience_used_fields'], ensure_ascii=False)}"
    )

    lines.extend(["", "## 代表性实例", ""])
    for item in summary["representative_instances"][:10]:
        lines.append(
            f"- `{item['instance_id']}` | repo=`{item['repo']}` | "
            f"fact=`{item['fact_label']}` | experience=`{item['experience_label']}` | "
            f"interaction=`{item['memory_interaction']}` | {item['summary_zh']}"
        )

    return "\n".join(lines).rstrip() + "\n"


def aggregate_results(
    per_instance_results: list[dict[str, Any]],
    run_dir: Path,
    run_mode: str,
    model: str,
) -> dict[str, Any]:
    fact_counts = Counter()
    exp_counts = Counter()
    interaction_counts = Counter()
    matrix: dict[str, Counter] = defaultdict(Counter)
    by_repo: dict[str, dict[str, Any]] = {}
    fact_fields_counter = Counter()
    exp_fields_counter = Counter()
    representative_instances = []

    for payload in per_instance_results:
        analysis = payload["analysis"]
        repo = analysis["repo"]
        fact_label = analysis["fact_contribution"]["label"]
        exp_label = analysis["experience_contribution"]["label"]
        interaction = analysis["memory_interaction"]

        fact_counts[fact_label] += 1
        exp_counts[exp_label] += 1
        interaction_counts[interaction] += 1
        matrix[fact_label][exp_label] += 1

        repo_entry = by_repo.setdefault(
            repo,
            {
                "instance_count": 0,
                "fact_label_counts": Counter(),
                "experience_label_counts": Counter(),
            },
        )
        repo_entry["instance_count"] += 1
        repo_entry["fact_label_counts"][fact_label] += 1
        repo_entry["experience_label_counts"][exp_label] += 1

        fact_fields_counter.update(analysis["fact_contribution"]["used_fields"])
        exp_fields_counter.update(analysis["experience_contribution"]["used_fields"])

        representative_instances.append(
            {
                "instance_id": analysis["instance_id"],
                "repo": repo,
                "fact_label": fact_label,
                "experience_label": exp_label,
                "memory_interaction": interaction,
                "summary_zh": analysis["final_assessment"]["summary_zh"],
                "overall_helpfulness": analysis["final_assessment"]["overall_helpfulness"],
                "fact_confidence": analysis["fact_contribution"]["confidence"],
                "experience_confidence": analysis["experience_contribution"]["confidence"],
            }
        )

    def counter_to_dict(counter: Counter) -> dict[str, int]:
        return {key: counter[key] for key in sorted(counter)}

    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "run_mode": run_mode,
        "model": model,
        "generated_at": utcnow(),
        "instance_count": len(per_instance_results),
        "fact_label_counts": counter_to_dict(fact_counts),
        "experience_label_counts": counter_to_dict(exp_counts),
        "memory_interaction_counts": counter_to_dict(interaction_counts),
        "fact_experience_matrix": {
            fact_label: counter_to_dict(exp_counter)
            for fact_label, exp_counter in sorted(matrix.items())
        },
        "by_repo": {
            repo: {
                "instance_count": repo_entry["instance_count"],
                "fact_label_counts": counter_to_dict(repo_entry["fact_label_counts"]),
                "experience_label_counts": counter_to_dict(
                    repo_entry["experience_label_counts"]
                ),
            }
            for repo, repo_entry in sorted(by_repo.items())
        },
        "top_fact_used_fields": fact_fields_counter.most_common(10),
        "top_experience_used_fields": exp_fields_counter.most_common(10),
        "representative_instances": sorted(
            representative_instances,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2, "none": 3, "unclear": 4}.get(
                    item["overall_helpfulness"], 5
                ),
                -max(item["fact_confidence"], item["experience_confidence"]),
                item["instance_id"],
            ),
        )[:10],
    }
    return summary


def output_paths(output_dir: Path, instance_name: str) -> dict[str, Path]:
    return {
        "instance_json": output_dir / "instances" / f"{instance_name}.json",
        "instance_md": output_dir / "instances" / f"{instance_name}.md",
        "evidence_json": output_dir / "artifacts" / f"{instance_name}.evidence.json",
        "raw_response": output_dir / "artifacts" / f"{instance_name}.raw_response.txt",
    }


def should_skip_instance(paths: dict[str, Path], args: argparse.Namespace) -> bool:
    if args.force:
        return False
    if not args.resume:
        return False
    return paths["instance_json"].exists() and paths["instance_md"].exists()


def analyze_instance(
    instance_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    run_mode = detect_run_mode(args.run_dir)
    paths = output_paths(output_dir, instance_dir.name)
    if should_skip_instance(paths, args):
        return {"instance_name": instance_dir.name, "skipped": True}

    try:
        context = build_instance_context(instance_dir, run_mode)
        client = OpenAICompatibleClient(args.model, args.base_url, args.api_key)
        analysis, raw_response = analyze_with_llm(client, context)
        validate_analysis_shape(analysis)
    except Exception as exc:
        instance = {}
        if (instance_dir / "instance.json").exists():
            try:
                instance = read_json(instance_dir / "instance.json")
            except Exception:
                instance = {}
        fallback_context = {
            "instance_name": instance_dir.name,
            "repo": instance.get("repo", ""),
            "task_title": get_task_title(str(instance.get("problem_statement", ""))),
            "agent_status": "UNKNOWN",
            "run_mode": run_mode,
            "injected_memory_summaries": [],
            "injected_memory_ids": [],
            "instance_dir": str(instance_dir),
            "history_traj_path": "",
            "summary_traj_path": None,
            "history_message_count": 0,
            "startup_fields": [],
            "saved_memory_ids": [],
            "missing_memory_ids": [],
            "evidence_bundle": [],
        }
        analysis = make_stub_analysis(fallback_context, str(exc))
        raw_response = ""
        context = fallback_context

    result_payload = {
        "metadata": {
            "generated_at": utcnow(),
            "instance_dir": context["instance_dir"],
            "run_mode": run_mode,
            "history_traj_path": context["history_traj_path"],
            "summary_traj_path": context["summary_traj_path"],
            "history_message_count": context["history_message_count"],
            "startup_fields": context["startup_fields"],
            "injected_memory_ids": context["injected_memory_ids"],
            "saved_memory_ids": context["saved_memory_ids"],
            "missing_memory_ids": context["missing_memory_ids"],
        },
        "analysis": analysis,
    }
    evidence_payload = {
        "instance_id": context["instance_name"],
        "repo": context["repo"],
        "task_title": context["task_title"],
        "injected_memories": context["injected_memory_summaries"],
        "evidence_bundle": context["evidence_bundle"],
    }

    write_json(paths["instance_json"], result_payload)
    write_text(paths["instance_md"], render_instance_markdown(analysis, context))
    write_json(paths["evidence_json"], evidence_payload)
    write_text(paths["raw_response"], raw_response)

    return {
        "instance_name": instance_dir.name,
        "skipped": False,
        "result_path": str(paths["instance_json"]),
    }


def load_finished_results(output_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((output_dir / "instances").glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, dict) and "analysis" in payload:
            results.append(payload)
    return results


def main() -> None:
    args = parse_args()
    validate_cli_args(args)

    output_dir = args.output_dir or (Path("analysis") / "out" / args.run_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_dirs = list_instance_dirs(args.run_dir)
    if args.limit is not None:
        instance_dirs = instance_dirs[: args.limit]
    if not instance_dirs:
        raise RuntimeError(f"no valid instance directories found in {args.run_dir}")

    print(
        f"[analyze_traj] run_dir={args.run_dir} instances={len(instance_dirs)} "
        f"workers={args.workers} run_mode={detect_run_mode(args.run_dir)}"
    )

    worker = lambda path: analyze_instance(path, args, output_dir)
    if args.workers == 1:
        for instance_dir in instance_dirs:
            result = worker(instance_dir)
            print(
                f"[analyze_traj] {result['instance_name']} "
                f"{'skipped' if result.get('skipped') else 'done'}"
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(worker, instance_dir): instance_dir.name
                for instance_dir in instance_dirs
            }
            for future in concurrent.futures.as_completed(futures):
                instance_name = futures[future]
                result = future.result()
                print(
                    f"[analyze_traj] {instance_name} "
                    f"{'skipped' if result.get('skipped') else 'done'}"
                )

    finished_results = load_finished_results(output_dir)
    summary = aggregate_results(
        finished_results,
        args.run_dir,
        detect_run_mode(args.run_dir),
        args.model,
    )
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "report.md", render_run_report(summary))
    print(f"[analyze_traj] wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
