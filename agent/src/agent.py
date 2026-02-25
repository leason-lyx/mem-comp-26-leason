import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded  # type: ignore

from memory_store import MEMORY_FIELDS, MemoryStore
from memory_summarizer import MemorySummarizer


class MemoryAgent(DefaultAgent):
    def __init__(self, memory_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_dir = Path(memory_path)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store = MemoryStore(self.memory_dir / "memory.db")

        model_config = getattr(self.model, "config", None)
        model_kwargs = getattr(model_config, "model_kwargs", {}) or {}
        self.memory_summarizer = MemorySummarizer(
            api_base=str(model_kwargs.get("api_base", "")),
            api_key=str(model_kwargs.get("api_key", "")),
            model_name=str(getattr(model_config, "model_name", "")),
        )

        self.initial_top_k = 25
        self.runtime_cooldown_steps = 6
        self.max_runtime_injections = 3
        self.runtime_min_overlap = 3
        self.runtime_max_bm25 = 30.0

        self._reset_run_state("")

    def _reset_run_state(self, task: str) -> None:
        self.current_task_text = task
        self.injected_memory_ids: set[int] = set()
        self.last_injection_step = -(10**9)
        self.runtime_injection_count = 0
        self._memory_saved_for_run = False
        self._initial_memory_loaded = False

    @staticmethod
    def _compact_text(value: str, limit: int = 260) -> str:
        value = " ".join(value.split())
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _extract_recent_commands(self, limit: int = 8) -> list[str]:
        commands: list[str] = []
        for message in reversed(self.messages):
            if len(commands) >= limit:
                break
            extra = message.get("extra", {})
            if not isinstance(extra, dict):
                continue
            actions = extra.get("actions", [])
            if not isinstance(actions, list):
                continue
            for action in reversed(actions):
                if not isinstance(action, dict):
                    continue
                if len(commands) >= limit:
                    break
                command = str(action.get("command", "")).strip()
                if command:
                    commands.append(command)
        commands.reverse()
        return commands

    def _build_initial_query_text(self, task: str) -> str:
        parts = [task.strip()]
        if len(self.messages) > 1:
            content = self.messages[1].get("content", "")
            if isinstance(content, str):
                parts.append(content[:4000])
        return "\n".join(p for p in parts if p).strip()

    def _build_runtime_query_text(self) -> str:
        snippets = [self.current_task_text[:1200]]
        for message in self.messages[-10:]:
            role = message.get("role")
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "tool":
                snippets.append(content[:500])
            elif role in {"assistant", "user"}:
                snippets.append(content[:350])
        snippets.extend(self._extract_recent_commands(limit=6))
        return "\n".join(s for s in snippets if s).strip()

    def _format_memory_injection(
        self,
        records: list[dict[str, Any]],
        *,
        stage: str,
    ) -> str:
        lines = [
            f"External memory retrieved ({stage}). Use this as hints, not strict instructions.",
            "Only apply items relevant to the current task.",
            "",
        ]
        for idx, record in enumerate(records, start=1):
            lines.append(f"[Memory #{idx} | id={record.get('id')}]")
            for field in MEMORY_FIELDS:
                value = self._compact_text(str(record.get(field, "")))
                if value:
                    lines.append(f"- {field}: {value}")
            lines.append("")
        return "\n".join(lines).strip()

    def _print_memory_records(
        self,
        records: list[dict[str, Any]],
        *,
        stage: str,
        step: int,
    ) -> None:
        print(f"[memory] injected stage={stage} step={step} count={len(records)}")
        for idx, record in enumerate(records, start=1):
            print(
                f"[memory]   #{idx} id={record.get('id')} "
                f"overlap={record.get('overlap', '?')} bm25={record.get('bm25', '?')}"
            )
            for field in MEMORY_FIELDS:
                value = str(record.get(field, "")).strip()
                if value:
                    print(
                        f"[memory]     {field}: {self._compact_text(value, limit=280)}"
                    )

    def _is_high_runtime_match(self, record: dict[str, Any]) -> bool:
        overlap = int(record.get("overlap", 0))
        bm25_raw = record.get("bm25", 9999.0)
        try:
            bm25 = float(bm25_raw)
        except Exception:
            bm25 = 9999.0
        return overlap >= self.runtime_min_overlap and bm25 <= self.runtime_max_bm25

    def _runtime_debug(self, message: str) -> None:
        print(f"[memory][runtime-debug] {message}")

    def _maybe_inject_initial_memories(self) -> None:
        if self._initial_memory_loaded:
            return
        self._initial_memory_loaded = True

        query_text = self._build_initial_query_text(self.current_task_text)
        if not query_text:
            return
        records = self.memory_store.search(
            query_text,
            top_k=self.initial_top_k,
            candidate_k=self.initial_top_k * 4,
        )
        if not records:
            return

        message = self.model.format_message(
            role="user",
            content=self._format_memory_injection(records, stage="task-start"),
        )
        self.add_messages(message)  # type: ignore
        self.injected_memory_ids.update(int(record.get("id", 0)) for record in records)
        self.last_injection_step = self.n_calls
        self._print_memory_records(records, stage="task-start", step=self.n_calls)
        print(f"loaded {len(records)} external memories for task start")

    def _maybe_inject_runtime_memories(self) -> None:
        self._runtime_debug(
            "check start: "
            f"step={self.n_calls} "
            f"runtime_injection_count={self.runtime_injection_count}/{self.max_runtime_injections} "
            f"last_injection_step={self.last_injection_step}"
        )
        if self.runtime_injection_count >= self.max_runtime_injections:
            self._runtime_debug("skip: reached max runtime injections")
            return
        step_since_last = self.n_calls - self.last_injection_step
        if step_since_last < self.runtime_cooldown_steps:
            self._runtime_debug(
                "skip: cooldown active "
                f"(steps_since_last={step_since_last}, need>={self.runtime_cooldown_steps})"
            )
            return

        query_text = self._build_runtime_query_text()
        if len(query_text) < 24:
            self._runtime_debug(
                f"skip: query_text too short (len={len(query_text)}), "
                f"query_preview={self._compact_text(query_text, limit=120)}"
            )
            return

        self._runtime_debug(
            f"query ready (len={len(query_text)}), preview={self._compact_text(query_text, limit=200)}"
        )
        candidates = self.memory_store.search(query_text, top_k=3, candidate_k=10)
        self._runtime_debug(f"search returned {len(candidates)} candidate(s)")
        selected: dict[str, Any] | None = None
        for candidate in candidates:
            cid = int(candidate.get("id", 0))
            overlap = int(candidate.get("overlap", 0))
            try:
                bm25 = float(candidate.get("bm25", 9999.0))
            except Exception:
                bm25 = 9999.0

            if cid in self.injected_memory_ids:
                self._runtime_debug(f"reject candidate id={cid}: already injected")
                continue
            if not self._is_high_runtime_match(candidate):
                self._runtime_debug(
                    f"reject candidate id={cid}: "
                    f"overlap={overlap} (need>={self.runtime_min_overlap}), "
                    f"bm25={bm25} (need<={self.runtime_max_bm25})"
                )
                continue
            self._runtime_debug(
                f"accept candidate id={cid}: overlap={overlap}, bm25={bm25}"
            )
            selected = candidate
            break

        if selected is None:
            self._runtime_debug("skip: no candidate passed runtime filters")
            return

        message = self.model.format_message(
            role="user",
            content=self._format_memory_injection([selected], stage="runtime"),
        )
        self.add_messages(message)  # type: ignore
        self.injected_memory_ids.add(int(selected.get("id", 0)))
        self.last_injection_step = self.n_calls
        self.runtime_injection_count += 1
        self._print_memory_records([selected], stage="runtime", step=self.n_calls)
        print(
            "injected runtime memory id="
            f"{selected.get('id')} overlap={selected.get('overlap')} bm25={selected.get('bm25')}"
        )

    def _extract_exit_status(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "exit":
                extra = message.get("extra", {})
                if isinstance(extra, dict):
                    return str(extra.get("exit_status", "Unknown"))
                return "Unknown"
        return "Unknown"

    def save_memory(self) -> None:
        if self._memory_saved_for_run:
            return

        exit_status = self._extract_exit_status()
        summary = self.memory_summarizer.summarize(
            task=self.current_task_text,
            messages=self.messages,
            exit_status=exit_status,
        )

        created_at = datetime.now(timezone.utc).isoformat()
        task_hash = hashlib.sha256(self.current_task_text.encode("utf-8")).hexdigest()[
            :16
        ]
        record: dict[str, Any] = {
            "created_at": created_at,
            "exit_status": exit_status,
            "task_hash": task_hash,
            "summary_version": self.memory_summarizer.summary_version,
            **summary,
        }
        for field in MEMORY_FIELDS:
            record.setdefault(field, "")

        memory_id = self.memory_store.insert_memory(record)
        self._memory_saved_for_run = True
        print(
            f"[memory] saved id={memory_id} "
            f"exit_status={record.get('exit_status', '')} "
            f"task_hash={record.get('task_hash', '')}"
        )
        for field in MEMORY_FIELDS:
            value = str(record.get(field, "")).strip()
            if value:
                print(f"[memory]   {field}: {self._compact_text(value, limit=280)}")
        print(f"saved structured memory id={memory_id} (exit_status={exit_status})")

    def run(self, task: str = "", **kwargs) -> dict:
        self._reset_run_state(task)
        try:
            return super().run(task, **kwargs)
        finally:
            try:
                self.save_memory()
            except Exception as exc:
                print(f"failed to save memory: {type(exc).__name__}: {exc}")

    def query(self) -> dict:
        """Query the model and return the response."""

        if (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {
                        "exit_status": "LimitsExceeded",
                        "submission": "",
                    },
                }
            )

        self._maybe_inject_initial_memories()
        self._maybe_inject_runtime_memories()

        print(f"query llm: step {self.n_calls}")

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)  # type: ignore
        return message
