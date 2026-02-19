from pathlib import Path
from datetime import datetime, timezone

import requests
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded  # type: ignore
from memory_logging import build_one_line_memory_hint, emit_memory_log


class AgenticMemoryAgent(DefaultAgent):
    AGENTS_STARTUP_MAX_CHARS = 12000
    AGENTS_TEMPLATE = """# AGENTS Memory

## Project Structure & Module Organization

## Build, Test, and Development Commands

## Testing Guidelines

## Coding Style & Naming Conventions

"""

    def __init__(self, memory_path: str, *args, **kwargs):
        usage_warning_ratio = self._to_float(kwargs.pop("usage_warning_ratio", 0.9))
        usage_warning_message = kwargs.pop("usage_warning_message")
        super().__init__(*args, **kwargs)
        self.memory_root = Path(memory_path)
        self.agents_md_path = self.memory_root / "AGENTS.md"
        self.topic_dir = self.memory_root / "cards"
        self.usage_warning_ratio = (
            usage_warning_ratio
            if usage_warning_ratio is not None and usage_warning_ratio > 0
            else 0.9
        )
        self.usage_warning_message = str(usage_warning_message)
        self._ensure_memory_layout()

    def _emit_memory_event(
        self,
        event: str,
        *,
        command: str = "",
        returncode: int = 0,
        is_write: bool = False,
        stdout: str = "",
        stderr: str = "",
        path_hint: str = "",
        output_len: int | None = None,
        details: dict | None = None,
    ) -> None:
        hint = build_one_line_memory_hint(
            event,
            command,
            returncode,
            is_write,
        )
        print(hint, flush=True)
        emit_memory_log(
            event,
            command=command,
            returncode=returncode,
            is_write=is_write,
            stdout=stdout,
            stderr=stderr,
            path_hint=path_hint or str(self.memory_root),
            output_len=output_len,
            details=details,
        )

    def _ensure_memory_layout(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        if not self.agents_md_path.exists():
            self.agents_md_path.write_text(self.AGENTS_TEMPLATE.strip() + "\n")
            self._emit_memory_event(
                "memory_bootstrap_created",
                command="bootstrap",
                path_hint=str(self.agents_md_path),
            )

    def _read_text_with_limit(self, path: Path, max_chars: int) -> str:
        if not path.exists():
            return "(missing)"
        text = path.read_text(errors="replace")
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        truncated = len(text) - (half * 2)
        return f"{text[:half]}\n...<TRUNCATED {truncated} CHARS>...\n{text[-half:]}"

    def _build_memory_context_message(self) -> tuple[str, int]:
        agents_text = self._read_text_with_limit(
            self.agents_md_path, self.AGENTS_STARTUP_MAX_CHARS
        )
        topic_count = sum(1 for _ in self.topic_dir.glob("*.md"))
        content = (
            "Persistent memory is available at `/mnt/memory`.\n"
            "Read AGENTS.md first, then discover relevant topic files via listing and keyword search.\n\n"
            "<agents_md>\n"
            f"{agents_text}\n"
            "</agents_md>\n\n"
            "You should update `/mnt/memory/AGENTS.md` and topic files in `/mnt/memory/cards/` with reusable lessons before submission."
        )
        return content, topic_count

    def _litellm_base_url(self) -> str:
        return self.model.config.model_kwargs["api_base"]

    def _litellm_headers(self) -> dict[str, str]:
        return {"Authorization": f'Bearer {self.model.config.model_kwargs["api_key"]}'}

    @staticmethod
    def _to_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _query_user_usage(self) -> tuple[float | None, float | None]:
        try:
            info = requests.get(
                f"{self._litellm_base_url()}/user/info",
                headers=self._litellm_headers(),
                timeout=2,
            ).json()
            user_info = info.get("user_info", {})
            if not isinstance(user_info, dict):
                return None, None
            return (
                self._to_float(user_info.get("spend")),
                self._to_float(user_info.get("max_budget")),
            )
        except Exception:
            return None, None

    def _query_key_usage(self) -> tuple[float | None, float | None]:
        try:
            info = requests.get(
                f"{self._litellm_base_url()}/key/info",
                headers=self._litellm_headers(),
                timeout=2,
            ).json()
            key_info = info.get("info", {})
            if not isinstance(key_info, dict):
                return None, None
            return (
                self._to_float(key_info.get("spend")),
                self._to_float(key_info.get("max_budget")),
            )
        except Exception:
            return None, None

    @staticmethod
    def _format_usage(spend: float | None, budget: float | None) -> str:
        if spend is None or budget is None:
            return "??? / ???"
        return f"{spend:.4f}$ / {budget:.4f}$"

    def _get_key_usage_context(self) -> tuple[str, str]:
        key_spend, key_budget = self._query_key_usage()
        usage_text = self._format_usage(key_spend, key_budget)
        usage_warning = self._build_usage_warning(key_spend, key_budget)
        return usage_text, usage_warning

    def _build_usage_warning(self, spend: float | None, budget: float | None) -> str:
        if spend is None or budget is None or budget <= 0:
            return ""
        if spend / budget < self.usage_warning_ratio:
            return ""
        return self.usage_warning_message

    def print_spend(self) -> None:
        # NOTE: spend info is updated asynchronously in litellm, and can be stale for 2~30 seconds
        # see https://github.com/BerriAI/litellm/blob/d07c87860db91c7bbf5a051d1cf82857432bf856/litellm/proxy/utils.py#L3874

        user_spend, user_budget = self._query_user_usage()
        key_spend, key_budget = self._query_key_usage()

        print(
            "spend so far: "
            f"key {self._format_usage(key_spend, key_budget)}, "
            f"user {self._format_usage(user_spend, user_budget)}"
        )

    def _build_submission_block_output(self, block_text: str) -> dict:
        return {
            "output": block_text,
            "returncode": 1,
            "exception_info": "",
            "extra": {
                "memory_stdout_preview": block_text,
                "memory_stderr_preview": "",
                "path_hint": str(self.memory_root),
            },
        }

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render_template(self.config.system_template),
            ),
            self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            ),
        )

        memory_context, topic_count = self._build_memory_context_message()
        self.add_messages(
            self.model.format_message(role="user", content=memory_context)
        )
        self._emit_memory_event(
            "memory_context_loaded",
            command="startup_memory_load",
            stdout=memory_context,
            path_hint=str(self.memory_root),
            details={"topic_count": topic_count},
        )

        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

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

        self.print_spend()
        print(f"query llm: step {self.n_calls}")

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)  # type: ignore
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []

        for action in actions:
            output = self.env.execute(action)
            outputs.append(output)

        # Attach key usage info to all outputs for better context in the next model query
        key_usage, usage_warning = self._get_key_usage_context()
        for output in outputs:
            if not isinstance(output, dict):
                continue
            extra = output.get("extra", {})
            if not isinstance(extra, dict):
                extra = {}
            extra["key_usage"] = key_usage
            extra["usage_warning"] = usage_warning
            output["extra"] = extra

        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )
