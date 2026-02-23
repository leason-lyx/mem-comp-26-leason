from pathlib import Path

import requests
from minisweagent.agents.default import DefaultAgent  # type: ignore
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded  # type: ignore


class AgenticMemoryAgent(DefaultAgent):
    def __init__(self, memory_path: str, *args, **kwargs):
        usage_warning_ratio = self._to_float(kwargs.pop("usage_warning_ratio", 0.9))
        usage_warning_message = kwargs.pop("usage_warning_message")
        super().__init__(*args, **kwargs)
        self.memory_root = Path(memory_path)
        self.usage_warning_ratio = (
            usage_warning_ratio
            if usage_warning_ratio is not None and usage_warning_ratio > 0
            else 0.9
        )
        self.usage_warning_message = str(usage_warning_message)

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
