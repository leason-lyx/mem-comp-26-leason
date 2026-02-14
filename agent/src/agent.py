import os
import json
import requests
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded  # type: ignore


class MemoryAgent(DefaultAgent):
    def __init__(self, memory_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_path = f"{memory_path}/memory.json"
        self.memorized_messages = []
        self.load_memory()

    @staticmethod
    def _sanitize_memory_message(message: dict) -> dict | None:
        if not isinstance(message, dict):
            return None

        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            return None

        return {k: v for k, v in message.items() if k != "timestamp"}

    def load_memory(self) -> None:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                loaded_messages = json.load(f)

            sanitized_messages = []
            if isinstance(loaded_messages, list):
                for message in loaded_messages:
                    sanitized = self._sanitize_memory_message(message)
                    if sanitized is not None:
                        sanitized_messages.append(sanitized)

            self.memorized_messages = [
                *sanitized_messages,
                {
                    "role": "user",
                    "content": "**A new task started. The project root path has been reset to a clean state for this task.**",
                },
            ]
            print(f"loaded {len(self.memorized_messages)} memorized messages")

    def save_memory(self) -> None:
        memory_messages = []
        for message in self.messages:
            if message.get("role") in {"system", "exit"}:
                continue
            sanitized = self._sanitize_memory_message(message)
            if sanitized is not None:
                memory_messages.append(sanitized)

        print(f"saved {len(memory_messages)} messages to memory")
        with open(self.memory_path, "w") as f:
            json.dump(memory_messages, f, indent=2)

    def print_spend(self) -> None:
        # NOTE: spend info is updated asynchronously in litellm, and can be stale for 2~30 seconds
        # see https://github.com/BerriAI/litellm/blob/d07c87860db91c7bbf5a051d1cf82857432bf856/litellm/proxy/utils.py#L3874

        u = self.model.config.model_kwargs["api_base"]
        h = {"Authorization": f'Bearer {self.model.config.model_kwargs["api_key"]}'}

        try:
            info = requests.get(f"{u}/user/info", headers=h, timeout=2).json()
            user_spend = f'{info["user_info"]["spend"]:.4f} / {info["user_info"]["max_budget"]:.4f}'
        except Exception:
            user_spend = "???"

        try:
            info = requests.get(f"{u}/key/info", headers=h, timeout=2).json()
            key_spend = (
                f'{info["info"]["spend"]:.4f} / {info["info"]["max_budget"]:.4f}'
            )
        except Exception:
            key_spend = "???"

        print(f"spend so far: key {key_spend}, user {user_spend}")

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

        # insert memorized messages after the first message (system prompt)
        messages = [*self.messages[:1], *self.memorized_messages, *self.messages[1:]]

        self.n_calls += 1
        message = self.model.query(messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)  # type: ignore
        return message
