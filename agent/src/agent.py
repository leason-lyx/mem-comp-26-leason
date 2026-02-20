import requests
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded  # type: ignore


class MemoryAgent(DefaultAgent):
    def __init__(self, memory_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep field/signature for backward compatibility; persistence is disabled.
        self.memory_path = memory_path
        self.memorized_messages = []
        self.load_memory()

    def load_memory(self) -> None:
        return

    def save_memory(self) -> None:
        return

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
