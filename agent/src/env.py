from pydantic import BaseModel
import paramiko
import requests
import shlex
import traceback
from typing import Any
from minisweagent.exceptions import Submitted  # type: ignore


class SshEnvironmentConfig(BaseModel):
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_password: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 180


class SshEnvironment:
    COST_WARNING_RATIO = 0.9

    def __init__(self, *, config_class: type = SshEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self._client: paramiko.SSHClient | None = None

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._client is None:
            return

        try:
            self._client.close()
        except Exception:
            return
        self._client = None

    def _ensure_client(self) -> paramiko.SSHClient:
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client

            self.close()

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.config.ssh_host,
            port=self.config.ssh_port,
            username=self.config.ssh_user,
            password=self.config.ssh_password or None,
        )
        self._client = client
        return client

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def _inject_observation_cost_context(self, result: dict[str, Any]) -> dict[str, Any]:
        u = self.config.llm_base_url
        api_key = self.config.llm_api_key
        spend_value: float | None = None
        budget_value: float | None = None
        ratio: float | None = None
        warning = False

        if u and api_key:
            h = {"Authorization": f"Bearer {api_key}"}
            try:
                info = requests.get(f"{u}/key/info", headers=h, timeout=2).json()
                key_info = info.get("info", {})
                if isinstance(key_info, dict):
                    spend_value = self._to_optional_float(key_info.get("spend"))
                    budget_value = self._to_optional_float(key_info.get("max_budget"))
            except Exception:
                pass

        if spend_value is not None and budget_value is not None and budget_value > 0:
            ratio = spend_value / budget_value
            warning = ratio > self.COST_WARNING_RATIO
            text = (
                f"Current task cost: {spend_value:.4f} / {budget_value:.4f} "
                f"({ratio:.2%} used)."
            )
        elif spend_value is not None:
            text = f"Current task cost: {spend_value:.4f} / unknown (budget unavailable)."
        else:
            text = "Current task cost: unknown / unknown (budget unavailable)."

        result["task_cost_used"] = spend_value
        result["task_cost_budget"] = budget_value
        result["task_cost_ratio"] = ratio
        result["task_cost_warning"] = warning
        result["task_cost_text"] = text
        return result

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        command = action.get("command", "")
        print("exec command:", command.replace("\n", " \\n ")[:80])

        cwd = cwd or self.config.cwd
        timeout = timeout or self.config.timeout
        if cwd:
            command = f"cd {shlex.quote(cwd)} ; {command}"

        try:
            client = self._ensure_client()
            stdin, stdout, stderr = client.exec_command(
                command, timeout=timeout, environment=self.config.env
            )
            stdin.close()

            # Preserve command output exactly (including leading tabs/spaces/newlines),
            # which is important for git diff content.
            output = stdout.read().decode("utf-8", errors="replace")
            output_stderr = stderr.read().decode("utf-8", errors="replace")
            if output_stderr:
                output = f"[stdout]\n{output}\n\n[stderr]\n{output_stderr}"

            result = {
                "output": output,
                "returncode": stdout.channel.recv_exit_status(),
                "exception_info": "",
            }
            self._inject_observation_cost_context(result)
            self._check_finished(result)
            return result
        except Submitted:
            raise

        except Exception as e:
            traceback.print_exc()
            self.close()
            result = {
                "output": f"Failed to execute ({type(e).__name__}): {e}",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            }
            return self._inject_observation_cost_context(result)

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            if output.get("returncode", 0) != 0:
                return
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
