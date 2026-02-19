from pydantic import BaseModel
import os
import paramiko
import shlex
import subprocess
import traceback
from typing import Any
from minisweagent.exceptions import Submitted  # type: ignore
from memory_logging import (
    build_memory_output_meta,
    build_one_line_memory_hint,
    emit_memory_log,
)


class SshAndMemoryEnvironmentConfig(BaseModel):
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_password: str = ""
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 180
    memory_path: str = "/mnt/memory"


class SshAndMemoryEnvironment:
    def __init__(self, *, config_class: type = SshAndMemoryEnvironmentConfig, **kwargs):
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
            path_hint=path_hint or self.config.memory_path,
            output_len=output_len,
        )

    def _is_memory_command(self, command: str) -> bool:
        memory_path = self.config.memory_path.strip()
        if not memory_path:
            return False
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return False
        return memory_path in command

    def _execute_local_memory_command(self, command: str, *, timeout: int | None) -> dict:
        env = dict(os.environ)
        env.update(self.config.env)

        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            if stderr_text:
                output = f"[stdout]\n{stdout_text}\n\n[stderr]\n{stderr_text}"
            else:
                output = stdout_text
            output_meta = build_memory_output_meta(stdout_text, stderr_text)

            return {
                "output": output,
                "returncode": completed.returncode,
                "exception_info": "",
                "extra": {
                    "path_hint": self.config.memory_path,
                    **output_meta,
                },
            }
        except subprocess.TimeoutExpired as e:
            stdout_text = e.stdout if isinstance(e.stdout, str) else ""
            stderr_text = e.stderr if isinstance(e.stderr, str) else ""
            timeout_msg = f"Command timed out after {timeout}s"
            output_meta = build_memory_output_meta(stdout_text, stderr_text)
            return {
                "output": timeout_msg,
                "returncode": -1,
                "exception_info": timeout_msg,
                "extra": {
                    "path_hint": self.config.memory_path,
                    **output_meta,
                },
            }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        command = action.get("command", "")
        print("exec command:", command.replace("\n", " \\n ")[:80])

        timeout = timeout or self.config.timeout

        try:
            if self._is_memory_command(command):
                self._emit_memory_event(
                    "memory_command_dispatched",
                    command=command,
                    returncode=0,
                    path_hint=self.config.memory_path,
                )
                result = self._execute_local_memory_command(
                    command,
                    timeout=timeout,
                )
                result_extra = result.get("extra", {})
                self._emit_memory_event(
                    "memory_command_result",
                    command=command,
                    returncode=int(result.get("returncode", -1)),
                    stdout=str(result_extra.get("memory_stdout_preview", "")),
                    stderr=str(result_extra.get("memory_stderr_preview", "")),
                    path_hint=str(
                        result_extra.get("path_hint", self.config.memory_path)
                    ),
                    output_len=result_extra.get("output_len"),
                )
                return result

            cwd = cwd or self.config.cwd
            remote_command = command
            if cwd:
                remote_command = f"cd {shlex.quote(cwd)} ; {command}"

            client = self._ensure_client()
            stdin, stdout, stderr = client.exec_command(
                remote_command,
                timeout=timeout,
                environment=self.config.env,
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
            self._check_finished(result)
            return result
        except Submitted:
            raise

        except Exception as e:
            traceback.print_exc()
            self.close()
            return {
                "output": f"Failed to execute ({type(e).__name__}): {e}",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            }

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
