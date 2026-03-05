import json
import re
from typing import Any

from litellm import completion as litellm_completion

from memory_store import DEFAULT_MEMORY_FIELDS


class MemorySummarizer:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: int = 45,
        summary_version: str = "v1",
        fields: list[str] | None = None,
        field_descriptions: dict[str, str] | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.summary_version = summary_version
        selected_fields = fields or list(DEFAULT_MEMORY_FIELDS)
        self.fields = tuple(str(field) for field in selected_fields)
        self.field_descriptions = {
            str(key): str(value)
            for key, value in (field_descriptions or {}).items()
            if value is not None
        }

    def _build_field_guidance(self) -> str:
        lines: list[str] = []
        for field in self.fields:
            description = self.field_descriptions.get(field)
            if description:
                lines.append(f"- {field}: {description}")
            else:
                lines.append(
                    f"- {field}: Keep this concise, concrete, and supported by the trajectory."
                )
        return "\n".join(lines)

    @staticmethod
    def _truncate_middle(
        text: str,
        *,
        max_len: int,
        head_len: int,
        tail_len: int,
    ) -> str:
        if len(text) <= max_len:
            return text
        return (
            text[:head_len]
            + f"\n...[{len(text) - head_len - tail_len} chars truncated]...\n"
            + text[-tail_len:]
        )

    def _extract_commands(
        self, messages: list[dict[str, Any]], limit: int = 24
    ) -> list[str]:
        commands: list[str] = []
        for message in messages:
            extra = message.get("extra", {})
            if not isinstance(extra, dict):
                continue
            actions = extra.get("actions", [])
            if not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, dict):
                    continue
                command = str(action.get("command", "")).strip()
                if command:
                    commands.append(command)
        return commands[-limit:]

    def _extract_paths(self, text: str, limit: int = 20) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for match in re.findall(r"/[A-Za-z0-9._/\-]+", text):
            if match in seen:
                continue
            seen.add(match)
            paths.append(match)
            if len(paths) >= limit:
                break
        return paths

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "tool":
                content = self._truncate_middle(
                    content, max_len=2400, head_len=1200, tail_len=1200
                )
            else:
                content = self._truncate_middle(
                    content, max_len=1200, head_len=800, tail_len=350
                )

            item: dict[str, Any] = {"role": role, "content": content}
            extra = message.get("extra", {})
            if isinstance(extra, dict):
                if role == "assistant":
                    actions = extra.get("actions", [])
                    if isinstance(actions, list) and actions:
                        item["actions"] = [
                            str(a.get("command", ""))[:400]
                            for a in actions
                            if isinstance(a, dict) and a.get("command")
                        ][:6]
                if role == "tool":
                    item["returncode"] = extra.get("returncode")
                    if extra.get("exception_info"):
                        item["exception_info"] = str(extra.get("exception_info"))[:240]
            compact.append(item)

        return compact

    def _normalize_record(self, raw: dict[str, Any]) -> dict[str, str]:
        record: dict[str, str] = {}
        for field in self.fields:
            value = raw.get(field, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            record[field] = str(value).strip()[:6000]
        return record

    def _extract_json_from_content(self, content: str) -> dict[str, Any]:
        block = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if block:
            return json.loads(block.group(1))

        generic = re.search(r"(\{.*\})", content, flags=re.DOTALL)
        if generic:
            return json.loads(generic.group(1))
        return json.loads(content)

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")

        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")

        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _fallback_summary(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        exit_status: str,
        reason: str,
    ) -> dict[str, str]:
        commands = self._extract_commands(messages)
        commands_text = "\n".join(commands[:24])[:6000]
        path_text = "\n".join(self._extract_paths(commands_text + "\n" + task))
        task_short = self._truncate_middle(
            task.strip(), max_len=1200, head_len=800, tail_len=350
        )

        fallback_values = {
            "issue_description": task_short or "No task description captured.",
            "keywords": ", ".join(re.findall(r"[A-Za-z0-9_]+", task.lower())[:20]),
            "root_cause": "Summary model output unavailable or unparsable.",
            "decisions": "Investigated the task by iterating with shell commands and code inspection.",
            "insights": (
                f"Task ended with exit status '{exit_status}'. Keep successful steps and avoid repeated dead ends."
            ),
            "related_paths": path_text or "No clear file path extracted.",
            "commands": commands_text or "No commands captured.",
            "notes": f"Fallback summary used. Reason: {reason}",
        }
        result = {field: fallback_values.get(field, "") for field in self.fields}
        if not any(result.values()):
            result[self.fields[0]] = task_short or f"Fallback summary used. Reason: {reason}"
        return result

    def summarize(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        exit_status: str,
    ) -> dict[str, str]:
        compact_messages = self._compact_messages(messages)
        compact_json = json.dumps(compact_messages, ensure_ascii=False)

        if len(compact_json) > 100000:
            compact_json = compact_json[-100000:]

        if not self.api_base or not self.api_key or not self.model_name:
            return self._fallback_summary(
                task=task,
                messages=messages,
                exit_status=exit_status,
                reason="Missing LLM API configuration",
            )

        field_guidance = self._build_field_guidance()
        system_prompt = (
            "You summarize a coding-agent trajectory into a reusable process memory. "
            "Output MUST be strict JSON object with exactly these keys: "
            + ", ".join(self.fields)
            + ". Each value must be a concise string. No markdown, no code fences, no extra keys.\n\n"
            + "Field guidance:\n"
            + field_guidance
            + "\n\n"
            + "Only write what can be supported by trajectory evidence. "
            + "Use empty string when a field has no reliable evidence."
        )

        user_prompt = (
            "Task:\n"
            f"{task[:5000]}\n\n"
            f"Exit status: {exit_status}\n\n"
            "Trajectory (compacted JSON):\n"
            f"{compact_json}\n\n"
            "Return strict JSON only."
        )

        response = litellm_completion(
            model=self.model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            drop_params=True,
        )
        content = self._extract_response_content(response)

        parsed = self._extract_json_from_content(content)
        normalized = self._normalize_record(parsed)
        if any(normalized.get(k, "") for k in self.fields):
            return normalized
        raise ValueError("Empty summary")
