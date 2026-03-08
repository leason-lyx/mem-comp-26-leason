import json
import re
from typing import Any

from litellm import completion as litellm_completion

from memory_store import DEFAULT_MEMORY_FIELDS

ROLE_BUCKETS = {"system", "user", "assistant", "tool"}


class MemorySummarizer:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: int = 300,
        summary_version: str = "v1",
        fields: list[str] | None = None,
        field_descriptions: dict[str, str] | None = None,
        summary_total_char_limit: int = 136000,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.summary_version = summary_version
        self.summary_total_char_limit = max(120, int(summary_total_char_limit))
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
    def _content_to_str(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _role_bucket(role: str) -> str:
        if role in ROLE_BUCKETS:
            return role
        return "user"

    @staticmethod
    def _truncate_middle(
        text: str,
        *,
        max_len: int,
        head_len: int,
        tail_len: int,
    ) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        marker = "\n...[truncated]...\n"
        if max_len <= len(marker):
            return text[:max_len]

        keep_len = max_len - len(marker)
        requested_keep = max(head_len, 0) + max(tail_len, 0)
        if requested_keep <= 0:
            head_keep = keep_len // 2
        else:
            head_keep = int(keep_len * max(head_len, 0) / requested_keep)
        tail_keep = keep_len - head_keep

        if tail_keep <= 0:
            return text[:keep_len] + marker[: max_len - keep_len]

        return text[:head_keep] + marker + text[-tail_keep:]

    def _truncate_for_budget(self, text: str, max_len: int) -> str:
        return self._truncate_middle(
            text,
            max_len=max_len,
            head_len=(max_len * 3) // 5,
            tail_len=max_len - ((max_len * 3) // 5),
        )

    @staticmethod
    def _extract_action_commands(extra: Any) -> list[str]:
        if not isinstance(extra, dict):
            return []
        actions = extra.get("actions", [])
        if not isinstance(actions, list):
            return []

        commands: list[str] = []
        for action in actions:
            if isinstance(action, dict):
                commands.append(str(action.get("command", "")))
            else:
                commands.append(str(action))
        return commands

    @staticmethod
    def _allocate_proportional_caps(lengths: list[int], total_cap: int) -> list[int]:
        caps = [0] * len(lengths)
        if total_cap <= 0:
            return caps

        positive = [(idx, length) for idx, length in enumerate(lengths) if length > 0]
        if not positive:
            return caps

        total_len = sum(length for _, length in positive)
        assigned = 0
        remainders: list[tuple[float, int]] = []
        for idx, length in positive:
            exact = total_cap * length / total_len
            cap = int(exact)
            caps[idx] = cap
            assigned += cap
            remainders.append((exact - cap, idx))

        remaining = total_cap - assigned
        for _, idx in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            caps[idx] += 1
            remaining -= 1

        return caps

    @staticmethod
    def _message_text_length(info: dict[str, Any]) -> int:
        bucket = info["bucket"]
        if bucket == "assistant":
            return len(info["content"]) + sum(len(command) for command in info["actions"])
        if bucket == "tool":
            return len(info["content"]) + len(info["exception_info"])
        return len(info["content"])

    def _build_summary_prompts(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        exit_status: str,
    ) -> tuple[str, str]:
        compact_messages = self._compact_messages(messages)
        compact_json = json.dumps(compact_messages, ensure_ascii=False)

        field_guidance = self._build_field_guidance()
        system_prompt = (
            "You summarize a coding-agent trajectory into a reusable process memory for future guidance. "
            "Output MUST be strict JSON object with exactly these keys: "
            + ", ".join(self.fields)
            + ". Each value must be a concise string. No markdown, no code fences, no extra keys.\n\n"
            + "Field guidance:\n"
            + field_guidance
            + "\n\n"
            + "Use empty string when a field has no reliable evidence."
        )

        task_text = self._truncate_for_budget(task.strip(), 6000)
        user_prompt = (
            "Task:\n"
            f"{task_text}\n\n"
            f"Exit status: {exit_status}\n\n"
            "Trajectory (compacted JSON):\n"
            f"{compact_json}\n\n"
            "Return strict JSON only."
        )
        return system_prompt, user_prompt

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        class_totals: dict[str, int] = {}
        class_counts: dict[str, int] = {}

        for message in messages:
            role = str(message.get("role", ""))
            bucket = self._role_bucket(role)
            content = self._content_to_str(message.get("content", ""))
            extra = message.get("extra", {})
            actions = self._extract_action_commands(extra) if bucket == "assistant" else []
            exception_info = ""
            returncode = None
            if bucket == "tool" and isinstance(extra, dict):
                returncode = extra.get("returncode")
                if extra.get("exception_info"):
                    exception_info = self._content_to_str(extra.get("exception_info"))

            info = {
                "role": role,
                "bucket": bucket,
                "content": content,
                "actions": actions,
                "returncode": returncode,
                "exception_info": exception_info,
            }
            info["raw_len"] = self._message_text_length(info)
            infos.append(info)

            class_totals[bucket] = class_totals.get(bucket, 0) + info["raw_len"]
            class_counts[bucket] = class_counts.get(bucket, 0) + 1

        total_raw_len = sum(class_totals.values())
        class_caps: dict[str, int] = {}
        for bucket, count in class_counts.items():
            if total_raw_len > 0:
                class_budget = (
                    self.summary_total_char_limit
                    * class_totals.get(bucket, 0)
                    / total_raw_len
                )
            else:
                class_budget = self.summary_total_char_limit / max(len(class_counts), 1)
            class_caps[bucket] = max(120, int(class_budget / max(count, 1)))

        compact: list[dict[str, Any]] = []
        for info in infos:
            per_message_cap = class_caps.get(info["bucket"], 120)
            item: dict[str, Any] = {"role": info["role"], "content": ""}

            if info["bucket"] in {"system", "user"}:
                item["content"] = self._truncate_for_budget(
                    info["content"], per_message_cap
                )
            elif info["bucket"] == "assistant":
                lengths = [len(info["content"])] + [
                    len(command) for command in info["actions"]
                ]
                caps = self._allocate_proportional_caps(lengths, per_message_cap)
                content_cap = caps[0] if caps else per_message_cap
                item["content"] = self._truncate_for_budget(
                    info["content"], content_cap
                )
                if info["actions"]:
                    item["actions"] = [
                        self._truncate_for_budget(
                            command,
                            caps[idx + 1] if idx + 1 < len(caps) else 0,
                        )
                        for idx, command in enumerate(info["actions"])
                    ]
            elif info["bucket"] == "tool":
                lengths = [len(info["content"]), len(info["exception_info"])]
                caps = self._allocate_proportional_caps(lengths, per_message_cap)
                content_cap = caps[0] if caps else per_message_cap
                item["content"] = self._truncate_for_budget(
                    info["content"], content_cap
                )
                item["returncode"] = info["returncode"]
                if info["exception_info"]:
                    exception_cap = caps[1] if len(caps) > 1 else 0
                    item["exception_info"] = self._truncate_for_budget(
                        info["exception_info"], exception_cap
                    )
            else:
                item["content"] = self._truncate_for_budget(
                    info["content"], per_message_cap
                )

            compact.append(item)

        return compact

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

    def _normalize_record(self, raw: dict[str, Any]) -> dict[str, str]:
        record: dict[str, str] = {}
        for field in self.fields:
            value = raw.get(field, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            record[field] = str(value).strip()[:10000]
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
        task_short = self._truncate_for_budget(task.strip(), 2400)

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
            result[self.fields[0]] = (
                task_short or f"Fallback summary used. Reason: {reason}"
            )
        return result

    def summarize(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        exit_status: str,
    ) -> dict[str, str]:
        if not self.api_base or not self.api_key or not self.model_name:
            return self._fallback_summary(
                task=task,
                messages=messages,
                exit_status=exit_status,
                reason="Missing LLM API configuration",
            )

        system_prompt, user_prompt = self._build_summary_prompts(
            task=task,
            messages=messages,
            exit_status=exit_status,
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
