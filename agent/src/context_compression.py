import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from litellm import completion as litellm_completion
from pydantic import BaseModel, Field

CONTEXT_SUMMARY_MARKER = "[CONTEXT_SUMMARY]"
AGENTDIET_MARKER = "[AGENTDIET]"
OBS_MASKED_MARKER = "[OBS_MASKED]"


class SummaryCompressionConfig(BaseModel):
    context_char_threshold: int = 48000
    keep_recent_steps: int = 6
    summary_prompt: str = """You are maintaining a context-aware state summary for an interactive agent working on software engineering tasks (specifically, bug fixes within the SWE-bench framework). You will be given a list of turns corresponding to actions taken by the agent and their resulting observations, and the most recent previous summary if one exists. Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

COMPLETED: (Sub-tasks completed so far, with brief results)
PENDING: (Key sub-tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant environmental state not covered elsewhere)
CODE_STATE: (File paths, function signatures, data structures followed by their current state)
TESTS: (Failing cases, error messages, outputs)
CHANGES: (Code edits, variable updates)
DEPS: (Dependencies, imports, external calls)

PRIORITIZE:

1. Capture key user requirements and goals (from the initial issue)
2. Distinguish between completed and pending sub-tasks clearly
3. Keep all sections concise and relevant to fixing the issue
4. Focus on information that quantifies the agent's progress towards a solution

Example format:

USER_CONTEXT: Fix a ZeroDivisionError in utils.calculate_ratio when denominator is zero. The function should return 0 in this case.
COMPLETED:
1. Reproduced ZeroDivisionError with calculate_ratio(5, 0).
2. Identified missing check for denominator == 0 in utils.py.
3. Added test case test_calculate_ratio_zero_denominator to tests/test_utils.py.
4. Modified utils.calculate_ratio to return 0 if denominator is 0.
5. Verified test_calculate_ratio_zero_denominator now passes.
6. All tests in tests/test_utils.py are passing. 
PENDING: 
1. Prepare final patch file for submission. 
CODE_STATE: 
1. app/utils.py: calculate_ratio function MODIFIED.
2. tests/test_utils.py: test_calculate_ratio_zero_denominator ADDED. 
TESTS:
1. tests/test_utils.py::test_calculate_ratio_zero_denominator: PASSED (previously FAILED).
2. All tests in tests/test_utils.py: PASSED. 
CHANGES:
1. app/utils.py: calculate_ratio(): Added conditional if denominator == 0: return 0.
2. tests/test_utils.py: Added test_calculate_ratio_zero_denominator to assert behavior with zero denominator. 
DEPS: None modified.
"""


class AgentDietCompressionConfig(BaseModel):
    start_step: int = 10
    a: int = 8
    b: int = 6
    target_min_chars: int = 2000
    min_gain_ratio: float = 0.35
    max_window_chars: int = 20000


class ObsMaskingCompressionConfig(BaseModel):
    min_age_steps: int = 6
    observation_char_threshold: int = 3000
    max_masks_per_query: int = 4
    placeholder_template: str = "[OBS_MASKED] omitted."


CompressionMode = Literal["off", "summary", "agentdiet", "obs_masking"]


class CompressionConfig(BaseModel):
    mode: CompressionMode = "off"
    compressor_model_name: str | None = None
    timeout_s: int = 45
    summary: SummaryCompressionConfig = Field(default_factory=SummaryCompressionConfig)
    agentdiet: AgentDietCompressionConfig = Field(
        default_factory=AgentDietCompressionConfig
    )
    obs_masking: ObsMaskingCompressionConfig = Field(
        default_factory=ObsMaskingCompressionConfig
    )


@dataclass
class CompressionEvent:
    mode: str
    executed: bool
    before_chars: int
    after_chars: int
    reason: str = ""


@dataclass
class StepSlice:
    step_index: int
    start_idx: int
    end_idx: int
    observation_indices: list[int]


def safe_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _truncate_middle(text: str, max_len: int, head_len: int, tail_len: int) -> str:
    if len(text) <= max_len:
        return text
    clipped = max(0, len(text) - head_len - tail_len)
    return text[:head_len] + f"\n...[{clipped} chars truncated]...\n" + text[-tail_len:]


def estimate_message_chars(message: dict[str, Any]) -> int:
    content_len = len(safe_content_to_str(message.get("content", "")))
    tool_calls_len = len(json.dumps(message.get("tool_calls", []), ensure_ascii=False))
    return content_len + tool_calls_len


def estimate_total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_chars(message) for message in messages)


def _is_action_assistant(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    extra = message.get("extra", {})
    if not isinstance(extra, dict):
        return False
    actions = extra.get("actions", [])
    return isinstance(actions, list) and len(actions) > 0


def extract_step_slices(messages: list[dict[str, Any]]) -> list[StepSlice]:
    action_positions = [
        i for i, message in enumerate(messages) if _is_action_assistant(message)
    ]
    slices: list[StepSlice] = []
    for idx, start_pos in enumerate(action_positions):
        end_pos = (
            action_positions[idx + 1]
            if idx + 1 < len(action_positions)
            else len(messages)
        )
        observation_indices = [
            j
            for j in range(start_pos + 1, end_pos)
            if messages[j].get("role") == "tool"
        ]
        slices.append(
            StepSlice(
                step_index=idx,
                start_idx=start_pos,
                end_idx=end_pos,
                observation_indices=observation_indices,
            )
        )
    return slices


def already_compressed_marker(message: dict[str, Any]) -> bool:
    content = safe_content_to_str(message.get("content", ""))
    if (
        CONTEXT_SUMMARY_MARKER in content
        or AGENTDIET_MARKER in content
        or OBS_MASKED_MARKER in content
    ):
        return True
    extra = message.get("extra", {})
    return isinstance(extra, dict) and "compression_meta" in extra


class ContextCompressor:
    def __init__(
        self,
        *,
        config: CompressionConfig,
        api_base: str,
        api_key: str,
        default_model_name: str,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_model_name = default_model_name
        self.log_fn = log_fn or print

    def maybe_compress(
        self,
        *,
        messages: list[dict[str, Any]],
        current_step: int,
        task: str,
    ) -> CompressionEvent:
        mode = self.config.mode
        before_chars = estimate_total_chars(messages)
        if mode == "off":
            return CompressionEvent(
                mode=mode,
                executed=False,
                before_chars=before_chars,
                after_chars=before_chars,
                reason="mode_off",
            )

        try:
            if mode == "summary":
                executed, reason = self._apply_summary(
                    messages=messages, current_step=current_step, task=task
                )
            elif mode == "agentdiet":
                executed, reason = self._apply_agentdiet(
                    messages=messages, current_step=current_step, task=task
                )
            elif mode == "obs_masking":
                executed, reason = self._apply_obs_masking(
                    messages=messages, current_step=current_step
                )
            else:
                executed, reason = False, "unknown_mode"
        except Exception as exc:
            self.log_fn(f"[compression] mode={mode} failed={type(exc).__name__}: {exc}")
            executed, reason = False, f"error:{type(exc).__name__}"

        after_chars = estimate_total_chars(messages)
        return CompressionEvent(
            mode=mode,
            executed=executed,
            before_chars=before_chars,
            after_chars=after_chars,
            reason=reason,
        )

    def call_compression_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        model_name = self.config.compressor_model_name or self.default_model_name
        if not model_name or not self.api_base or not self.api_key:
            return ""

        response = litellm_completion(
            model=model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.config.timeout_s,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            drop_params=True,
        )
        content = self._extract_response_content(response)
        # content = self._strip_code_fence(content).strip()
        return content.strip()

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

    # @staticmethod
    # def _strip_code_fence(text: str) -> str:
    #     match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, re.DOTALL)
    #     if match:
    #         return match.group(1)
    #     return text

    @staticmethod
    def _compact_messages_for_prompt(
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
        content_limit: int = 1400,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        total_chars = 0
        for message in messages:
            role = str(message.get("role", ""))
            content = _truncate_middle(
                safe_content_to_str(message.get("content", "")),
                max_len=content_limit,
                head_len=min(900, content_limit),
                tail_len=min(400, content_limit),
            )
            item: dict[str, Any] = {"role": role, "content": content}

            extra = message.get("extra", {})
            if isinstance(extra, dict):
                if role == "assistant":
                    actions = extra.get("actions", [])
                    if isinstance(actions, list):
                        commands = []
                        for action in actions:
                            if not isinstance(action, dict):
                                continue
                            command = str(action.get("command", "")).strip()
                            if command:
                                commands.append(command[:300])
                        if commands:
                            item["actions"] = commands[:6]
                if role == "tool":
                    item["returncode"] = extra.get("returncode")

            encoded = json.dumps(item, ensure_ascii=False)
            if max_chars is not None and total_chars + len(encoded) > max_chars:
                break
            compact.append(item)
            total_chars += len(encoded)
        return compact

    def _apply_summary(
        self,
        *,
        messages: list[dict[str, Any]],
        current_step: int,
        task: str,
    ) -> tuple[bool, str]:
        cfg = self.config.summary
        if len(messages) <= 2:
            return False, "insufficient_messages"

        steps = extract_step_slices(messages)
        compress_start = 2
        if cfg.keep_recent_steps > 0 and steps:
            keep_index = max(0, len(steps) - cfg.keep_recent_steps)
            compress_end = steps[keep_index].start_idx
        else:
            compress_end = len(messages)
        compress_end = max(compress_start, min(compress_end, len(messages)))

        if compress_end <= compress_start:
            return False, "no_old_history"

        target_messages = messages[compress_start:compress_end]
        compress_chars = sum(
            estimate_message_chars(message) for message in target_messages
        )
        if compress_chars < cfg.context_char_threshold:
            return False, "below_context_threshold"
        if len(target_messages) == 1 and already_compressed_marker(target_messages[0]):
            return False, "already_summary_marker"

        compact = self._compact_messages_for_prompt(target_messages)
        if not compact:
            return False, "empty_compact_messages"

        system_prompt = cfg.summary_prompt
        user_prompt = (
            f"Task:\n{task[:4000]}\n\n"
            "Old history to compress (JSON):\n"
            f"{json.dumps(compact, ensure_ascii=False)}\n\n"
            "Write one concise summary for future steps."
        )
        summary_text = self.call_compression_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not summary_text:
            return False, "llm_empty_or_unavailable"

        summary_message: dict[str, Any] = {
            "role": "user",
            "content": (
                f"{CONTEXT_SUMMARY_MARKER}\n"
                "Compressed earlier context (system and initial task input unchanged):\n"
                f"{summary_text}"
            ),
            "extra": {
                "compression_meta": {
                    "type": "summary",
                    "compressed_message_count": len(target_messages),
                    "original_chars": compress_chars,
                    "step": current_step,
                    "timestamp": time.time(),
                }
            },
        }
        messages[:] = (
            messages[:compress_start] + [summary_message] + messages[compress_end:]
        )
        print(
            f"[compression] summary applied at step={current_step} "
            f"compressed_messages={len(target_messages)} "
            f"original_chars={compress_chars} "
            f"summary_chars={len(summary_text)}"
        )
        return True, f"summary_compacted_{len(target_messages)}"

    def _apply_agentdiet(
        self,
        *,
        messages: list[dict[str, Any]],
        current_step: int,
        task: str,
    ) -> tuple[bool, str]:
        cfg = self.config.agentdiet
        if current_step < cfg.start_step:
            return False, "before_start_step"

        steps = extract_step_slices(messages)
        if not steps:
            return False, "no_steps_found"

        target_step = current_step - cfg.a
        if target_step < 0:
            return False, "negative_target_step"
        if target_step >= len(steps):
            return False, "target_step_unavailable"

        target_slice = steps[target_step]
        target_obs_indices = target_slice.observation_indices
        if not target_obs_indices:
            return False, "target_step_has_no_observation"

        if any(already_compressed_marker(messages[idx]) for idx in target_obs_indices):
            return False, "target_already_compressed"

        target_chars = sum(
            estimate_message_chars(messages[idx]) for idx in target_obs_indices
        )
        if target_chars < cfg.target_min_chars:
            return False, "target_too_short"

        latest_window_step = min(len(steps) - 1, max(0, current_step - 1))
        window_start_step = max(0, target_step - cfg.b)
        window_slices = steps[window_start_step : latest_window_step + 1]
        if not window_slices:
            return False, "empty_window"

        window_messages: list[dict[str, Any]] = []
        for step_slice in window_slices:
            window_messages.extend(messages[step_slice.start_idx : step_slice.end_idx])

        compact_window = self._compact_messages_for_prompt(
            window_messages, max_chars=cfg.max_window_chars
        )
        if not compact_window:
            return False, "empty_window_after_compact"

        target_observations = [messages[idx] for idx in target_obs_indices]
        compact_target = self._compact_messages_for_prompt(
            target_observations, max_chars=min(cfg.max_window_chars, 8000)
        )

        system_prompt = (
            "You compress one historical tool-observation step in an agent trajectory. "
            "Keep only details that are useful for future debugging and decision making. "
            "Return plain text only."
        )
        user_prompt = (
            f"Task:\n{task[:3000]}\n\n"
            f"Current step s={current_step}, target step t={target_step}.\n"
            "Local window JSON:\n"
            f"{json.dumps(compact_window, ensure_ascii=False)}\n\n"
            "Target step observations JSON:\n"
            f"{json.dumps(compact_target, ensure_ascii=False)}\n\n"
            "Return a concise replacement summary for target observations."
        )
        compressed_text = self.call_compression_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not compressed_text:
            return False, "llm_empty_or_unavailable"

        gain_ratio = (target_chars - len(compressed_text)) / max(1, target_chars)
        if gain_ratio < cfg.min_gain_ratio:
            return False, "gain_ratio_too_low"

        first_obs_idx = target_obs_indices[0]
        first_obs = dict(messages[first_obs_idx])
        first_extra = (
            dict(first_obs.get("extra", {}))
            if isinstance(first_obs.get("extra"), dict)
            else {}
        )
        first_extra["compression_meta"] = {
            "type": "agentdiet",
            "step": target_step,
            "original_chars": target_chars,
            "gain_ratio": gain_ratio,
            "timestamp": time.time(),
        }
        first_obs["content"] = (
            f"{AGENTDIET_MARKER} step={target_step}\n{compressed_text.strip()}"
        )
        first_obs["extra"] = first_extra
        messages[first_obs_idx] = first_obs

        for idx in target_obs_indices[1:]:
            msg = dict(messages[idx])
            extra = (
                dict(msg.get("extra", {})) if isinstance(msg.get("extra"), dict) else {}
            )
            returncode = extra.get("returncode", "unknown")
            msg["content"] = (
                f"{AGENTDIET_MARKER} step={target_step} "
                "additional tool output compressed into first observation; "
                f"returncode={returncode}"
            )
            extra["compression_meta"] = {
                "type": "agentdiet_placeholder",
                "step": target_step,
                "collapsed_to": first_obs_idx,
                "timestamp": time.time(),
            }
            msg["extra"] = extra
            messages[idx] = msg

        return True, f"agentdiet_step_{target_step}"

    def _apply_obs_masking(
        self,
        *,
        messages: list[dict[str, Any]],
        current_step: int,
    ) -> tuple[bool, str]:
        cfg = self.config.obs_masking
        steps = extract_step_slices(messages)
        if not steps:
            return False, "no_steps_found"

        obs_to_step: dict[int, int] = {}
        for step_slice in steps:
            for obs_idx in step_slice.observation_indices:
                obs_to_step[obs_idx] = step_slice.step_index

        masked_count = 0
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            step_idx = obs_to_step.get(idx)
            if step_idx is None:
                continue
            age_steps = current_step - step_idx
            if age_steps < cfg.min_age_steps:
                continue
            if already_compressed_marker(message):
                continue

            content = safe_content_to_str(message.get("content", ""))
            original_chars = len(content)
            if original_chars <= cfg.observation_char_threshold:
                continue

            msg = dict(message)
            extra = (
                dict(msg.get("extra", {})) if isinstance(msg.get("extra"), dict) else {}
            )
            returncode = extra.get("returncode", "unknown")
            placeholder = cfg.placeholder_template
            msg["content"] = placeholder
            extra["compression_meta"] = {
                "type": "obs_masking",
                "step": step_idx,
                "original_chars": original_chars,
                "age_steps": age_steps,
                "timestamp": time.time(),
            }
            print(
                f"[compression] obs_masking step={step_idx} age={age_steps} chars={original_chars} returncode={returncode}"
            )
            msg["extra"] = extra
            messages[idx] = msg

            masked_count += 1
            if masked_count >= cfg.max_masks_per_query:
                break

        if masked_count == 0:
            return False, "no_observation_masked"
        return True, f"masked_{masked_count}"
