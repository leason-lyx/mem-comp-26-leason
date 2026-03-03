import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from litellm import completion as litellm_completion
from pydantic import BaseModel, Field

CONTEXT_SUMMARY_MARKER = "[CONTEXT_SUMMARY]"
AGENTDIET_MARKER = "[Compressed with AgentDiet]"
OBS_MASKED_MARKER = "[Result Ommited]"
AGENTDIET_SYSTEM_PROMPT = """You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the given id to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

You should only process the text within the <step> tag with the given id. STOP OUTPUT IMMEDIATELY AFTER </step>."""


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
    n_ctx_before: int = 2
    n_ctx_after: int = 1
    show_ctx: bool = True
    bypass_filter: bool = False
    min_char_reduction: int = 400
    max_char_ratio: float = 0.80


class ObsMaskingCompressionConfig(BaseModel):
    min_age_steps: int = 6
    observation_char_threshold: int = 3000
    max_masks_per_query: int = 4
    placeholder_template: str = "[OBS_MASKED] omitted."


CompressionMode = Literal["off", "summary", "agentdiet", "obsmasking"]


class CompressionConfig(BaseModel):
    mode: CompressionMode = "off"
    compressor_model_name: str | None = None
    timeout_s: int = 45
    summary: SummaryCompressionConfig = Field(default_factory=SummaryCompressionConfig)
    agentdiet: AgentDietCompressionConfig = Field(
        default_factory=AgentDietCompressionConfig
    )
    obsmasking: ObsMaskingCompressionConfig = Field(
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
            elif mode == "obsmasking":
                executed, reason = self._apply_obsmasking(
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
        messages: list[dict[str, Any]],
    ) -> tuple[str, str]:
        response = self._call_compression_completion(messages=messages)
        content = self._extract_response_content(response).strip()
        finish_reason = self._extract_finish_reason(response)
        return content, finish_reason

    def _call_compression_completion(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> Any:
        model_name = self.config.compressor_model_name or self.default_model_name
        if not model_name or not self.api_base or not self.api_key:
            return None

        return litellm_completion(
            model=model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.config.timeout_s,
            messages=messages,
            drop_params=True,
        )

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        if response is None:
            return ""
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

    @staticmethod
    def _extract_finish_reason(response: Any) -> str:
        if response is None:
            return ""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")
        else:
            finish_reason = getattr(first, "finish_reason", "")
        return finish_reason or ""

    @staticmethod
    def _message_extra(message: dict[str, Any]) -> dict[str, Any]:
        extra = message.get("extra", {})
        if isinstance(extra, dict):
            return extra
        return {}

    def _assistant_has_agentdiet_compression(self, message: dict[str, Any]) -> bool:
        extra = self._message_extra(message)
        compression_meta = extra.get("compression_meta")
        if (
            isinstance(compression_meta, dict)
            and compression_meta.get("type") == "agentdiet"
            and isinstance(compression_meta.get("compressed_step_xml"), str)
        ):
            return True
        content = safe_content_to_str(message.get("content", ""))
        return AGENTDIET_MARKER in content

    def _render_step_xml(
        self,
        *,
        messages: list[dict[str, Any]],
        step_slice: StepSlice,
    ) -> str:
        assistant_message = messages[step_slice.start_idx]
        compression_meta = self._message_extra(assistant_message).get(
            "compression_meta"
        )
        if (
            isinstance(compression_meta, dict)
            and compression_meta.get("type") == "agentdiet"
            and isinstance(compression_meta.get("compressed_step_xml"), str)
        ):
            return compression_meta["compressed_step_xml"]

        parts = [f'<step id="{step_slice.step_index}">']
        think_text = safe_content_to_str(assistant_message.get("content", ""))
        parts.append(f"<think>{think_text}</think>")

        actions = self._message_extra(assistant_message).get("actions", [])
        action_texts: list[str] = []
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and "command" in action:
                    action_texts.append(safe_content_to_str(action.get("command")))
                else:
                    action_texts.append(safe_content_to_str(action))

        result_texts = [
            safe_content_to_str(messages[idx].get("content", ""))
            for idx in step_slice.observation_indices
        ]

        item_count = max(len(action_texts), len(result_texts))
        for idx in range(item_count):
            if idx < len(action_texts):
                parts.append(f'<call tool="bash">{action_texts[idx]}</call>')
            if idx < len(result_texts):
                parts.append(f"<result>{result_texts[idx]}</result>")

        parts.append("</step>")
        return "\n".join(parts)

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

        system_prompt = cfg.summary_prompt
        user_prompt = (
            f"Task:\n{task[:4000]}\n\n"
            "Old history to compress (JSON):\n"
            f"{json.dumps(target_messages, ensure_ascii=False, default=str)}\n\n"
            "Write one concise summary for future steps."
        )
        summary_text, _ = self.call_compression_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        steps = extract_step_slices(messages)
        if not steps:
            return False, "no_steps_found"

        target_step = len(steps) - 1 - cfg.n_ctx_after
        if target_step < 0:
            return False, "target_step_unavailable"
        if target_step - cfg.n_ctx_before < 0:
            return False, "insufficient_context_before"
        if target_step + cfg.n_ctx_after >= len(steps):
            return False, "insufficient_context_after"

        target_slice = steps[target_step]
        assistant_message = messages[target_slice.start_idx]
        if self._assistant_has_agentdiet_compression(assistant_message):
            return False, "target_already_compressed"

        traj_content = [
            self._render_step_xml(messages=messages, step_slice=steps[idx])
            for idx in range(
                target_step - cfg.n_ctx_before,
                target_step + cfg.n_ctx_after + 1,
            )
        ]
        if not cfg.show_ctx:
            for idx in range(len(traj_content)):
                if idx != cfg.n_ctx_before:
                    traj_content[idx] = ""

        system_prompt = AGENTDIET_SYSTEM_PROMPT
        if cfg.bypass_filter:
            system_prompt = system_prompt.replace("think", "talk")
            system_prompt = system_prompt.replace("agent", "engineer")

        original_step_xml = traj_content[cfg.n_ctx_before]
        user_prompt = (
            "\n".join(traj_content) + f"\n\nNow, compress the step {target_step}."
        )
        compressed_text, finish_reason = self.call_compression_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": (
                        "Sure. Here is the compressed content of step "
                        f'{target_step}: <step id="{target_step}">'
                    ),
                },
            ]
        )
        if not compressed_text:
            return False, "llm_empty_or_unavailable"

        content, splitter, _ = compressed_text.partition("</step>")
        if not splitter and finish_reason != "stop":
            return False, "invalid_completion"

        if "<step" in content[:200]:
            content = content.partition("<step")[2]
            if ">" in content[:20]:
                content = content.partition(">")[2]

        compressed_inner = content
        old_chars = len(original_step_xml)
        new_chars = len(compressed_inner)
        if not (
            old_chars - new_chars >= cfg.min_char_reduction
            or new_chars < cfg.max_char_ratio * old_chars
        ):
            return False, "no_erase"

        compressed_step_xml = f'<step id="{target_step}">{compressed_inner}</step>'
        compress_ratio = new_chars / max(1, old_chars)

        updated_assistant = dict(assistant_message)
        assistant_extra = dict(self._message_extra(updated_assistant))
        assistant_extra["compression_meta"] = {
            "type": "agentdiet",
            "step": target_step,
            "original_chars": old_chars,
            "compressed_chars": new_chars,
            "compress_ratio": compress_ratio,
            "compressed_step_xml": compressed_step_xml,
            "timestamp": time.time(),
        }
        print(
            f"[compression] agentdiet applied at step={current_step} target_step={target_step} "
            f"original_chars={old_chars} compressed_chars={new_chars} "
            f"compress_ratio={compress_ratio:.2f}"
        )
        updated_assistant["content"] = (
            f"{AGENTDIET_MARKER} step={target_step}\n{compressed_step_xml}"
        )
        updated_assistant["extra"] = assistant_extra
        messages[target_slice.start_idx] = updated_assistant

        for idx in target_slice.observation_indices:
            msg = dict(messages[idx])
            extra = dict(self._message_extra(msg))
            returncode = extra.get("returncode", "unknown")
            msg["content"] = (
                f"{AGENTDIET_MARKER} step={target_step} "
                "compressed into assistant step payload; "
                f"returncode={returncode}"
            )
            extra["compression_meta"] = {
                "type": "agentdiet_placeholder",
                "step": target_step,
                "collapsed_to": target_slice.start_idx,
                "timestamp": time.time(),
            }
            msg["extra"] = extra
            messages[idx] = msg

        return True, f"agentdiet_step_{target_step}"

    def _apply_obsmasking(
        self,
        *,
        messages: list[dict[str, Any]],
        current_step: int,
    ) -> tuple[bool, str]:
        cfg = self.config.obsmasking
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
                "type": "obsmasking",
                "step": step_idx,
                "original_chars": original_chars,
                "age_steps": age_steps,
                "timestamp": time.time(),
            }
            print(
                f"[compression] obsmasking step={step_idx} age={age_steps} chars={original_chars} returncode={returncode}"
            )
            msg["extra"] = extra
            messages[idx] = msg

            masked_count += 1
            if masked_count >= cfg.max_masks_per_query:
                break

        if masked_count == 0:
            return False, "no_observation_masked"
        return True, f"masked_{masked_count}"
