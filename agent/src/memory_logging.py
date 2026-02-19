import json
from datetime import datetime, timezone


DEFAULT_PREVIEW_LIMIT = 1200


def preview_text(text: str, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
    text = text or ""
    if limit <= 0:
        return text
    if len(text) <= limit * 2:
        return text
    snip_count = len(text) - limit * 2
    return f"{text[:limit]}\n...<SNIP {snip_count} CHARS>...\n{text[-limit:]}"


def build_memory_output_meta(
    stdout: str, stderr: str, limit: int = DEFAULT_PREVIEW_LIMIT
) -> dict:
    stdout = stdout or ""
    stderr = stderr or ""
    return {
        "memory_stdout_preview": preview_text(stdout, limit),
        "memory_stderr_preview": preview_text(stderr, limit),
        "output_len": len(stdout) + len(stderr),
    }


def build_memory_log_payload(
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
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> dict:
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": returncode,
        "is_write": bool(is_write),
        "output_len": output_len if output_len is not None else len(stdout) + len(stderr),
        "stdout_preview": preview_text(stdout, preview_limit),
        "stderr_preview": preview_text(stderr, preview_limit),
        "path_hint": path_hint,
    }
    if details:
        payload["details"] = details
    return payload


def emit_memory_log(
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
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> None:
    try:
        payload = build_memory_log_payload(
            event,
            command=command,
            returncode=returncode,
            is_write=is_write,
            stdout=stdout,
            stderr=stderr,
            path_hint=path_hint,
            output_len=output_len,
            details=details,
            preview_limit=preview_limit,
        )
        print(f"MEMORY_LOG {json.dumps(payload, ensure_ascii=False)}", flush=True)
    except Exception as e:
        safe_command = (command or "").replace("\n", "\\n")
        if len(safe_command) > 120:
            safe_command = safe_command[:117] + "..."
        print(
            f"MEMORY_LOG_FAILED event={event} type={type(e).__name__} command={safe_command}",
            flush=True,
        )


def build_one_line_memory_hint(
    event: str,
    command: str,
    returncode: int,
    is_write: bool,
) -> str:
    safe_command = (command or "").replace("\n", "\\n").strip()
    if len(safe_command) > 96:
        safe_command = safe_command[:93] + "..."
    return f"MEMORY_EVT event={event} rc={returncode} write={int(bool(is_write))} cmd={safe_command}"
