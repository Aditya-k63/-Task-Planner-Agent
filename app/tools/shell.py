"""Shell execution tool — runs commands in sandboxed workspace."""

from __future__ import annotations

import subprocess
import os

from .registry import Tool

WORKSPACE = "/tmp/task-planner-workspace"
TIMEOUT = 30
BLOCKED = {"rm -rf /", "mkfs", "dd if=", "> /dev/sd", ":(){ :|:& };:"}


def _run_shell(command: str) -> str:
    cmd = command.strip()
    for pattern in BLOCKED:
        if pattern in cmd:
            return f"ERROR: Blocked dangerous command pattern: {pattern}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=WORKSPACE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > 5000:
            output = output[:5000] + f"\n... (truncated, {len(output)} total chars)"
        return output
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


def get_shell_tools() -> list[Tool]:
    return [
        Tool(
            "run_shell",
            "Execute a shell command in the workspace. Input: shell command string.",
            _run_shell,
            safe=False,
            requires_confirmation=True,
        ),
    ]
