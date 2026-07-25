"""File operations tools — read, write, list, search."""

from __future__ import annotations

import os
import pathlib

from .registry import Tool

WORKSPACE = pathlib.Path("/tmp/task-planner-workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)


def _read_file(path: str) -> str:
    p = WORKSPACE / path.strip()
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: Not a file: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 10000:
            return content[:10000] + f"\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def _write_file(args: str) -> str:
    parts = args.split("\n", 1)
    if len(parts) < 2:
        return "ERROR: Format: <path>\\n<content>"
    path, content = parts[0].strip(), parts[1]
    p = WORKSPACE / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: Wrote {len(content)} chars to {path}"


def _list_dir(path: str) -> str:
    p = WORKSPACE / path.strip().lstrip("/")
    if not p.exists():
        return f"ERROR: Directory not found: {path}"
    if not p.is_dir():
        return f"ERROR: Not a directory: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        prefix = "DIR " if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else 0
        entries.append(f"[{prefix}] {item.name} ({size} bytes)")
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


def _file_exists(path: str) -> str:
    p = WORKSPACE / path.strip()
    return f"EXISTS: {p.exists()}"


def _search_in_files(pattern: str) -> str:
    """Simple grep-like search across files in workspace."""
    import re
    try:
        regex = re.compile(pattern.strip(), re.IGNORECASE)
    except re.error as e:
        return f"ERROR: Invalid regex: {e}"

    matches = []
    for root, _, files in os.walk(WORKSPACE):
        for fname in files:
            fp = pathlib.Path(root) / fname
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if regex.search(line):
                        rel = fp.relative_to(WORKSPACE)
                        matches.append(f"{rel}:{i}: {line.strip()}")
                        if len(matches) >= 50:
                            return "\n".join(matches)
            except Exception:
                continue
    if not matches:
        return "No matches found"
    return "\n".join(matches)


def get_file_tools() -> list[Tool]:
    return [
        Tool("read_file", "Read contents of a file. Input: file path relative to workspace.", _read_file, safe=True),
        Tool("write_file", "Write content to a file. Input: path\\ncontent (newline-separated).", _write_file, safe=False),
        Tool("list_directory", "List files and directories. Input: directory path (or '.' for root).", _list_dir, safe=True),
        Tool("file_exists", "Check if a file exists. Input: file path.", _file_exists, safe=True),
        Tool("search_files", "Regex search across all files. Input: regex pattern.", _search_in_files, safe=True),
    ]
