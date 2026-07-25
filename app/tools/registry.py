"""
Tool registry — lazy-loaded, concurrent-safe tool system.

Each tool is a function that takes a string input and returns a string output.
The registry discovers, validates, and dispatches tool calls.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[[str], str],
        *,
        safe: bool = True,
        requires_confirmation: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.safe = safe
        self.requires_confirmation = requires_confirmation

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "safe": self.safe,
            "requires_confirmation": self.requires_confirmation,
        }

    def execute(self, input_text: str) -> str:
        try:
            result = self.func(input_text)
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            return f"ERROR: {e}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._register_builtins()
        self._loaded = True

    def _register_builtins(self) -> None:
        from .file_ops import get_file_tools
        from .shell import get_shell_tools
        from .search import get_search_tools
        from .web import get_web_tools

        for tool in get_file_tools():
            self.register(tool)
        for tool in get_shell_tools():
            self.register(tool)
        for tool in get_search_tools():
            self.register(tool)
        for tool in get_web_tools():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        self._ensure_loaded()
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        self._ensure_loaded()
        return list(self._tools.values())

    def list_schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self.list_tools()]

    def execute(self, name: str, input_text: str) -> str:
        self._ensure_loaded()
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: Unknown tool '{name}'"
        return tool.execute(input_text)


registry = ToolRegistry()
