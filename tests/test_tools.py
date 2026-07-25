import pytest
from app.tools.registry import Tool, ToolRegistry


def test_register_and_get_tool():
    reg = ToolRegistry()
    tool = Tool("echo", "Echo input", lambda x: x)
    reg.register(tool)
    assert reg.get("echo") is not None
    assert reg.get("echo").name == "echo"


def test_execute_tool():
    reg = ToolRegistry()
    tool = Tool("upper", "Uppercase input", lambda x: x.upper())
    reg.register(tool)
    result = reg.execute("upper", "hello")
    assert result == "HELLO"


def test_execute_unknown_tool():
    reg = ToolRegistry()
    result = reg.execute("nonexistent", "input")
    assert "ERROR" in result


def test_tool_error_handling():
    def bad_tool(x):
        raise ValueError("boom")

    reg = ToolRegistry()
    tool = Tool("bad", "Fails", bad_tool)
    reg.register(tool)
    result = reg.execute("bad", "input")
    assert "ERROR" in result
    assert "boom" in result


def test_list_schemas():
    reg = ToolRegistry()
    reg.register(Tool("a", "Tool A", lambda x: x, safe=True))
    reg.register(Tool("b", "Tool B", lambda x: x, safe=False))
    schemas = reg.list_schemas()
    assert len(schemas) == 2
    assert any(s["name"] == "a" and s["safe"] for s in schemas)
    assert any(s["name"] == "b" and not s["safe"] for s in schemas)


def test_builtin_tools_loaded():
    from app.tools.registry import registry
    tools = registry.list_tools()
    tool_names = {t.name for t in tools}
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "list_directory" in tool_names
    assert "run_shell" in tool_names
    assert "web_search" in tool_names
    assert "fetch_url" in tool_names
