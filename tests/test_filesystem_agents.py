"""Exercise filesystem handles through the actual JAZ runtime and default REPL policy."""

import asyncio
import json

import pytest

import jaz
from jaz.filesystem import File, from_local
from jaz.hooks import IterationLimit, ReturnType
from jaz.llm import MockLLM
from jaz.repl import PythonREPL


def scripted(*responses):
    replies = iter(responses)
    messages_seen = []

    def reply(messages):
        messages_seen.append(messages)
        return next(replies)

    return jaz.ConfigOverride(llm=MockLLM(reply), repl=PythonREPL()), messages_seen


def test_agent_input_description_and_native_return(tmp_path):
    (tmp_path / "note.txt").write_text("hello\nworld\n", encoding="utf-8")
    file = from_local(tmp_path).file("note.txt")
    config, seen = scripted("return (document, document.read_text(offset=2, limit=1))")
    result, text = jaz.invoke(
        config, IterationLimit(2), task="read a file", document=file
    )
    assert result is file
    assert text == "world\n"
    prompt = json.dumps(seen[0])
    assert "read_text(*, offset=1, limit=200)" in prompt
    assert "Explicitly supplied file/directory handles grant separate access" in prompt
    assert "hello" not in prompt


def test_nested_handoff_keeps_parent_access_and_returned_handle_alive(tmp_path):
    file = from_local(tmp_path, access="write").file("note.txt")
    file.write_text("original")
    config, _ = scripted(
        "child = invoke(task='edit', document=document)\n"
        "document.replace_text('child', 'parent')\nreturn child",
        "document.replace_text('original', 'child')\nreturn document",
    )
    with config:
        result = jaz.invoke(
            IterationLimit(3), ReturnType(File), task="delegate", document=file
        )
    assert result is file
    assert result.read_text() == "parent"
    result.write_text("after return")
    assert file.read_text() == "after return"


@pytest.mark.parametrize(
    "code, message",
    [
        ("document.write_text('bad')", "read-only"),
        ("document.handle(access='write')", "cannot grant write"),
        ("open(path).read()", "not allowed"),
        ("document.access = 'write'", "AttributeError"),
        ("type(document).from_local('/')", "AttributeError"),
    ],
)
def test_default_repl_does_not_widen_permissions(tmp_path, code, message):
    path = tmp_path / "note.txt"
    path.write_text("original", encoding="utf-8")
    file = from_local(tmp_path).file("note.txt")
    config, seen = scripted(code, "return document.read_text()")
    result = jaz.invoke(
        config, IterationLimit(3), task="check access", document=file, path=str(path)
    )
    assert result == "original"
    assert message in json.dumps(seen[-1])
    assert path.read_text(encoding="utf-8") == "original"


def test_directory_navigation_inside_agent(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.txt").write_text("data", encoding="utf-8")
    config, seen = scripted(
        "return project.directory('sub').file('note.txt').read_text()"
    )
    assert jaz.invoke(config, IterationLimit(2), project=from_local(tmp_path)) == "data"
    assert "directory(path): get an existing subdirectory" in json.dumps(seen[0])


def test_write_handle_does_not_grant_raw_open_access(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("original", encoding="utf-8")
    file = from_local(tmp_path, access="write").file("note.txt")
    config, seen = scripted(
        "open(path, 'w').write('bad')", "return document.read_text()"
    )
    assert (
        jaz.invoke(config, IterationLimit(3), document=file, path=str(path))
        == "original"
    )
    assert "not allowed" in json.dumps(seen[-1])
    assert path.read_text(encoding="utf-8") == "original"


def test_async_agent_roundtrip(tmp_path):
    file = from_local(tmp_path, access="write").file("note.txt")
    config, _ = scripted("document.write_text('async')\nreturn document")
    result = asyncio.run(jaz.ainvoke(config, IterationLimit(2), document=file))
    assert result is file
    assert file.read_text() == "async"
