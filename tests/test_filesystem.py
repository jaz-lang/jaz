"""Filesystem contracts against real temporary files, with no model/provider calls."""

import copy
import gc
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from jaz.filesystem import Directory, File, from_local


@pytest.fixture
def root(tmp_path):
    return from_local(tmp_path, access="write")


def test_shared_handles_and_python_copies(root):
    writable = root.file("note.txt")
    writable.write_text("before")
    readonly = writable.handle(access="read")
    other = root.handle(access="write").file("note.txt")
    assert readonly is not writable
    assert readonly.access == "read"
    other.replace_text("before", "after")
    assert readonly.read_text() == writable.read_text() == "after"
    assert copy.copy(readonly) is readonly
    assert copy.deepcopy({"file": readonly})["file"] is readonly


@pytest.mark.parametrize(
    "operation",
    [
        lambda file: file.write_text("changed"),
        lambda file: file.replace_text("original", "changed"),
        lambda file: file.handle(access="write"),
    ],
)
def test_read_handles_reject_writes(root, operation):
    file = root.file("note.txt")
    file.write_text("original")
    readonly = file.handle(access="read")
    with pytest.raises(PermissionError):
        operation(readonly)
    assert file.read_text() == "original"


def test_default_read_access_and_directory_attenuation(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.txt").write_text("hello", encoding="utf-8")
    root = from_local(tmp_path)
    with pytest.raises(PermissionError):
        root.handle(access="write")
    sub = root.directory("sub")
    file = sub.list()[0]
    assert file.access == "read"
    assert file.read_text() == "hello"
    with pytest.raises(PermissionError):
        file.write_text("changed")
    with pytest.raises(PermissionError):
        sub.file("../sibling.txt")


@pytest.mark.parametrize("access", [None, "", "rw", "READ", 1])
def test_invalid_access_rejected(tmp_path, access):
    root = from_local(tmp_path)
    with pytest.raises(ValueError):
        root.handle(access=access)
    with pytest.raises(ValueError):
        from_local(tmp_path, access=access)


def test_access_rejects_custom_equality(root):
    class ImpersonatedAccess(str):
        def __eq__(self, other):
            return other == "read"

        def __ne__(self, other):
            return False

    with pytest.raises(ValueError):
        root.handle(access="read").handle(access=ImpersonatedAccess("write"))


def test_overriding_public_metadata_cannot_grant_access(root, monkeypatch):
    readonly = root.handle(access="read")
    monkeypatch.setattr(Directory, "access", "write")
    child = readonly.file("note.txt")
    assert child.access == "read"
    with pytest.raises(PermissionError):
        child.write_text("denied")


def test_grants_have_no_mutable_state_or_root_factory(root):
    for handle in (root, root.file("note.txt")):
        for name, value in (("access", "write"), ("name", "other"), ("_path", "/")):
            with pytest.raises(AttributeError):
                setattr(handle, name, value)
        assert not hasattr(handle, "__dict__")
        assert not hasattr(handle, "from_local")
        assert not hasattr(type(handle), "from_local")
        with pytest.raises(TypeError, match="from_local"):
            type(handle)()


def test_unused_handles_are_collectible(root):
    handle = root.file("note.txt")
    ref = weakref.ref(handle)
    del handle
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize(
    "path", ["../outside.txt", "sub/../../outside.txt", "/tmp/outside.txt"]
)
def test_directory_scope_rejects_escape(root, path):
    with pytest.raises(PermissionError):
        root.file(path)
    with pytest.raises(PermissionError):
        root.directory(path)


def test_symlinks_cannot_expand_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (project / "escape.txt").symlink_to(outside)
    (project / "parent").symlink_to(tmp_path, target_is_directory=True)
    root = from_local(project, access="write")
    with pytest.raises(PermissionError):
        root.file("escape.txt")
    with pytest.raises(PermissionError):
        root.directory("parent")
    assert root.list() == []
    assert "private" not in str(root)


def test_in_scope_symlink_pins_target_and_keeps_entry_name(root, tmp_path):
    file = root.file("target.txt")
    file.write_text("original")
    alias_path = tmp_path / "alias.txt"
    alias_path.symlink_to(tmp_path / "target.txt")
    alias = root.file("alias.txt")
    assert alias.name == "alias.txt"
    alias.replace_text("original", "shared")
    assert file.read_text() == "shared"
    alias_path.unlink()
    alias_path.symlink_to(tmp_path.parent / "elsewhere")
    assert alias.read_text() == "shared"


@pytest.mark.parametrize("existing", [False, True])
def test_path_replaced_with_symlink_is_rejected(root, tmp_path, existing):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "sub").mkdir()
    sub = root.directory("sub")
    handle = sub.file("note.txt")
    if existing:
        handle.write_text("before")
        (tmp_path / "sub" / "note.txt").unlink()
    (tmp_path / "sub").rmdir()
    (tmp_path / "sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        handle.write_text("outside")
    with pytest.raises(PermissionError):
        sub.file("another.txt")
    assert "Entries unavailable" in str(sub)
    assert list(outside.iterdir()) == []


def test_text_reads_and_exact_replacements_preserve_newlines(root, tmp_path):
    file = root.file("note.txt")
    original = "first\r\nβeta\nlast"
    file.write_text(original)
    assert file.read_text() == original
    assert file.read_text(offset=2, limit=1) == "βeta\n"
    assert file.read_text(offset=3) == "last"
    assert file.read_text(offset=4) == ""
    file.replace_text("βeta\n", "short\r\n")
    assert (tmp_path / "note.txt").read_bytes() == b"first\r\nshort\r\nlast"
    file.write_text("x")
    assert file.read_text() == "x"
    file.write_text("")
    assert file.read_text() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": 0},
        {"offset": -1},
        {"offset": True},
        {"offset": 1.5},
        {"limit": 0},
        {"limit": -2},
        {"limit": False},
        {"limit": "2"},
    ],
)
def test_read_argument_validation(root, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        root.file("note.txt").read_text(**kwargs)


def test_read_limit_defaults_to_200_lines(root):
    file = root.file("note.txt")
    file.write_text("line\n" * 201)
    assert file.read_text() == "line\n" * 200
    assert file.read_text(offset=201) == "line\n"


@pytest.mark.parametrize("text", ["a" * 65536, "é" * 32768])
def test_read_byte_ceiling_is_explicit_and_utf8_aware(root, text):
    file = root.file("note.txt")
    file.write_text(text)
    assert file.read_text() == text
    file.write_text(text + "\nnext\n")
    with pytest.raises(ValueError, match="64 KiB"):
        file.read_text(limit=1)
    assert file.read_text(offset=2) == "next\n"


def test_long_skipped_lines_do_not_hide_following_lines(root):
    file = root.file("note.txt")
    file.write_text("x" * 200000 + "\nlast\n")
    assert file.read_text(offset=2, limit=1) == "last\n"


@pytest.mark.parametrize(
    "contents, old",
    [
        ("hello", ""),
        ("hello", "missing"),
        ("hello hello", "hello"),
        ("aaa", "aa"),
    ],
)
def test_replace_requires_one_unambiguous_match(root, contents, old):
    file = root.file("note.txt")
    file.write_text(contents)
    with pytest.raises(ValueError):
        file.replace_text(old, "replacement")
    assert file.read_text() == contents
    file.write_text("lock released")
    assert file.read_text() == "lock released"


def test_invalid_text_does_not_truncate_existing_file(root):
    file = root.file("note.txt")
    file.write_text("original")
    with pytest.raises(TypeError):
        file.write_text(b"bytes")
    with pytest.raises(UnicodeEncodeError):
        file.write_text("\ud800")
    with pytest.raises(UnicodeEncodeError):
        file.replace_text("original", "\ud800")
    assert file.read_text() == "original"


def test_missing_files_and_parent_directories(root, tmp_path):
    missing = root.file("missing.txt")
    with pytest.raises(FileNotFoundError):
        missing.read_text()
    with pytest.raises(FileNotFoundError):
        missing.replace_text("old", "new")
    assert not (tmp_path / "missing.txt").exists()
    with pytest.raises(FileNotFoundError):
        root.file("absent/note.txt").write_text("hello")
    with pytest.raises(FileNotFoundError):
        root.directory("absent")
    missing.write_text("created")
    assert missing.read_text() == "created"
    (tmp_path / "missing.txt").unlink()
    with pytest.raises(FileNotFoundError):
        missing.read_text()


def test_type_mismatches_and_missing_roots(root, tmp_path):
    root.file("note.txt").write_text("text")
    with pytest.raises(NotADirectoryError):
        from_local(tmp_path / "note.txt")
    with pytest.raises(FileNotFoundError):
        from_local(tmp_path / "missing")
    with pytest.raises(NotADirectoryError):
        root.directory("note.txt")
    with pytest.raises(IsADirectoryError):
        root.file(".")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special files")
def test_special_files_are_rejected_without_blocking(root, tmp_path):
    future = root.file("pipe")
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="regular files"):
        root.file("pipe")
    with pytest.raises(ValueError, match="regular files"):
        future.read_text()
    assert root.list() == []


def test_listing_and_descriptions_are_shallow_and_bounded(root, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "hidden.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "broken").symlink_to(tmp_path / "missing")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    for index in range(25):
        root.file(f"{index:02}.txt").write_text("never rendered")
    entries = root.handle(access="read").list()
    assert [entry.name for entry in entries] == [f"{i:02}.txt" for i in range(25)] + [
        "sub"
    ]
    assert all(entry.access == "read" for entry in entries)
    assert isinstance(entries[-1], Directory)
    assert isinstance(entries[0], File)
    description = str(root)
    assert sum(line.startswith("    '") for line in description.splitlines()) == 20
    assert "more entries" in description
    assert "hidden.txt" not in description
    assert "never rendered" not in description
    assert "write_text" not in str(entries[0])
    assert "write_text" in str(root.file("00.txt"))
    assert str(tmp_path) not in description
    assert "hidden.txt" not in repr(entries)


@pytest.mark.parametrize("alias_kind", ["same_path", "symlink", "hardlink"])
def test_parallel_exact_edits_share_locks_across_independent_roots(
    tmp_path, alias_kind
):
    count = 24
    file = from_local(tmp_path, access="write").file("note.txt")
    file.write_text("".join(f"<{i}>\n" for i in range(count)))
    alias_path = tmp_path / "alias.txt"
    if alias_kind == "symlink":
        alias_path.symlink_to(tmp_path / "note.txt")
    elif alias_kind == "hardlink":
        os.link(tmp_path / "note.txt", alias_path)
    else:
        alias_path = tmp_path / "note.txt"
    barrier = Barrier(count)

    def edit(index):
        # Independent registration must not create an independent lock.
        name = "note.txt" if index % 2 else alias_path.name
        handle = from_local(tmp_path, access="write").file(name)
        barrier.wait(timeout=10)
        handle.replace_text(f"<{index}>", f"done-{index}")

    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(edit, range(count)))
    assert file.read_text() == "".join(f"done-{i}\n" for i in range(count))
