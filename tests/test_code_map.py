"""Tests for the code_map module."""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from memfun_agent.code_map import (
    Definition,
    FileMap,
    _extract_generic,
    _extract_javascript,
    _extract_python,
    _format_file,
    build_code_map,
    code_map_to_string,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── Python Extraction ────────────────────────────────────────


class TestExtractPython:
    def test_simple_function(self) -> None:
        src = "def hello(name: str) -> str:\n    return f'hi {name}'\n"
        fm = _extract_python(src, "mod.py", len(src))
        assert len(fm.definitions) == 1
        d = fm.definitions[0]
        assert d.kind == "function"
        assert d.name == "hello"
        assert "name: str" in d.signature
        assert "-> str" in d.signature

    def test_async_function(self) -> None:
        src = "async def fetch(url: str) -> bytes:\n    pass\n"
        fm = _extract_python(src, "net.py", len(src))
        assert len(fm.definitions) == 1
        assert fm.definitions[0].signature.startswith("async def")

    def test_class_with_methods(self) -> None:
        src = textwrap.dedent("""\
            class Engine(Base):
                def __init__(self, config: Config) -> None:
                    pass

                def run(self) -> Result:
                    pass

                def __repr__(self) -> str:
                    pass

                def _private(self) -> None:
                    pass
        """)
        fm = _extract_python(src, "engine.py", len(src))
        names = [d.name for d in fm.definitions]
        # class + __init__ + run + _private (skips __repr__)
        assert "Engine" in names
        assert "Engine.__init__" in names
        assert "Engine.run" in names
        assert "Engine._private" in names
        assert "Engine.__repr__" not in names

    def test_class_bases(self) -> None:
        src = "class Foo(Bar, Baz):\n    pass\n"
        fm = _extract_python(src, "foo.py", len(src))
        assert fm.definitions[0].signature == "class Foo(Bar, Baz)"

    def test_class_no_bases(self) -> None:
        src = "class Plain:\n    pass\n"
        fm = _extract_python(src, "plain.py", len(src))
        assert fm.definitions[0].signature == "class Plain"

    def test_decorated_function(self) -> None:
        src = "@app.route('/api')\ndef handler(req: Request) -> Response:\n    pass\n"
        fm = _extract_python(src, "api.py", len(src))
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "handler"

    def test_syntax_error(self) -> None:
        src = "def broken(:\n"
        fm = _extract_python(src, "bad.py", len(src))
        assert fm.definitions == ()
        assert fm.path == "bad.py"

    def test_empty_file(self) -> None:
        fm = _extract_python("", "empty.py", 0)
        assert fm.definitions == ()

    def test_nested_class_not_captured(self) -> None:
        src = textwrap.dedent("""\
            class Outer:
                class Inner:
                    pass
                def method(self) -> None:
                    pass
        """)
        fm = _extract_python(src, "nested.py", len(src))
        names = [d.name for d in fm.definitions]
        assert "Outer" in names
        assert "Outer.method" in names
        # Inner is nested inside Outer body, not a top-level child
        assert "Inner" not in names

    def test_long_annotation_truncated(self) -> None:
        long_type = "dict[str, list[tuple[int, float, complex, bytes, str, bool]]]"
        src = f"def f(x: {long_type}) -> None:\n    pass\n"
        fm = _extract_python(src, "long.py", len(src))
        sig = fm.definitions[0].signature
        # Should be truncated with ...
        assert "..." in sig or len(long_type) <= 60

    def test_star_args(self) -> None:
        src = "def f(*args: int, **kwargs: str) -> None:\n    pass\n"
        fm = _extract_python(src, "star.py", len(src))
        sig = fm.definitions[0].signature
        assert "*args" in sig
        assert "**kwargs" in sig


# ── JavaScript / TypeScript Extraction ────────────────────────


class TestExtractJavaScript:
    def test_export_function(self) -> None:
        src = "export function fetchData(url, options) {\n}\n"
        fm = _extract_javascript(src, "api.js", len(src))
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "fetchData"
        assert fm.definitions[0].kind == "function"

    def test_class_extends(self) -> None:
        src = "export class ApiClient extends BaseClient {\n}\n"
        fm = _extract_javascript(src, "client.ts", len(src))
        assert len(fm.definitions) == 1
        d = fm.definitions[0]
        assert d.name == "ApiClient"
        assert d.kind == "class"
        assert "BaseClient" in d.signature

    def test_class_no_extends(self) -> None:
        src = "class Router {\n}\n"
        fm = _extract_javascript(src, "router.js", len(src))
        assert fm.definitions[0].signature == "class Router"

    def test_arrow_function(self) -> None:
        src = "export const handler = async (req, res) => {\n}\n"
        fm = _extract_javascript(src, "handler.ts", len(src))
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "handler"

    def test_typescript_interface(self) -> None:
        src = "export interface UserConfig {\n  name: string\n}\n"
        fm = _extract_javascript(src, "types.ts", len(src))
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "UserConfig"
        assert "interface" in fm.definitions[0].signature

    def test_typescript_type(self) -> None:
        src = "export type Result = Success | Failure\n"
        fm = _extract_javascript(src, "types.ts", len(src))
        # type and interface share the same regex
        assert len(fm.definitions) == 1

    def test_multiple_defs(self) -> None:
        src = textwrap.dedent("""\
            export class App {
            }
            export function init(config) {
            }
            export const run = () => {
            }
        """)
        fm = _extract_javascript(src, "app.ts", len(src))
        names = {d.name for d in fm.definitions}
        assert names == {"App", "init", "run"}


# ── Generic Language Extraction ───────────────────────────────


class TestExtractGeneric:
    def test_go_func(self) -> None:
        src = "func ProcessRequest(w http.ResponseWriter, r *http.Request) {\n}\n"
        fm = _extract_generic(src, "handler.go", len(src), "go")
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "ProcessRequest"

    def test_go_method(self) -> None:
        src = "func (s *Server) Start(addr string) error {\n}\n"
        fm = _extract_generic(src, "server.go", len(src), "go")
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "Start"

    def test_rust_fn(self) -> None:
        src = "pub async fn handle(req: Request) -> Response {\n}\n"
        fm = _extract_generic(src, "api.rs", len(src), "rust")
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "handle"

    def test_java_class(self) -> None:
        src = "public class UserService {\n}\n"
        fm = _extract_generic(src, "UserService.java", len(src), "java")
        assert len(fm.definitions) == 1
        assert fm.definitions[0].name == "UserService"

    def test_unknown_language(self) -> None:
        fm = _extract_generic("some code", "file.xyz", 9, "unknown")
        assert fm.definitions == ()
        assert fm.path == "file.xyz"


# ── build_code_map ────────────────────────────────────────────


class TestBuildCodeMap:
    def test_with_manifest(self, tmp_path: Path) -> None:
        # Create a small Python file
        py_file = tmp_path / "app.py"
        py_file.write_text("def main() -> None:\n    pass\n")

        manifest = [("app.py", py_file.stat().st_size)]
        maps = build_code_map(tmp_path, manifest=manifest)
        assert len(maps) == 1
        assert maps[0].path == "app.py"
        assert len(maps[0].definitions) == 1

    def test_mixed_languages(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("class App:\n    pass\n")
        (tmp_path / "util.js").write_text("function helper() {}\n")
        (tmp_path / "data.txt").write_text("plain text")

        manifest = [
            ("app.py", 20),
            ("util.js", 21),
            ("data.txt", 10),
        ]
        maps = build_code_map(tmp_path, manifest=manifest)
        assert len(maps) == 3

        py_map = next(m for m in maps if m.path == "app.py")
        assert len(py_map.definitions) == 1

        js_map = next(m for m in maps if m.path == "util.js")
        assert len(js_map.definitions) == 1

        txt_map = next(m for m in maps if m.path == "data.txt")
        assert len(txt_map.definitions) == 0

    def test_max_files(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"mod{i}.py").write_text(f"def f{i}(): pass\n")

        manifest = [(f"mod{i}.py", 20) for i in range(10)]
        maps = build_code_map(tmp_path, manifest=manifest, max_files=3)
        assert len(maps) == 3

    def test_missing_file(self, tmp_path: Path) -> None:
        manifest = [("nonexistent.py", 100)]
        maps = build_code_map(tmp_path, manifest=manifest)
        assert len(maps) == 1
        assert maps[0].definitions == ()


# ── code_map_to_string ────────────────────────────────────────


class TestCodeMapToString:
    def test_empty(self) -> None:
        assert code_map_to_string([]) == ""

    def test_basic_format(self) -> None:
        fm = FileMap(
            path="src/app.py",
            definitions=(
                Definition("App", "class", "class App(Base)", 1),
                Definition("App.__init__", "method", "def __init__(self)", 2),
                Definition("main", "function", "def main() -> None", 10),
            ),
            size=1500,
        )
        result = code_map_to_string([fm])
        assert "src/app.py" in result
        assert "class App(Base)" in result
        assert "def __init__(self)" in result
        assert "def main() -> None" in result

    def test_method_double_indented(self) -> None:
        fm = FileMap(
            path="m.py",
            definitions=(
                Definition("C", "class", "class C", 1),
                Definition("C.run", "method", "def run(self)", 2),
            ),
            size=100,
        )
        result = code_map_to_string([fm])
        lines = result.split("\n")
        # Method should be double-indented
        method_line = next(ln for ln in lines if "def run" in ln)
        assert method_line.startswith("    ")  # 4 spaces (2x indent)

    def test_truncation(self) -> None:
        # Create many files that exceed the token budget
        maps: list[FileMap] = []
        for i in range(100):
            maps.append(FileMap(
                path=f"pkg/module_{i:03d}.py",
                definitions=(
                    Definition(f"Class{i}", "class", f"class Class{i}", 1),
                    Definition(f"func_{i}", "function", f"def func_{i}(x: int) -> str", 10),
                ),
                size=500,
            ))

        # Very small budget
        result = code_map_to_string(maps, max_tokens=200)
        assert "... and" in result
        assert "more files" in result

    def test_files_without_defs_as_compact(self) -> None:
        maps = [
            FileMap(
                path="main.py",
                definitions=(
                    Definition("main", "function", "def main()", 1),
                ),
                size=100,
            ),
            FileMap(path="data.csv", definitions=(), size=5000),
        ]
        result = code_map_to_string(maps)
        assert "main.py" in result
        assert "def main()" in result
        assert "data.csv" in result
        # data.csv should just be a compact line (no definitions)
        csv_line = next(ln for ln in result.split("\n") if "data.csv" in ln)
        assert "4.9KB" in csv_line

    def test_prioritizes_classes(self) -> None:
        maps = [
            FileMap(
                path="funcs.py",
                definitions=(
                    Definition("helper", "function", "def helper()", 1),
                ),
                size=100,
            ),
            FileMap(
                path="classes.py",
                definitions=(
                    Definition("Engine", "class", "class Engine", 1),
                ),
                size=100,
            ),
        ]
        # With tight budget, class-containing file should come first
        result = code_map_to_string(maps, max_tokens=500)
        lines = result.split("\n")
        class_idx = next(i for i, ln in enumerate(lines) if "classes.py" in ln)
        func_idx = next(i for i, ln in enumerate(lines) if "funcs.py" in ln)
        assert class_idx < func_idx


# ── _format_file ──────────────────────────────────────────────


class TestFormatFile:
    def test_file_with_no_definitions(self) -> None:
        fm = FileMap(path="config.yaml", size=200)
        result = _format_file(fm)
        assert result == "config.yaml (200B)"

    def test_file_size_formatting(self) -> None:
        fm = FileMap(path="big.py", definitions=(), size=2048)
        result = _format_file(fm)
        assert "2.0KB" in result
