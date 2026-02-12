"""Tests for RLM code cleaning (markdown fence stripping)."""

from memfun_agent.rlm import _clean_generated_code


def test_plain_python_unchanged():
    code = "files = list_files('.')\nfor f in files:\n    print(f)"
    assert _clean_generated_code(code) == code.strip()


def test_strip_python_fences():
    code = '```python\nfiles = list_files(".")\nprint(files)\n```'
    result = _clean_generated_code(code)
    assert "```" not in result
    assert 'files = list_files(".")' in result
    assert "print(files)" in result


def test_strip_py_fences():
    code = "```py\nprint('hello')\n```"
    result = _clean_generated_code(code)
    assert "```" not in result
    assert "print('hello')" in result


def test_strip_bare_fences():
    code = "```\nx = 1\n```"
    result = _clean_generated_code(code)
    assert "```" not in result
    assert "x = 1" in result


def test_multiple_code_blocks():
    code = (
        "First I'll list files:\n"
        "```python\nfiles = list_files('.')\n```\n"
        "Then read them:\n"
        "```python\nfor f in files:\n"
        "    content = read_file(f)\n```"
    )
    result = _clean_generated_code(code)
    assert "```" not in result
    assert "list_files('.')" in result
    assert "read_file(f)" in result


def test_empty_code():
    assert _clean_generated_code("") == ""
    assert _clean_generated_code("   ") == ""


def test_whitespace_preserved_inside():
    code = "```python\nif True:\n    print('indented')\n```"
    result = _clean_generated_code(code)
    assert "    print('indented')" in result


def test_no_fences_just_code():
    code = "state['FINAL'] = 'done'"
    assert _clean_generated_code(code) == code
