import pytest
from formats.inilite import dump_ini, parse_ini

TEXT = """
# comment
[server]
host = example.org
url = http://x/?a=b

; another
[client]
retries=3
[server]
host = override
"""


def test_parse():
    assert parse_ini(TEXT) == {
        "server": {"host": "override", "url": "http://x/?a=b"},
        "client": {"retries": "3"},
    }


@pytest.mark.parametrize(("text", "line"), [("x = 1", 1), ("[a]\nok = 1\njunk", 3)])
def test_errors_name_line(text, line):
    with pytest.raises(ValueError, match=f"line {line}"):
        parse_ini(text)


def test_dump_format_and_roundtrip():
    data = {"a": {"x": "1", "y": "two words"}, "b": {}, "c": {"k": "v=w"}}
    text = dump_ini(data)
    assert text == "[a]\nx = 1\ny = two words\n\n[b]\n\n[c]\nk = v=w\n"
    assert parse_ini(text) == data
