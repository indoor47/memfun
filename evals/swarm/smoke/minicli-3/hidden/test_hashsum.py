import hashlib
import io

from minicli.main import main


def test_sha256_default_and_missing(tmp_path):
    a = tmp_path / "a.txt"
    a.write_bytes(b"hello\n")
    missing = tmp_path / "nope.txt"
    out = io.StringIO()
    rc = main(["hashsum", str(a), str(missing)], out)
    assert rc == 1
    lines = out.getvalue().splitlines()
    assert lines[0] == f"{hashlib.sha256(b'hello' + bytes([10])).hexdigest()}  {a}"
    assert lines[1] == f"error: {missing}: not found"


def test_md5(tmp_path):
    b = tmp_path / "b.bin"
    b.write_bytes(bytes(range(256)) * 1000)
    out = io.StringIO()
    assert main(["hashsum", "--algo", "md5", str(b)], out) == 0
    assert out.getvalue() == f"{hashlib.md5(b.read_bytes()).hexdigest()}  {b}\n"
