import io
import json

from minicli.main import main


def run(argv):
    out = io.StringIO()
    rc = main(argv, out)
    return rc, out.getvalue()


def test_text_output():
    rc, text = run(["stats", "4", "1", "3", "2", "10"])
    assert rc == 0
    assert text.splitlines() == ["count 5", "mean 4", "median 3", "min 1", "max 10"]


def test_even_median_and_floats():
    rc, text = run(["stats", "1.5", "2.5", "-1", "7"])
    assert "median 2" in text.splitlines()
    assert "mean 2.5" in text.splitlines()


def test_json():
    rc, text = run(["stats", "--json", "2", "4"])
    assert json.loads(text) == {"count": 2, "mean": 3, "median": 3, "min": 2, "max": 4}
