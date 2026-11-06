import io

from minicli.main import main


def test_stats_count():
    out = io.StringIO()
    assert main(["stats", "1", "2", "3"], out) == 0
    assert "count 3" in out.getvalue()
