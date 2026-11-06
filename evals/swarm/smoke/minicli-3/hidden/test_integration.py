import io

from minicli.main import build_parser, main


def test_all_subcommands_registered(tmp_path):
    help_text = build_parser().format_help()
    for name in ("stats", "convert", "hashsum"):
        assert name in help_text
    f = tmp_path / "x"
    f.write_text("x")
    for argv in (["stats", "1"], ["convert", "1", "m", "cm"], ["hashsum", str(f)]):
        assert main(argv, io.StringIO()) == 0
