"""Streaming a long run: grading that stays sealed until every rollout is final, shards joined
into one run, and the submission folder with its self-audit.  Hermetic: no docker, no model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from evals.swebench_live import grade as gr  # noqa: E402
from evals.swebench_live import runner as rn  # noqa: E402
from evals.swebench_live import sealed as sl  # noqa: E402
from evals.swebench_live import submission as sub  # noqa: E402
from tests.test_swebench_live_runner import SENTINELS  # noqa: E402
from tests.test_swebench_live_runner import env as env  # noqa: E402

TEMPLATE = "Fix the issue.\n<issue>\n{problem_statement}\n</issue>"
SETTINGS = {"model": "m", "prompt_template": TEMPLATE, "agent_version": "abc", "solver": "cf"}
TEST_LINE = "    assert widget('') == [], 'an empty input must give an empty list'"
ROW = {
    "instance_id": "octo__widget-1",
    "problem_statement": "Widget crashes on empty input.",
    "hints_text": "A maintainer says: use the frobnicate helper from widget.util for this case.",
    "test_patch": "diff --git a/t.py b/t.py\n+def test_widget_handles_empty_input():\n+"
    + TEST_LINE
    + "\n",
    "patch": "diff --git a/w.py b/w.py\n"
    "+    return frobnicate(value) if value else []  # the fix\n",
    "FAIL_TO_PASS": ["t.py::test_widget_handles_empty_input"],
}


def call(seq: int, prompt: str, answer: str) -> str:
    doc = {
        "seq": seq,
        "request": {"messages": [{"role": "user", "content": prompt}]},
        "response": {"choices": [{"message": {"content": answer}}]},
    }
    return json.dumps(doc)


def make_run(
    root: Path,
    ids: list[str],
    *,
    finalized: bool = True,
    settings: dict[str, Any] | None = None,
    calls: list[str] | None = None,
    log: str = "agent ran\n",
    record: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True)
    meta = {
        "settings": settings or SETTINGS,
        "instance_ids": ids,
        "created_at": 1.0,
        "finalized": finalized,
    }
    (root / "run.json").write_text(json.dumps(meta))
    preds = {}
    for iid in ids:
        idir = root / "instances" / iid
        idir.mkdir(parents=True)
        rec = {
            "instance_id": iid,
            "status": "done",
            "lockdown": "lockdown ok: 5 commits reachable",
            "network_selftest": {"proxy": True, "internet": [], "dns": [], "host_ports": []},
            "usage": {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20},
            "cost_usd": 0.01,
            **(record or {}),
        }
        (idir / "instance.json").write_text(json.dumps(rec))
        prompt = TEMPLATE.format(problem_statement=ROW["problem_statement"])
        lines = calls if calls is not None else [call(1, prompt, "I will edit w.py")]
        (idir / "llm.jsonl").write_text("\n".join(lines) + "\n")
        (idir / "agent.log").write_text(log)
        (idir / "patch.diff").write_text("diff --git a/w.py b/w.py\n")
        preds[iid] = {"model_name_or_path": "m", "model_patch": "diff --git a/w.py b/w.py\n"}
    (root / "preds.json").write_text(json.dumps(preds))
    return root


# ── sealed grading ───────────────────────────────────────────────────────────────────────────


def fake_evaluator(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The official evaluation replaced by something that writes a verdict where it was told."""
    cmds: list[list[str]] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        cmds.append(cmd)
        out = Path(cmd[cmd.index("--output_dir") + 1])
        iid = cmd[cmd.index("--instance_ids") + 1]
        (out / iid).mkdir(parents=True, exist_ok=True)
        (out / iid / "report.json").write_text('{"resolved": true, "VERDICT-SENTINEL": 1}')
        kw["stdout"].write(b"Success: 1 VERDICT-SENTINEL\n")
        return type("P", (), {"returncode": 1})()  # even the exit code must not get out

    monkeypatch.setattr(sl.subprocess, "run", run)
    return cmds


def test_a_final_instance_is_graded_where_nobody_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    iid = "octo__widget-1"
    run = make_run(tmp_path / "run", [iid], finalized=False)
    sealed = tmp_path / "sealed"
    cmds = fake_evaluator(monkeypatch)
    got = sl.grade_final(tmp_path / "eval", run, sealed, iid, "lite", gold=True)
    assert sorted(got) == ["agent_patch_evaluated_at", "gold_checked_at", "instance_id"]
    assert (sealed.stat().st_mode & 0o777) == 0o700
    first, second = cmds
    assert first[first.index("--patch_dir") + 1] == str(sealed / "preds" / f"{iid}.json")
    assert second[second.index("--patch_dir") + 1] == "gold"
    assert "--remove-image" not in first and "--remove-image" in second  # one pull, two uses
    assert first[first.index("--output_dir") + 1] == str(sealed / "eval")
    assert "VERDICT-SENTINEL" in (sealed / "logs" / f"{iid}.agent.log").read_text()
    # nothing of the outcome is outside the sealed directory: not in the run, not on the terminal
    outside = "".join(p.read_text() for p in run.rglob("*") if p.is_file())
    assert "SENTINEL" not in outside and "resolved" not in outside and "ended" not in outside
    assert capsys.readouterr().out == ""
    assert json.loads((run / "sealed.jsonl").read_text()).keys() == got.keys()


def test_only_a_final_instance_is_graded_and_only_a_finalized_run_is_unsealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iid = "octo__widget-1"
    run = make_run(tmp_path / "run", [iid], finalized=False, record={"status": "infra_error"})
    cmds = fake_evaluator(monkeypatch)
    with pytest.raises(SystemExit, match="not final"):
        sl.grade_final(tmp_path / "eval", run, tmp_path / "sealed", iid, "lite")
    assert cmds == []
    with pytest.raises(SystemExit, match="not finalized"):
        sl.unseal(tmp_path / "eval", run, sl.make(tmp_path / "sealed"), "lite", 1)


def test_unseal_collects_the_official_results_once_the_run_is_finalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    iid = "octo__widget-1"
    run = make_run(tmp_path / "run", [iid])
    sealed = sl.make(tmp_path / "sealed")
    (sealed / "gold" / iid).mkdir()
    seen: list[tuple[str, list[str]]] = []

    def evaluate(base: Path, patch: str, out: Path, split: str, workers: int, ids: list[str],
                 remove: bool) -> Path:  # fmt: skip
        seen.append((patch, ids))
        out.mkdir(exist_ok=True)
        (out / "results.json").write_text(json.dumps({"success": 1, "success_ids": ids}))
        return out / "results.json"

    monkeypatch.setattr(gr, "evaluate", evaluate)
    got = sl.unseal(tmp_path / "eval", run, sealed, "lite", 2)
    assert seen == [(str(run / "preds.json"), [iid]), ("gold", [iid])]
    assert got == {"success": 1, "gold": {"success": 1}}
    assert json.loads((run / "results.json").read_text())["success_ids"] == [iid]
    assert (run / "gold-results.json").is_file()


def test_the_runner_grades_each_final_instance_sealed_and_leaves_the_image_to_it(
    env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graded: list[tuple[str, bool]] = []

    def grade_final(base: Path, run_dir: Path, sealed: Path, iid: str, split: str, **kw: Any):
        assert rn.instance_status(run_dir, iid) == "done"  # final BEFORE it is graded
        graded.append((iid, kw["gold"]))
        return {"instance_id": iid}

    monkeypatch.setattr(sl, "grade_final", grade_final)
    e = env()
    (e.bundle / "FREEZE.sha256").write_text("d1g35t\n")
    sealed = tmp_path / "sealed-out"
    args = ("--sealed", str(sealed), "--evaluator", str(tmp_path / "ev"), "--gold-check")
    assert rn.main(e.argv(*args, "--split", "full")) == 0
    assert graded == [(SENTINELS["instance_id"], True)]
    assert not [c for c in e.calls() if c[0] == "rmi"]  # the evaluation deletes the image
    assert "resolved" not in capsys.readouterr().out.lower()
    meta = json.loads((e.run_dir / "run.json").read_text())
    assert meta["settings"]["bundle_digest"] == "d1g35t"
    with pytest.raises(SystemExit, match="--evaluator"):
        rn.main(e.argv("--sealed", str(sealed)))


# ── the self-audit ───────────────────────────────────────────────────────────────────────────


def violations(run: Path) -> list[str]:
    report = sub.audit(run, {ROW["instance_id"]: ROW})
    found = report["instances"].get(ROW["instance_id"], {"violations": []})["violations"]
    return report["run"] + found


def test_a_clean_run_passes_the_audit(tmp_path: Path) -> None:
    run = make_run(tmp_path / "run", [ROW["instance_id"]])
    report = sub.audit(run, {ROW["instance_id"]: ROW})
    assert report["n_violations"] == 0 and report["instances"] == {} and report["n_instances"] == 1


def test_the_audit_finds_what_the_rules_forbid(tmp_path: Path) -> None:
    iid = ROW["instance_id"]
    prompt = TEMPLATE.format(problem_statement=ROW["problem_statement"])
    cases: dict[str, dict[str, Any]] = {
        "the instance id is in a prompt": {"calls": [call(1, prompt + f"\nwork on {iid}", "ok")]},
        "a line of `hints_text` is in a prompt": {
            "calls": [call(1, prompt + "\n" + ROW["hints_text"], "ok")]
        },
        "a line of `test_patch` is in a prompt": {
            "calls": [call(1, prompt, "ok"), call(2, prompt + "\n" + TEST_LINE, "ok")]
        },
        "a line of `patch` is in a prompt": {
            "calls": [
                call(1, prompt + "\n    return frobnicate(value) if value else []  # the fix", "")
            ]
        },
        "the first prompt is not the prompt template": {"calls": [call(1, "Solve widget.", "ok")]},
        "attempted network access": {"log": "Web search failed for 'x': DNSError\n"},
        "looks like a credential": {"log": "export KEY=sk-abcdefghijklmnopqrstuvwxyz012345\n"},
        "network self-test": {"record": {"network_selftest": {"proxy": True, "internet": ["x"]}}},
        "not final": {"record": {"status": "infra_error"}},
        "run.json is not finalized": {"finalized": False},
    }
    for n, (needle, kw) in enumerate(cases.items()):
        found = violations(make_run(tmp_path / f"run{n}", [iid], **kw))
        assert any(needle in v for v in found), (needle, found)
        assert len(found) == 1, (needle, found)


def test_the_repository_as_it_was_is_not_a_leak_of_the_reference_patch(tmp_path: Path) -> None:
    """Found on the first dry test: a diff also holds unchanged and removed lines, which are
    the repository's own text and are in every prompt that shows the file."""
    iid = ROW["instance_id"]
    existing = "    log.debug('widget called with {0!s} and {1!s}'.format(value, mode))"
    moved = "    result = normalise_the_value(value, strict=True, mode=mode)"
    row = {
        **ROW,
        "patch": "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1,3 +1,3 @@\n"
        f" {existing}\n-{moved}\n+{moved}\n"
        "+    return frobnicate(result) if result else []  # fix\n",
    }
    prompt = TEMPLATE.format(problem_statement=ROW["problem_statement"])
    shows_file = call(1, f"{prompt}\n=== FILE: w.py ===\n{existing}\n{moved}\n", "ok")
    run = make_run(tmp_path / "run", [iid], calls=[shows_file])
    assert sub.audit(run, {iid: row})["n_violations"] == 0
    leaked = call(1, f"{prompt}\n    return frobnicate(result) if result else []  # fix", "ok")
    run2 = make_run(tmp_path / "run2", [iid], calls=[leaked])
    assert sub.audit(run2, {iid: row})["n_violations"] == 1


def test_what_the_agent_wrote_itself_is_not_a_leak_when_it_comes_back(tmp_path: Path) -> None:
    """A resolved task: the agent writes the same line as the reference patch, and its own
    edit is in the next prompt.  That is not the dataset leaking."""
    iid = ROW["instance_id"]
    prompt = TEMPLATE.format(problem_statement=ROW["problem_statement"])
    fix = "    return frobnicate(value) if value else []  # the fix"
    calls = [
        call(1, prompt, f"edit:\n{fix}"),
        call(2, prompt + f"\n=== FILE: w.py ===\n{fix}", "ok"),
    ]
    run = make_run(tmp_path / "run", [iid], calls=calls, log="curl: (6) Could not resolve host\n")
    report = sub.audit(run, {iid: ROW})
    assert report["n_violations"] == 0
    assert report["n_review"] == 1  # a command wanted the network: listed, not a violation


# ── shards and the folder ────────────────────────────────────────────────────────────────────


def test_shards_of_one_run_are_joined_and_anything_else_is_refused(tmp_path: Path) -> None:
    a = make_run(tmp_path / "a", ["o__r-1", "o__r-2"])
    b = make_run(tmp_path / "b", ["o__r-3"])
    (a / "attempts.jsonl").write_text('{"instance_id": "o__r-1", "status": "infra_error"}\n')
    doc = sub.merge([a, b], tmp_path / "all")
    assert doc["instance_ids"] == ["o__r-1", "o__r-2", "o__r-3"] and doc["finalized"] is True
    assert sorted(json.loads((tmp_path / "all" / "preds.json").read_text())) == doc["instance_ids"]
    assert (tmp_path / "all" / "instances" / "o__r-3" / "llm.jsonl").is_file()
    assert "o__r-1" in (tmp_path / "all" / "attempts.jsonl").read_text()
    other = make_run(tmp_path / "c", ["o__r-4"], settings={**SETTINGS, "model": "another"})
    with pytest.raises(SystemExit, match="different settings"):
        sub.merge([a, other], tmp_path / "x")
    twice = make_run(tmp_path / "d", ["o__r-2"])
    with pytest.raises(SystemExit, match="rolled out twice"):
        sub.merge([a, twice], tmp_path / "y")
    open_run = make_run(tmp_path / "e", ["o__r-5"], finalized=False)
    with pytest.raises(SystemExit, match="finalized"):
        sub.merge([a, open_run], tmp_path / "z")


def test_the_folder_is_built_only_from_a_clean_graded_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    iid = ROW["instance_id"]
    rows = {iid: ROW}
    dirty = make_run(tmp_path / "dirty", [iid], log="Web search failed for 'x'\n")
    assert sub.build(dirty, rows, tmp_path / "out0", "memfun", "Model") == 3
    assert not (tmp_path / "out0").exists() and "NOT BUILT" in capsys.readouterr().out
    run = make_run(tmp_path / "run", [iid])
    with pytest.raises(SystemExit, match="no results"):
        sub.build(run, rows, tmp_path / "out1", "memfun", "Model")
    (run / "results.json").write_text(
        json.dumps({"submitted": 1, "success": 1, "success_ids": [iid]})
    )
    out = tmp_path / "submissions" / "lite" / "memfun" / "model"
    assert sub.build(run, rows, out, "memfun", "Model-27B") == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "audit.json", "preds.json", "results.json", "run.json", "trajs",
    ]  # fmt: skip
    assert sorted(p.name for p in (out / "trajs" / iid).iterdir()) == [
        "agent.log", "instance.json", "llm.jsonl", "patch.diff",
    ]  # fmt: skip
    text = (out / "README.md").read_text()
    assert "# memfun + Model-27B" in text and "One rollout per instance (1 instances)" in text
    assert gr.EVALUATOR_COMMIT in text and '"success": 1' in text and "0 violations" in text
