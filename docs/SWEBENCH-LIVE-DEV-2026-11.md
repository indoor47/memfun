# SWE-bench-Live, known-solvable dev tasks: memfun next to mini-swe-agent

Run on 2026-11-10 with the runner in `evals/swebench_live/`. Both agents used the same model (Qwen 3.8 27B
on Cerebras, thinking off), the same official task containers, the same no-internet network and the same
recording proxy. Every verdict comes from the official SWE-bench-Live evaluator.

**This is a development set, not a leaderboard number.** The tasks were chosen because older agents solved
them, so the resolve rates here are far above what a random sample gives. The purpose is diagnostic: on these
tasks a failure is the agent's fault, not the task's.

## The task set

16 instances from `full` minus `lite` minus `verified`, at most 2 per repository, each solved by at least 3 of
4 older published agents (SWE-agent and OpenHands with GPT-4.1, GPT-4o and Claude 3.7).

Each was first checked with its own gold patch on our machine. 15 pass. One is dropped:

| Instance | Why the gold patch fails |
|---|---|
| `ag2ai__faststream-1619` | All 1584 pass-to-pass tests pass. The target test compares hash-valued dictionary keys that differ between runs. |

Four gold checks first ended in an infrastructure error (an image pull or the container runtime timed out
while a 20 GB image was being pulled next to them). They passed when repeated alone. A pull failure is
reported by the evaluator as "not found in registry": repeat it before believing it.

## Arms

| Arm | Settings |
|---|---|
| memfun | Context-first solver from PR 94 (commit `80fba6a` of the work branch) with the project's own tests, a reproduction script, probes without a guard (the container is disposable), 2 probe rounds, 4 fix attempts. Budget 520K tokens per task. |
| mini-swe-agent | Version 2.4.6, its standard `swebench.yaml`, unchanged. Budget 1.45M tokens per task. |

One rollout per task. A rollout that got no answer from the model (the account ran out of credit in the
middle of the batch) was restarted from scratch; the failed attempts are kept with the runs.

## Result

| Arm | Resolved | Tokens | Cost | Mean work time per task |
|---|---|---|---|---|
| memfun | 12 of 15 | 2.40M | $2.45 | 96 s |
| mini-swe-agent | 13 of 15 | 4.83M | $4.81 | 72 s |

Each arm resolved tasks the other did not: memfun alone resolved `keras-19838` and `pylint-9346`;
mini-swe-agent alone resolved `haystack-6922`, `aiogram-1497` and `faker-2018`.

Per task (model calls, tokens, cost, work time):

| Instance | memfun | calls | tokens | cost | time | mini-swe-agent | calls | tokens | cost | time |
|---|---|---|---|---|---|---|---|---|---|---|
| `reata__sqllineage-572` | resolved | 20 | 215K | $0.23 | 106 s | resolved | 14 | 115K | $0.11 | 37 s |
| `pylint-dev__pylint-9346` | resolved | 7 | 118K | $0.12 | 39 s | unresolved | 68 | 972K | $0.97 | 108 s |
| `python-attrs__attrs-1401` | resolved | 5 | 117K | $0.12 | 42 s | resolved | 8 | 29K | $0.03 | 22 s |
| `reata__sqllineage-619` | resolved | 5 | 42K | $0.04 | 30 s | resolved | 39 | 522K | $0.52 | 136 s |
| `deepset-ai__haystack-6922` | unresolved | 13 | 388K | $0.39 | 150 s | resolved | 36 | 302K | $0.30 | 116 s |
| `keras-team__keras-19838` | resolved | 6 | 156K | $0.16 | 84 s | empty patch | 83 | 1465K | $1.46 | 266 s |
| `matplotlib__matplotlib-27833` | resolved | 6 | 81K | $0.08 | 21 s | resolved | 35 | 211K | $0.21 | 96 s |
| `aiogram__aiogram-1497` | empty patch | 8 | 114K | $0.12 | 52 s | resolved | 18 | 109K | $0.11 | 28 s |
| `sphinx-doc__sphinx-12470` | resolved | 6 | 117K | $0.12 | 52 s | resolved | 10 | 29K | $0.03 | 14 s |
| `piskvorky__smart_open-820` | resolved | 6 | 132K | $0.13 | 260 s | resolved | 14 | 97K | $0.10 | 25 s |
| `geopandas__geopandas-3416` | resolved | 6 | 111K | $0.11 | 49 s | resolved | 40 | 350K | $0.35 | 62 s |
| `joke2k__faker-2018` | unresolved | 10 | 102K | $0.12 | 36 s | resolved | 25 | 166K | $0.17 | 35 s |
| `keras-team__keras-20703` | resolved | 15 | 522K | $0.52 | 478 s | resolved | 26 | 239K | $0.24 | 50 s |
| `streamlink__streamlink-5926` | resolved | 12 | 70K | $0.07 | 30 s | resolved | 17 | 96K | $0.10 | 23 s |
| `koxudaxi__datamodel-code-generator-2185` | resolved | 6 | 119K | $0.12 | 13 s | resolved | 20 | 124K | $0.12 | 69 s |

## Where memfun went wrong

Read from each trajectory next to mini-swe-agent's successful one.

| Instance | First wrong decision |
|---|---|
| `deepset-ai__haystack-6922` | The edit equals mini-swe-agent's except for an unquoted return annotation that is undefined at that point on Python 3.8. memfun's own import check caught it in round 1 and round 2 fixed it. At the end of the loop the round-1 state was put back as "best", so the submitted patch cannot be imported. |
| `aiogram__aiogram-1497` | The task changes an output format. An existing test asserts the old format and failed; the fix loop edited the code until that test passed again, which is the original code. Empty patch. mini-swe-agent changed the code and left that test failing on purpose. |
| `joke2k__faker-2018` | The first edit changed the separator and also dropped digits from two formats. The model-written reproduction script then failed for a reason the issue never states, and three rounds rewrote all formats to satisfy it. |

`keras-team__keras-20703` was resolved by the first solution but cost 522K tokens and 478 s. Four verify
rounds followed it: the reproduction script crashed on its own wrong attribute name, and the project's linter
flagged memfun's own scratch file. Each round re-ran the keras test files.

Causes in order of damage:

1. A model-written reproduction script is accepted, and then used as a gate, when it fails for its own
   reasons (2 of the 3 misses, and the cost of `keras-20703`).
2. A state that fails the import check can be restored as the best state.
3. An existing test that encodes the behaviour the task changes is treated as a regression, and the loop can
   end on a no-op.
4. The project's linter runs over memfun's scratch files.
5. Careless first edits, which a working reproduction would have caught.

None of these is fixed in PR 94; they are its open items.

## Where mini-swe-agent went wrong

| Instance | What happened |
|---|---|
| `keras-team__keras-19838` | 83 steps, ran into its token budget without submitting. |
| `pylint-dev__pylint-9346` | 68 steps, submitted a patch; the target test still fails, nothing else broke. |

## What this does and does not show

- On tasks known to be solvable, the plain loop and memfun's context-first solver are close (13 and 12 of 15),
  and memfun used half the tokens.
- All three memfun misses come from its verification loop making a good or nearly good first answer worse,
  not from finding the wrong file: it edited the right source file on 14 of 15 tasks (the exception is the
  reverted `aiogram-1497`).
- Fifteen hand-picked tasks cannot rank the two agents. A random sample is needed for a resolve rate.
