# AGENTS.md

How to work in this repo. Applies to any agent. This file holds working
practice only — what the project *is* lives in `docs/`.

## Code

- No backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, feature flags, or migrations.
- Simplest implementation that fully meets the current requirement. No speculative abstractions, config, or indirection for hypothetical futures.
- Grow in layers: get the smallest end-to-end version working, then add capabilities on top of something that already runs. Never trade a working product for unfinished complexity.
- Keep components modular with clearly separated concerns.
- Prefer established, well-maintained libraries over reimplementing common functionality. Check what's already a dependency, and check the library's actual capabilities/docs before assuming it can't do something.
- Decide architecture for the long term — don't ship a stopgap that's "meant to be replaced later."

## Communication

- Write all responses to the user in ASD-STE100 Simplified Technical English: short sentences, active voice, one idea per sentence, simple approved words.

## Working style

- Match scope to what was asked. Make routine judgment calls yourself; check in only when different readings of the request would produce materially different work. Don't quietly narrow, widen, or transform the task.
- Don't add verification steps beyond what the task needs (extra review passes, re-checks, "double-check this"). Do the work correctly the first time; verify with tests/build/lint when those exist, not with redundant re-reading.
- Delegate to a subagent only for large, genuinely independent, parallelizable work. Don't spawn one to double-check work you just did yourself.
- When you catch and fix your own mistake, just fix it — don't narrate the correction unless it changes what the user sees or decides.

## Results and provenance

- **Commit and push every code change.** An uncommitted change is undocumented: it cannot be attributed, reviewed, reverted, or tied to the result it produced. Commit when a change is coherent — not at the end of a session — with a message saying what changed and *why*, including the measured consequence when there is one. Push so the work exists somewhere other than this machine.
- Never commit data or generated output. `.gitignore` covers them by DIRECTORY rule (`/data/`, `/raw_data/`, `/runs/`, `/inference/`, `/eval_results/`, …). Do not append individual file paths — they go stale as soon as a run is renamed, and they hide the fact that a whole tree is untracked. If something large is not covered, add the directory.
- Analysis output goes in `runs/campaigns/<NN>-<name>/` — one campaign per question, never a new ad-hoc directory. Each campaign carries a `README.md` (question, exact command, checkpoint and settings, headline numbers, caveats, vault note) and a line in `runs/campaigns/INDEX.md`.
- Every result must be traceable to committed code. Commit the script that produced it.
- A measurement is only as good as its validation on known-good data. State what you validated against, and what the method's failure modes are.

## Where things are

- `docs/CODEMAP.md` — maps every module, script, config family and data/run directory. Read it first to locate code, and update it whenever you add or remove a file.
- `docs/ARCHITECTURE.md` — config system, data pipeline, decoder output contract, run structure, key invariants.
- `docs/DEVELOPMENT.md` — install, tests (including known pre-existing failures), training commands, GPU deployment target.
- `/home/jorgecabrejas/Dev/PhDTracker` — the vault: decisions log, roadmap, experiment history, research notes, publication drafts. Always the source of truth for *why*.
