# Hotfix — generic-query resolver + helpful error_message

**Trigger**: Yusen live REPL 2026-04-19 — `抓个东西` failed with `Cannot locate target object`, VGG Layer-3 re-plan injected `scan_and_detect` → `No perception backend`.

## Debug

See `.sdd/DEBUG.md` — Hypothesis loop OBSERVE → HYPOTHESIZE → EXPERIMENT → CONCLUDE.

## Root cause

Two orthogonal gaps:
1. Resolver returned `None` for "抓个东西" because no existing fallback covered ambiguous multi-pickable + generic-word queries.
2. `error_message="Cannot locate target object"` didn't expose known pickables, so the re-plan LLM couldn't retry with a valid label and defaulted to perception.

## Fix (2 files)

- `vector_os_nano/skills/pick_top_down.py` — generic-word helper + step-6 fallback + richer error_message
- `vector_os_nano/skills/mobile_pick.py` — richer error_message (mirror)

## Tests (4 added)

- Generic CN token → first pickable
- Generic EN token → first pickable
- `红色的东西` (color + generic) → color path wins
- Wrong label → error_message contains known labels inline

## Verify

- 109/109 tests green, ruff clean
- Live resolver smoke: 7 queries all produce expected output
