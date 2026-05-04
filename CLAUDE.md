# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**CAMS Historical Telescope** — a single-file, zero-dependency browser app that visualises structural coordination physics across 9 societies from 10 CE to 2026. Everything lives in `index.html`: HTML structure, CSS, inline JS, and the full `DATA` JSON blob (~700 KB).

Open `index.html` directly in a browser — no build step, no server required.

## Architecture

All code is in one file with clearly delimited sections (marked with `// ──` banners):

| Section | What it does |
|---|---|
| `:root` CSS vars | Design tokens — cream paper palette, 5 font stacks |
| `DATA` | Nested JSON: `DATA[society][year]` → `{nodes, b, sk, praetorian, cog_gap, dominant, weakest, …}` |
| `NODES` / `SLOW_NODES` / `FAST_NODES` | 8 coordination nodes; split into reflective (teal) vs reactive (amber) for coloring |
| `SOCIETY_META` | Display metadata per society (era string, color) |
| Spring physics (`sim`, `stepPhysics`, `computeTargets`, `drawFrame`, `startAnim`) | rAF loop; each node is a particle with position+velocity; bonds drive spring stiffness and rest length; central Λ gravity; stress jitter; motion trails |
| Timeline (`drawTimeline`) | Canvas sparkline of S/K ratio over all years for the current society |
| Basin classification (`classifyBasin`, `BASINS`, `threatLevel`) | Derives attractor basin (7 types) and threat level from `{sk, praetorian, cog_gap, b}` |
| Dossier (`buildDossier`, `render`) | Builds the right-panel HTML: Zeitgeist, Power Topology, Survival Protocol, Trajectory |
| Controls (`onYearSlide`, `setSociety`, `toggleRoll`/`startRoll`/`stopRoll`) | Year slider, society tab bar, autoplay roll with speed selector |

## Key data shape

Each `DATA[society][year]` entry has:
- `nodes` — object keyed by node name, each `{C, K, S, A, V, b}` (Capacity, Knowledge, Stress, Abstraction, Value, bond strength)
- `b` — system-wide bond coherence (Λ); drives central gravity in the physics sim
- `sk` — S/K ratio (stress/capacity); main health metric; >1.4 = high threat
- `praetorian` — Shield minus Helm node values; positive = military over-reach
- `cog_gap` — slow-loop minus fast-loop aggregate; negative = cognitive capture risk
- `dominant` / `weakest` — node names

## Design system

Fonts: `--font-d` Fraunces (display/body), `--font-s` Source Serif 4 (dossier prose), `--font-m` JetBrains Mono (labels/values), `--font-ui` Inter (UI chrome), `--font-t` Special Elite (typewriter accents).

Palette anchors: `--bg` `#F5F1E4` cream paper, `--amber` `#D98B2B` primary accent, `--teal` `#2F6F7D` safe/positive, `--red` `#C84C3E` danger, `--text` `#1B2A33` ink.

## Extending

**Add a society** — add an entry to `DATA` and `SOCIETY_META`; the society bar auto-populates.

**Add a year** — add an entry under the society key in `DATA`; the slider range updates automatically.

**Tune physics** — adjust `stiffness`, `damping`, `gravStr`, and `jitter` constants inside `stepPhysics`.

**Add a basin type** — add to `BASINS`, extend `classifyBasin`'s if-chain, add entries to `zeitgeist_map` and `survival_map` inside `buildDossier`.

**Change roll speeds** — edit the `<option value="…">` ms values in the `#roll-speed` select.

## Notes (wintermute repo)

In this repo the telescope lives in `test/` as `telescope_1.html` (clean base) and `telescope_2.html` (base + 5 cam5 societies injected: Argentina, Germany, China, Thailand, Sweden). The CLAUDE.md references `index.html` generically — treat that as `telescope_1.html`.
