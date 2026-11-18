import { Link } from "@tanstack/react-router";
import { useEffect, useReducer, useRef, useState } from "react";
import { Button } from "@/components/button";
import { Instruments } from "@/components/instruments";
import { Ring } from "@/components/ring";
import { SeriesChart, type ChartPoint } from "@/components/series-chart";
import { cn } from "@/lib/cn";
import {
  ARCHETYPES,
  BEST_KEY,
  CALLS_PER_ROUND,
  GLYPHS,
  LEAD_WINDOW,
  bandLabel,
  caseSeries,
  formatSigned,
  identityOptions,
  judgeCall,
  loadCases,
  passOnsets,
  pickSession,
  reading,
  scoreRound,
  type ArchetypeId,
  type CaseRound,
  type CrisisCall,
  type RoundScore,
  type Sample,
  type YearMetrics,
} from "@/lib/cams/model";

type Phase = "title" | "intro" | "stream" | "paused" | "unblind" | "reveal" | "report";

type State = {
  phase: Phase;
  rounds: CaseRound[];
  roundIdx: number;
  cursor: number;
  calls: CrisisCall[];
  resolved: number[];
  passed: number[];
  options: string[];
  identity: string | null;
  archetype: ArchetypeId | null;
  results: RoundScore[];
  best: number;
  speed: 1 | 2 | 4;
  flash: CrisisCall | null;
  metrics: YearMetrics[];
  samples: Sample[];
  loading: boolean;
  error: string | null;
  cases: CaseRound[];
};

const initial: State = {
  phase: "title",
  rounds: [],
  roundIdx: 0,
  cursor: 0,
  calls: [],
  resolved: [],
  passed: [],
  options: [],
  identity: null,
  archetype: null,
  results: [],
  best: 0,
  speed: 1,
  flash: null,
  metrics: [],
  samples: [],
  loading: true,
  error: null,
  cases: [],
};

type Action =
  | { type: "loaded"; cases: CaseRound[] }
  | { type: "failed"; message: string }
  | { type: "hydrateBest"; best: number }
  | { type: "start" }
  | { type: "begin" }
  | { type: "tick"; dt: number }
  | { type: "toggle" }
  | { type: "scrub"; index: number }
  | { type: "speed"; speed: 1 | 2 | 4 }
  | { type: "call" }
  | { type: "clearFlash" }
  | { type: "unblind" }
  | { type: "identity"; name: string }
  | { type: "archetype"; id: ArchetypeId }
  | { type: "next" }
  | { type: "quit" };

function openRound(round: CaseRound, session: CaseRound[]) {
  const series = caseSeries(round);
  return {
    cursor: 0,
    calls: [] as CrisisCall[],
    resolved: [] as number[],
    passed: [] as number[],
    options: identityOptions(round, session),
    identity: null,
    archetype: null,
    flash: null,
    metrics: series.metrics,
    samples: series.samples,
    phase: "intro" as Phase,
  };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "loaded":
      return { ...state, loading: false, error: null, cases: action.cases };
    case "failed":
      return { ...state, loading: false, error: action.message };
    case "hydrateBest":
      return { ...state, best: Math.max(state.best, action.best) };
    case "start": {
      const rounds = pickSession(state.cases);
      const round = rounds[0];
      if (!round) return state;
      return { ...state, rounds, roundIdx: 0, results: [], ...openRound(round, rounds) };
    }
    case "begin":
      if (state.phase !== "intro" && state.phase !== "paused") return state;
      return { ...state, phase: "stream" };
    case "toggle":
      if (state.phase === "stream") return { ...state, phase: "paused" };
      if (state.phase === "paused" || state.phase === "intro") return { ...state, phase: "stream" };
      return state;
    case "speed":
      return { ...state, speed: action.speed };
    case "tick": {
      if (state.phase !== "stream") return state;
      const round = state.rounds[state.roundIdx];
      if (!round) return state;
      const max = round.years.length - 1;
      const prev = state.cursor;
      const cursor = Math.min(max, prev + action.dt * 0.85 * state.speed);
      const prevI = Math.floor(prev + 1e-6);
      const nextI = Math.floor(cursor + 1e-6);
      const passed = nextI > prevI ? passOnsets(round, prevI, nextI, state.resolved, state.passed) : state.passed;
      return {
        ...state,
        cursor,
        passed,
        phase: cursor >= max - 1e-6 ? "paused" : state.phase,
      };
    }
    case "scrub": {
      if (state.phase !== "stream" && state.phase !== "paused") return state;
      const round = state.rounds[state.roundIdx];
      if (!round) return state;
      const max = round.years.length - 1;
      const cursor = Math.min(max, Math.max(0, action.index));
      const prevI = Math.floor(state.cursor + 1e-6);
      const nextI = Math.floor(cursor + 1e-6);
      const passed = nextI > prevI ? passOnsets(round, prevI, nextI, state.resolved, state.passed) : state.passed;
      return { ...state, cursor, passed };
    }
    case "call": {
      if ((state.phase !== "stream" && state.phase !== "paused") || state.calls.length >= CALLS_PER_ROUND) {
        return state;
      }
      const round = state.rounds[state.roundIdx];
      if (!round) return state;
      const index = Math.floor(state.cursor + 1e-6);
      const call = judgeCall(round, index, state.resolved);
      const resolved =
        call.onsetIndex === null ? state.resolved : [...state.resolved, call.onsetIndex];
      return { ...state, calls: [...state.calls, call], resolved, flash: call };
    }
    case "clearFlash":
      return { ...state, flash: null };
    case "unblind": {
      if (state.phase !== "stream" && state.phase !== "paused") return state;
      const round = state.rounds[state.roundIdx];
      if (!round) return state;
      const index = Math.floor(state.cursor + 1e-6);
      return {
        ...state,
        phase: "unblind",
        passed: passOnsets(round, index, round.years.length - 1, state.resolved, state.passed),
      };
    }
    case "identity":
      if (state.phase !== "unblind" || state.identity) return state;
      return { ...state, identity: action.name };
    case "archetype": {
      if (state.phase !== "unblind" || !state.identity || state.archetype) return state;
      const round = state.rounds[state.roundIdx];
      if (!round) return state;
      const scored = scoreRound(round, state.calls, state.resolved, state.identity, action.id);
      return { ...state, archetype: action.id, results: [...state.results, scored], phase: "reveal" };
    }
    case "next": {
      if (state.phase !== "reveal") return state;
      const nextIdx = state.roundIdx + 1;
      const round = state.rounds[nextIdx];
      if (!round) {
        const total = state.results.reduce((sum, item) => sum + item.total, 0);
        return { ...state, phase: "report", best: Math.max(state.best, total) };
      }
      return { ...state, roundIdx: nextIdx, ...openRound(round, state.rounds) };
    }
    case "quit":
      return { ...state, phase: "title", flash: null };
    default:
      return state;
  }
}

function callCopy(call: CrisisCall): string {
  if (call.kind === "hit") {
    const unit = call.lead === 1 ? "year" : "years";
    return `Hit · ${call.lead} ${unit} early · ${formatSigned(call.points, 0)}`;
  }
  if (call.kind === "late") return `Late · ${formatSigned(call.points, 0)}`;
  return `False call · ${formatSigned(call.points, 0)}`;
}

export function ProphetView() {
  const [state, dispatch] = useReducer(reducer, initial);
  const [clock, setClock] = useState(0);
  const [scoreReady, setScoreReady] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    let live = true;
    loadCases()
      .then((cases) => {
        if (live) dispatch({ type: "loaded", cases });
      })
      .catch((err: unknown) => {
        if (live) dispatch({ type: "failed", message: err instanceof Error ? err.message : "Blind set failed." });
      });
    try {
      const stored = window.localStorage.getItem(BEST_KEY);
      if (stored) dispatch({ type: "hydrateBest", best: Number(stored) || 0 });
    } catch {
      /* ignore private mode */
    }
    setScoreReady(true);
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!scoreReady) return;
    try {
      window.localStorage.setItem(BEST_KEY, String(state.best));
    } catch {
      /* ignore private mode */
    }
  }, [scoreReady, state.best]);

  useEffect(() => {
    if (state.phase !== "title" || state.loading) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = window.setInterval(() => setClock((value) => value + 1), 1100);
    return () => window.clearInterval(id);
  }, [state.phase, state.loading]);

  useEffect(() => {
    if (!state.flash) return;
    const id = window.setTimeout(() => dispatch({ type: "clearFlash" }), 2200);
    return () => window.clearTimeout(id);
  }, [state.flash]);

  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    let acc = 0;
    const loop = (now: number) => {
      acc += now - last;
      last = now;
      if (acc >= 80) {
        const dt = Math.min(0.2, acc / 1000);
        acc = 0;
        if (stateRef.current.phase === "stream") dispatch({ type: "tick", dt });
      }
      frame = window.requestAnimationFrame(loop);
    };
    frame = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA")) {
        return;
      }
      const phase = stateRef.current.phase;
      if (event.code === "Space") {
        event.preventDefault();
        if (phase === "title") dispatch({ type: "start" });
        else if (phase === "intro" || phase === "stream" || phase === "paused") dispatch({ type: "toggle" });
      } else if ((event.key === "c" || event.key === "C") && (phase === "stream" || phase === "paused")) {
        dispatch({ type: "call" });
      } else if (event.key === "Escape" && phase === "stream") {
        dispatch({ type: "toggle" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const round = state.rounds[state.roundIdx];
  const index = Math.min(Math.floor(state.cursor + 1e-6), Math.max(0, state.metrics.length - 1));
  const metrics = state.metrics[index];
  const sample = state.samples[index];
  const atEnd = !!round && index >= round.years.length - 1 && (state.phase === "stream" || state.phase === "paused");
  const sessionPoints = state.results.reduce((sum, item) => sum + item.total, 0);
  const revealed = state.phase === "reveal" || state.phase === "report";

  const demo = state.cases[0];
  const demoSeries = demo ? caseSeries(demo) : null;
  const demoFrame = demoSeries ? clock % demoSeries.samples.length : 0;

  if (state.phase === "title" || state.loading || state.error) {
    const ghostMetrics = demoSeries?.metrics[demoFrame];
    const ghostSample = demoSeries?.samples[demoFrame];
    return (
      <div className="mx-auto grid max-w-5xl gap-8 px-4 py-8 md:grid-cols-[minmax(0,1fr)_20rem] md:py-14">
        <div>
          <p className="font-mono text-xs tracking-widest text-brass">CAMS · BLIND SET</p>
          <h1 className="mt-3 font-display text-5xl leading-none text-balance md:text-7xl">Blind Prophet</h1>
          <p className="mt-5 max-w-xl text-pretty text-lg text-mute">
            The calendar is covered. Eight nodes stay visible. You get two calls. Name the society afterwards, and
            separately the shape of what it did. A window with no onset, left uncalled, is its own kind of correct.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button disabled={state.loading || !!state.error} onClick={() => dispatch({ type: "start" })}>
              {state.loading ? "Opening the set…" : "Begin — 3 rounds"}
            </Button>
            <p className="font-mono text-sm text-mute">
              Best <span className="text-bone tabular-nums">{state.best}</span>
            </p>
          </div>
          {state.error ? <p className="mt-4 text-danger">{state.error}</p> : null}
          <dl className="mt-8 grid max-w-xl grid-cols-2 gap-x-6 gap-y-3 text-sm text-mute">
            <div>
              <dt className="text-bone">Two calls</dt>
              <dd>A hit inside {LEAD_WINDOW} years scores more the earlier you are. Late is +40. A false call is −50.</dd>
            </div>
            <div>
              <dt className="text-bone">Then two names</dt>
              <dd>The society is +100. The archetype is another +100. They are different skills.</dd>
            </div>
          </dl>
          <p className="mt-6 text-sm text-faint">Space starts. C calls. Esc pauses.</p>
        </div>
        <div className="opacity-80">
          {ghostMetrics && ghostSample ? (
            <Ring
              ghost
              metrics={ghostMetrics}
              nodes={ghostSample.nodes}
              yearLabel=""
              selected={null}
            />
          ) : (
            <div className="h-80 border border-line" />
          )}
        </div>
      </div>
    );
  }

  if (state.phase === "report") {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 md:py-12">
        <p className="font-mono text-xs tracking-widest text-brass">SESSION REPORT</p>
        <h1 className="mt-2 font-display text-4xl md:text-6xl">Blind set complete</h1>
        <p className="mt-3 font-mono text-2xl tabular-nums text-brass">{sessionPoints} pts</p>
        <p className="mt-1 text-sm text-mute">Best {state.best}. Naming the society and naming the shape are scored apart.</p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[36rem] text-left text-sm">
            <thead className="text-xs tracking-wide text-faint">
              <tr>
                <th className="py-2 font-medium">Round</th>
                <th className="py-2 font-medium">Truth</th>
                <th className="py-2 font-medium">Calls</th>
                <th className="py-2 font-medium">Name</th>
                <th className="py-2 font-medium">Shape</th>
                <th className="py-2 text-right font-medium">Pts</th>
              </tr>
            </thead>
            <tbody>
              {state.results.map((result, resultIndex) => (
                <tr key={result.round.id} className="border-t border-line">
                  <td className="py-3 font-mono">{GLYPHS[resultIndex]}</td>
                  <td className="py-3">{result.round.truth}</td>
                  <td className="py-3 font-mono tabular-nums">
                    {formatSigned(result.callPoints, 0)}
                    {result.missed > 0 ? <span className="text-danger"> · missed {result.missed}</span> : null}
                  </td>
                  <td className={cn("py-3", result.identityPoints ? "text-good" : "text-danger")}>
                    {result.identityPoints ? "Yes" : result.identityGuess}
                  </td>
                  <td className={cn("py-3", result.archetypePoints ? "text-good" : "text-danger")}>
                    {result.archetypePoints ? "Yes" : "No"}
                  </td>
                  <td className="py-3 text-right font-mono tabular-nums">{result.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-6 flex flex-wrap gap-2">
          <Button onClick={() => dispatch({ type: "start" })}>New session</Button>
          <Button tone="ghost" onClick={() => dispatch({ type: "quit" })}>
            Title
          </Button>
        </div>
      </div>
    );
  }

  if (!round || !metrics || !sample) return null;

  const points: ChartPoint[] = state.metrics.slice(0, revealed ? state.metrics.length : index + 1).map((item, itemIndex) => ({
    x: revealed ? (state.samples[itemIndex]?.y ?? itemIndex) : itemIndex,
    y: item.vMean,
    compare: null,
    crisis: item.crisis,
  }));
  const yearLabel = revealed ? String(sample.y) : `T+${index}`;
  const result = state.results[state.results.length - 1];
  const showResult = state.phase === "reveal" && result?.round.id === round.id ? result : null;

  return (
    <div className="lg:h-full lg:overflow-y-auto">
      <div className="mx-auto flex max-w-5xl flex-col gap-5 px-4 py-5 md:px-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="font-mono text-xs tracking-widest text-brass">
              SOCIETY {GLYPHS[state.roundIdx] ?? "Ω"} · ROUND {state.roundIdx + 1}/3
            </p>
            <h1 className="mt-1 font-display text-4xl leading-none md:text-5xl">
              {state.phase === "reveal" ? round.truth : "Name withheld"}
            </h1>
            <p className="mt-2 text-sm text-mute">
              {state.phase === "reveal"
                ? `${round.years[0]?.y}–${round.years[round.years.length - 1]?.y} · ${bandLabel(metrics)}`
                : `${round.years.length} sampled years · era masked`}
            </p>
          </div>
          <p className="font-mono text-sm text-mute">
            Session <span className="text-bone tabular-nums">{sessionPoints}</span>
          </p>
        </div>

        {state.phase === "intro" ? (
          <p className="max-w-2xl text-pretty text-mute">
            Read the opening state. When you start, the window runs forward and the years stay hidden. Call only if you
            mean it — scrubbing past an onset without a call spends it.
          </p>
        ) : null}

        {showResult ? (
          <div className="border border-line bg-raise px-4 py-4">
            <p className="max-w-2xl text-pretty">{round.blurb}</p>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
              <div>
                <dt className="text-faint">Calls</dt>
                <dd className="font-mono tabular-nums">{formatSigned(showResult.callPoints, 0)}</dd>
              </div>
              <div>
                <dt className="text-faint">Society</dt>
                <dd className={showResult.identityPoints ? "text-good" : "text-danger"}>
                  {showResult.identityPoints ? "+100" : showResult.identityGuess}
                </dd>
              </div>
              <div>
                <dt className="text-faint">Shape</dt>
                <dd className={showResult.archetypePoints ? "text-good" : "text-danger"}>
                  {showResult.archetypePoints ? "+100" : "Missed"}
                </dd>
              </div>
              <div>
                <dt className="text-faint">Round</dt>
                <dd className="font-mono tabular-nums text-brass">{showResult.total}</dd>
              </div>
            </dl>
            {showResult.discipline ? <p className="mt-2 text-sm text-good">Discipline bonus +120</p> : null}
            {showResult.missed > 0 ? (
              <p className="mt-2 text-sm text-danger">{showResult.missed} onset{showResult.missed === 1 ? "" : "s"} missed.</p>
            ) : null}
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-3">
          <div className="font-display text-4xl leading-none tabular-nums">{yearLabel}</div>
          <div className="flex gap-1" role="group" aria-label="Speed">
            {([1, 2, 4] as const).map((speed) => (
              <button
                key={speed}
                type="button"
                onClick={() => dispatch({ type: "speed", speed })}
                className={cn(
                  "h-11 px-3 font-mono text-sm",
                  state.speed === speed ? "text-brass" : "text-mute hover:text-bone",
                )}
              >
                {speed}×
              </button>
            ))}
          </div>
          <p className="font-mono text-xs text-faint" aria-label="Calls remaining">
            Calls {CALLS_PER_ROUND - state.calls.length}
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {state.flash ? (
            <p
              className={cn(
                "w-full font-mono text-sm",
                state.flash.kind === "false" ? "text-danger" : state.flash.kind === "late" ? "text-brass" : "text-good",
              )}
              aria-live="polite"
            >
              {callCopy(state.flash)}
            </p>
          ) : null}
          {state.phase === "intro" ? (
            <Button onClick={() => dispatch({ type: "begin" })}>Start the stream</Button>
          ) : null}
          {state.phase === "stream" || state.phase === "paused" ? (
            <Button tone="ghost" onClick={() => dispatch({ type: "toggle" })}>
              {state.phase === "stream" ? "Pause" : "Resume"}
            </Button>
          ) : null}
          {state.phase === "intro" || state.phase === "stream" || state.phase === "paused" ? (
            <>
              <Button
                tone="danger"
                disabled={state.phase === "intro" || state.calls.length >= CALLS_PER_ROUND}
                onClick={() => dispatch({ type: "call" })}
              >
                Call crisis
              </Button>
              <Button tone="ghost" disabled={state.phase === "intro"} onClick={() => dispatch({ type: "unblind" })}>
                {atEnd ? "Close the window" : "Close early"}
              </Button>
              <Button tone="quiet" onClick={() => dispatch({ type: "quit" })}>
                Quit
              </Button>
            </>
          ) : null}
        </div>

        <SeriesChart points={points} cursor={revealed ? sample.y : index} />
        {state.phase === "stream" || state.phase === "paused" ? (
          <label>
            <span className="sr-only">Scrub the masked window</span>
            <input
              type="range"
              min={0}
              max={round.years.length - 1}
              value={index}
              onChange={(event) => dispatch({ type: "scrub", index: Number(event.target.value) })}
            />
          </label>
        ) : null}

        <Instruments metrics={metrics} />
        <p className="max-w-2xl text-pretty">{reading(metrics)}</p>
        {metrics.pathologies.length > 0 ? (
          <ul className="flex flex-wrap gap-2">
            {metrics.pathologies.map((item) => (
              <li key={item} className="border border-danger px-2 py-1 text-xs text-danger">
                {item}
              </li>
            ))}
          </ul>
        ) : null}

        <div className="max-w-sm">
          <Ring metrics={metrics} nodes={sample.nodes} yearLabel={yearLabel} selected={null} />
        </div>

        <div>
          {state.phase === "unblind" && !state.identity ? (
            <div>
              <p className="font-mono text-xs tracking-widest text-brass">THE WINDOW HAS CLOSED</p>
              <h2 className="mt-1 font-display text-3xl">Which society was this?</h2>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {state.options.map((name) => (
                  <Button key={name} tone="ghost" onClick={() => dispatch({ type: "identity", name })}>
                    {name}
                  </Button>
                ))}
              </div>
            </div>
          ) : null}

          {state.phase === "unblind" && state.identity ? (
            <div>
              <h2 className="font-display text-3xl">And its shape?</h2>
              <div className="mt-3 grid gap-2">
                {ARCHETYPES.map((item) => (
                  <Button key={item.id} tone="ghost" className="h-auto py-3 text-left" onClick={() => dispatch({ type: "archetype", id: item.id })}>
                    <span>
                      <span className="block">{item.name}</span>
                      <span className="block text-xs font-normal text-mute">{item.blurb}</span>
                    </span>
                  </Button>
                ))}
              </div>
            </div>
          ) : null}

          {state.phase === "reveal" ? (
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => dispatch({ type: "next" })}>
                {state.roundIdx + 1 < state.rounds.length ? "Next round" : "Session report"}
              </Button>
              <Link
                to="/"
                search={{ society: round.truth, year: round.onsets[0] ?? round.years[0]?.y }}
                className="inline-flex h-11 items-center px-3 text-sm text-brass"
              >
                Open in the atlas
              </Link>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
