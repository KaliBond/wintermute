import { useEffect, useMemo, useState } from "react";
import { Pause, Play } from "lucide-react";
import { Button } from "@/components/button";
import { Instruments } from "@/components/instruments";
import { Ring } from "@/components/ring";
import { SeriesChart, type ChartPoint } from "@/components/series-chart";
import { cn } from "@/lib/cn";
import {
  NODES,
  bandLabel,
  findCase,
  formatSigned,
  loadCases,
  loadPanel,
  nearestIndex,
  notables,
  reading,
  rowVMean,
  sharpestDrop,
  societySeries,
  topBonds,
  type CaseRound,
  type PanelSociety,
} from "@/lib/cams/model";

type SortKey = "name" | "strained" | "span";

function bandClass(label: string): string {
  if (label === "Freeze" || label === "Systemic stress") return "text-danger";
  if (label === "High coherence") return "text-good";
  return "text-mute";
}

export function AtlasView({ society, year }: { society?: string; year?: number }) {
  const [panel, setPanel] = useState<PanelSociety[] | null>(null);
  const [cases, setCases] = useState<CaseRound[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("name");
  const [selected, setSelected] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [node, setNode] = useState(0);
  const [compare, setCompare] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [listOpen, setListOpen] = useState(false);

  useEffect(() => {
    let live = true;
    loadPanel()
      .then((rows) => {
        if (live) setPanel(rows);
      })
      .catch((err: unknown) => {
        if (live) setError(err instanceof Error ? err.message : "The panel failed to load.");
      });
    loadCases()
      .then((rows) => {
        if (live) setCases(rows);
      })
      .catch(() => {
        if (live) setCases([]);
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!panel) return;
    const wanted = society
      ? panel.findIndex((item) => item.name.toLowerCase() === society.toLowerCase())
      : panel.findIndex((item) => item.name === "Germany");
    const index = wanted >= 0 ? wanted : 0;
    const series = societySeries(panel[index] as PanelSociety);
    setSelected(index);
    setCompare(null);
    setNode(0);
    setPlaying(false);
    if (year !== undefined) setCursor(nearestIndex(series.samples, year));
    else setCursor(sharpestDrop(series.metrics));
  }, [panel, society, year]);

  const current = panel?.[selected] ?? null;

  useEffect(() => {
    if (!playing || !current) return;
    const id = window.setInterval(() => {
      setCursor((value) => (value >= current.years.length - 1 ? 0 : value + 1));
    }, 650);
    return () => window.clearInterval(id);
  }, [playing, current]);

  const ranked = useMemo(() => {
    if (!panel) return [];
    const needle = query.trim().toLowerCase();
    const rows = panel
      .map((item, index) => {
        const last = item.years[item.years.length - 1];
        const first = item.years[0];
        return {
          index,
          name: item.name,
          now: last ? rowVMean(last) : 0,
          span: (last?.[0] ?? 0) - (first?.[0] ?? 0),
          from: first?.[0] ?? 0,
          to: last?.[0] ?? 0,
        };
      })
      .filter((item) => item.name.toLowerCase().includes(needle));
    rows.sort((a, b) => {
      if (sort === "strained") return a.now - b.now;
      if (sort === "span") return b.span - a.span;
      return a.name.localeCompare(b.name);
    });
    return rows;
  }, [panel, query, sort]);

  if (error) {
    return (
      <div className="mx-auto max-w-lg px-4 py-16">
        <p className="font-mono text-xs tracking-widest text-brass">PANEL</p>
        <h1 className="mt-2 font-display text-4xl">The panel failed to load.</h1>
        <p className="mt-3 text-mute">{error}</p>
        <Button className="mt-6" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </div>
    );
  }

  if (!panel || !current) {
    return (
      <div className="px-4 py-16">
        <p className="font-mono text-xs tracking-widest text-mute">Opening the JUNO panel…</p>
      </div>
    );
  }

  const series = societySeries(current);
  const safeCursor = Math.min(cursor, series.samples.length - 1);
  const sample = series.samples[safeCursor];
  const metrics = series.metrics[safeCursor];
  if (!sample || !metrics) return null;

  const compareSociety = compare === null ? null : (panel[compare] ?? null);
  const compareSeries = compareSociety ? societySeries(compareSociety) : null;
  const compareAt = compareSeries ? nearestIndex(compareSeries.samples, sample.y) : -1;
  const compareSample = compareAt >= 0 ? compareSeries?.samples[compareAt] : undefined;
  const compareMetrics = compareAt >= 0 ? compareSeries?.metrics[compareAt] : undefined;

  const points: ChartPoint[] = series.samples.map((item, index) => {
    const metric = series.metrics[index];
    let overlay: number | null = null;
    if (compareSeries) {
      const hit = compareSeries.samples.findIndex((other) => other.y === item.y);
      overlay = hit >= 0 ? (compareSeries.metrics[hit]?.vMean ?? null) : null;
    }
    return {
      x: item.y,
      y: metric?.vMean ?? 0,
      compare: overlay,
      crisis: metric?.crisis ?? false,
    };
  });

  const note = findCase(cases, current.name);
  const marks = notables(series.samples, series.metrics);
  const band = bandLabel(metrics);
  const bonds = topBonds(metrics.B, node, 2);
  const years = current.years.length;
  const totalYears = panel.reduce((sum, item) => sum + item.years.length, 0);

  const jump = (index: number) => {
    setPlaying(false);
    setCursor(index);
  };

  return (
    <div className="lg:grid lg:h-full lg:grid-cols-[18rem_minmax(0,1fr)]">
      <aside
        className={cn(
          "overflow-y-auto border-b border-line lg:block lg:h-full lg:max-h-none lg:border-r lg:border-b-0",
          listOpen ? "block max-h-96" : "hidden lg:block",
        )}
      >
        <div className="sticky top-0 z-10 border-b border-line bg-canvas px-3 py-3">
          <label className="block">
            <span className="sr-only">Search societies</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search the panel"
              className="h-11 w-full rounded-sm border border-line bg-raise px-3 text-sm text-bone outline-none placeholder:text-faint"
            />
          </label>
          <div className="mt-2 flex gap-1" role="group" aria-label="Sort societies">
            {(
              [
                ["name", "Name"],
                ["strained", "Strained"],
                ["span", "Span"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setSort(key)}
                className={cn(
                  "h-8 px-2 text-xs",
                  sort === key ? "text-brass" : "text-mute hover:text-bone",
                )}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-1 font-mono text-xs text-faint">
            {panel.length} societies · {totalYears.toLocaleString("en-AU")} years
          </p>
        </div>
        <ul>
          {ranked.map((item) => (
            <li key={item.name}>
              <button
                type="button"
                onClick={() => {
                  const next = societySeries(panel[item.index] as PanelSociety);
                  setSelected(item.index);
                  setCursor(sharpestDrop(next.metrics));
                  setNode(0);
                  setPlaying(false);
                  setListOpen(false);
                }}
                className={cn(
                  "flex w-full items-baseline justify-between gap-3 border-l-2 px-3 py-2.5 text-left",
                  item.index === selected ? "border-brass bg-raise" : "border-transparent hover:bg-raise",
                )}
              >
                <span className="min-w-0">
                  <span className="block truncate">{item.name}</span>
                  <span className="font-mono text-xs text-faint">
                    {item.from}–{item.to}
                  </span>
                </span>
                <span
                  className={cn(
                    "shrink-0 font-mono text-sm tabular-nums",
                    item.now < 0 ? "text-danger" : item.now > 4 ? "text-bone" : "text-mute",
                  )}
                >
                  {formatSigned(item.now)}
                </span>
              </button>
            </li>
          ))}
          {ranked.length === 0 ? <li className="px-3 py-6 text-sm text-mute">Nothing matches.</li> : null}
        </ul>
      </aside>

      <div className="lg:h-full lg:overflow-y-auto">
        <div className="mx-auto flex max-w-5xl flex-col gap-6 px-4 py-5 md:px-6 md:py-7">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="min-w-0">
              <p className="font-mono text-xs tracking-widest text-brass">JUNO PANEL</p>
              <h1 className="mt-1 font-display text-4xl leading-none text-balance md:text-6xl">
                {current.name}
              </h1>
              <p className="mt-2 text-sm text-mute">
                {series.samples[0]?.y}–{series.samples[series.samples.length - 1]?.y} · {years} sampled years
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button tone="ghost" className="lg:hidden" onClick={() => setListOpen((open) => !open)}>
                {listOpen ? "Hide list" : "Societies"}
              </Button>
              <label className="block text-sm text-mute">
                <span className="mr-2">Overlay</span>
                <select
                  value={compare ?? ""}
                  onChange={(event) => {
                    const value = event.target.value;
                    setCompare(value === "" ? null : Number(value));
                  }}
                  className="h-11 rounded-sm border border-line bg-raise px-2 text-bone"
                  aria-label="Overlay another society"
                >
                  <option value="">None</option>
                  {panel.map((item, index) =>
                    index === selected ? null : (
                      <option key={item.name} value={index}>
                        {item.name}
                      </option>
                    ),
                  )}
                </select>
              </label>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div>
              <div className="font-display text-5xl leading-none tabular-nums">{sample.y}</div>
              <div className={cn("mt-1 text-sm", bandClass(band))}>{band}</div>
            </div>
            <Button
              tone="ghost"
              className="w-11 px-0"
              aria-label={playing ? "Pause" : "Play"}
              onClick={() => setPlaying((value) => !value)}
            >
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </Button>
            <label className="min-w-48 flex-1">
              <span className="sr-only">Year scrubber</span>
              <input
                type="range"
                min={0}
                max={series.samples.length - 1}
                value={safeCursor}
                onChange={(event) => jump(Number(event.target.value))}
              />
            </label>
            <div className="flex gap-1">
              <Button tone="ghost" className="w-11 px-0" aria-label="Previous year" onClick={() => jump(Math.max(0, safeCursor - 1))}>
                −
              </Button>
              <Button
                tone="ghost"
                className="w-11 px-0"
                aria-label="Next year"
                onClick={() => jump(Math.min(series.samples.length - 1, safeCursor + 1))}
              >
                +
              </Button>
            </div>
          </div>

          <SeriesChart
            points={points}
            cursor={sample.y}
            compareName={compareSociety?.name ?? null}
            onPick={(x) => jump(nearestIndex(series.samples, x))}
          />
          <p className="text-xs text-faint">
            Brass is {current.name}. The dashed line at 4 is the strained floor. Red dots are systemic flags.
            {compareSociety && compareSample
              ? ` Overlay is ${compareSociety.name} at ${compareSample.y}.`
              : ""}
          </p>

          <div className="flex flex-wrap gap-2">
            {marks.drops.map((item) => (
              <button
                key={`drop-${item.year}`}
                type="button"
                onClick={() => jump(item.index)}
                className="h-11 border border-line px-3 text-left text-sm hover:border-mute"
              >
                <span className="block text-xs text-faint">Sharp drop</span>
                <span className="font-mono tabular-nums">
                  {item.year} · {formatSigned(item.value)}
                </span>
              </button>
            ))}
            {marks.omega[0] ? (
              <button
                type="button"
                onClick={() => jump(marks.omega[0].index)}
                className="h-11 border border-line px-3 text-left text-sm hover:border-mute"
              >
                <span className="block text-xs text-faint">Wildest step</span>
                <span className="font-mono tabular-nums">
                  {marks.omega[0].year} · ω {marks.omega[0].value.toFixed(2)}
                </span>
              </button>
            ) : null}
            {marks.eta[0] ? (
              <button
                type="button"
                onClick={() => jump(marks.eta[0].index)}
                className="h-11 border border-line px-3 text-left text-sm hover:border-mute"
              >
                <span className="block text-xs text-faint">Thinnest memory</span>
                <span className="font-mono tabular-nums">
                  {marks.eta[0].year} · η {marks.eta[0].value.toFixed(2)}
                </span>
              </button>
            ) : null}
          </div>

          <Instruments metrics={metrics} />

          <p className="max-w-2xl text-pretty text-lg">{reading(metrics)}</p>
          {metrics.pathologies.length > 0 ? (
            <ul className="flex flex-wrap gap-2">
              {metrics.pathologies.map((item) => (
                <li key={item} className="border border-danger px-2 py-1 text-xs tracking-wide text-danger">
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-faint">No pathology signature on this step.</p>
          )}

          {compareMetrics && compareSample ? (
            <p className="text-sm text-mute">
              {compareSociety?.name} in {compareSample.y} sits at{" "}
              <span className="font-mono text-bone">{formatSigned(compareMetrics.vMean)}</span>
              {" · "}
              {bandLabel(compareMetrics)}.
            </p>
          ) : null}

          <div className="grid items-center gap-4 lg:grid-cols-[minmax(0,22rem)_1fr]">
            <Ring
              metrics={metrics}
              nodes={sample.nodes}
              yearLabel={String(sample.y)}
              selected={node}
              onSelect={setNode}
            />
            <div>
              <p className="text-xs tracking-widest text-faint">NODE ORDER</p>
              <p className="mt-1 text-sm text-mute">
                Executive in red, cognitive in brass, material in green. Node size is its value. A halo means stress
                above 6. Bonds fade as stress rises.
              </p>
              <p className="mt-4 font-mono text-sm text-bone">
                {NODES[node]?.id} · {NODES[node]?.gloss}
              </p>
              <p className="mt-1 font-mono text-sm tabular-nums text-mute">
                C {sample.nodes[node]?.[0].toFixed(1)} · K {sample.nodes[node]?.[1].toFixed(1)} · S{" "}
                {sample.nodes[node]?.[2].toFixed(1)} · A {sample.nodes[node]?.[3].toFixed(1)}
              </p>
              <p className="mt-1 font-mono text-sm tabular-nums">
                V = C+K−S+½A = {formatSigned(metrics.V[node] ?? 0)}
              </p>
              <p className="mt-1 text-sm text-mute">
                Strongest bonds:{" "}
                {bonds.map((bond) => `${NODES[bond.index]?.id} ${bond.bond.toFixed(2)}`).join(" · ")}
              </p>
            </div>
          </div>

          <ul className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            {NODES.map((item, index) => {
              const value = metrics.V[index] ?? 0;
              const vec = sample.nodes[index] ?? [0, 0, 0, 0];
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => setNode(index)}
                    className={cn(
                      "h-full w-full border px-3 py-3 text-left",
                      node === index ? "border-brass bg-raise" : "border-line hover:border-mute",
                    )}
                  >
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="font-mono text-xs tracking-wide">{item.id}</span>
                      <span className={cn("font-mono text-sm tabular-nums", value < 0 ? "text-danger" : "text-bone")}>
                        {formatSigned(value)}
                      </span>
                    </span>
                    <span className="mt-1 block text-xs text-faint">{item.gloss}</span>
                    <span className="mt-2 grid grid-cols-4 gap-1 font-mono text-xs tabular-nums text-mute">
                      <span>C {vec[0].toFixed(1)}</span>
                      <span>K {vec[1].toFixed(1)}</span>
                      <span>S {vec[2].toFixed(1)}</span>
                      <span>A {vec[3].toFixed(1)}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>

          {note ? (
            <aside className="border border-line bg-raise px-4 py-4">
              <p className="font-mono text-xs tracking-widest text-brass">FIELD NOTE · BLIND SET</p>
              <p className="mt-2 max-w-2xl text-pretty">{note.blurb}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {note.onsets.map((onset) => (
                  <Button key={onset} tone="ghost" onClick={() => jump(nearestIndex(series.samples, onset))}>
                    Go to {onset}
                  </Button>
                ))}
                {note.onsets.length === 0 ? (
                  <span className="text-sm text-mute">No systemic onset is marked in this window.</span>
                ) : null}
              </div>
            </aside>
          ) : null}
        </div>
      </div>
    </div>
  );
}
