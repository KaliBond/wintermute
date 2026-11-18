import { NODES, type NodeVec, type YearMetrics } from "@/lib/cams/model";

const ORDER = [0, 1, 4, 7, 5, 6, 2, 3];

function at(slot: number, radius: number): [number, number] {
  const angle = -Math.PI / 2 + (slot / 8) * Math.PI * 2;
  return [200 + Math.cos(angle) * radius, 206 + Math.sin(angle) * radius];
}

function fill(family: (typeof NODES)[number]["family"]): string {
  if (family === "executive") return "var(--color-danger)";
  if (family === "cognitive") return "var(--color-brass)";
  return "var(--color-good)";
}

export function Ring({
  metrics,
  nodes,
  yearLabel,
  selected,
  onSelect,
  ghost = false,
}: {
  metrics: YearMetrics;
  nodes: NodeVec[];
  yearLabel: string;
  selected: number | null;
  onSelect?: (index: number) => void;
  ghost?: boolean;
}) {
  const slotOf = ORDER.map((node, slot) => ({ node, slot }));
  const bonds = slotOf.flatMap(({ node: i }) =>
    slotOf.flatMap(({ node: j }) => {
      if (j <= i) return [];
      const bond = metrics.B[i * 8 + j] ?? 0;
      if (bond < 0.08) return [];
      const [x1, y1] = at(ORDER.indexOf(i), 118);
      const [x2, y2] = at(ORDER.indexOf(j), 118);
      const dim = selected !== null && selected !== i && selected !== j;
      return [
        {
          key: `${i}-${j}`,
          x1,
          y1,
          x2,
          y2,
          opacity: dim ? 0.06 : Math.min(0.15 + bond, 0.85),
          width: 0.6 + bond * 7,
        },
      ];
    }),
  );

  return (
    <svg
      viewBox="-44 -16 488 448"
      className="mx-auto w-full max-w-sm"
      role={ghost ? "img" : "group"}
      aria-label={ghost ? undefined : "Eight institutional nodes"}
      aria-hidden={ghost || undefined}
    >
      <circle cx="200" cy="206" r="118" fill="none" stroke="var(--color-line)" strokeWidth="1" />
      {bonds.map((bond) => (
        <line
          key={bond.key}
          x1={bond.x1}
          y1={bond.y1}
          x2={bond.x2}
          y2={bond.y2}
          stroke="var(--color-bone)"
          strokeOpacity={bond.opacity}
          strokeWidth={bond.width}
        />
      ))}
      {slotOf.map(({ node, slot }) => {
        const vec = nodes[node] ?? [0, 0, 0, 0];
        const value = metrics.V[node] ?? 0;
        const stress = vec[2];
        const radius = Math.max(8, Math.min(24, 11 + value * 0.85));
        const [x, y] = at(slot, 118);
        const [lx, ly] = at(slot, 168);
        const nodeMeta = NODES[node];
        const negative = value < 0;
        const active = selected === node;
        const anchor = lx < 176 ? "end" : lx > 224 ? "start" : "middle";
        return (
          <g
            key={node}
            role={onSelect ? "button" : undefined}
            tabIndex={onSelect ? 0 : undefined}
            className={onSelect ? "cursor-pointer" : undefined}
            onClick={onSelect ? () => onSelect(node) : undefined}
            onKeyDown={
              onSelect
                ? (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect(node);
                    }
                  }
                : undefined
            }
          >
            {stress > 6 ? (
              <circle
                cx={x}
                cy={y}
                r={radius + 8}
                fill="var(--color-danger)"
                opacity={Math.min(0.35, 0.12 + (stress - 6) * 0.04)}
              />
            ) : null}
            <circle
              cx={x}
              cy={y}
              r={radius}
              fill={negative ? "transparent" : fill(nodeMeta?.family ?? "material")}
              stroke={negative ? "var(--color-danger)" : "transparent"}
              strokeWidth={negative ? 2 : 0}
            />
            {active ? (
              <circle cx={x} cy={y} r={radius + 5} fill="none" stroke="var(--color-bone)" strokeWidth="1.5" />
            ) : null}
            <text
              x={lx}
              y={ly}
              textAnchor={anchor}
              dominantBaseline="middle"
              fill={active ? "var(--color-bone)" : "var(--color-mute)"}
              fontSize="12"
              fontFamily="var(--font-mono)"
            >
              {nodeMeta?.id}
            </text>
          </g>
        );
      })}
      <text
        x="200"
        y="210"
        textAnchor="middle"
        fill="var(--color-bone)"
        fontSize="15"
        fontFamily="var(--font-mono)"
      >
        {yearLabel || `λ₂ ${metrics.lambda2.toFixed(2)}`}
      </text>
    </svg>
  );
}
