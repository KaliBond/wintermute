import { useEffect, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatSigned } from "@/lib/cams/model";

export type ChartPoint = {
  x: number;
  y: number;
  compare: number | null;
  crisis: boolean;
};

function TickTip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: ChartPoint }[];
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="border border-line bg-raise px-3 py-2 text-sm">
      <div className="font-mono tabular-nums text-bone">{point.x}</div>
      <div className="font-mono tabular-nums text-brass">V̄ {formatSigned(point.y)}</div>
      {point.compare !== null ? (
        <div className="font-mono tabular-nums text-mute">Overlay {formatSigned(point.compare)}</div>
      ) : null}
      {point.crisis ? <div className="text-danger">Systemic flag</div> : null}
    </div>
  );
}

export function SeriesChart({
  points,
  cursor,
  onPick,
  compareName,
}: {
  points: ChartPoint[];
  cursor: number | null;
  onPick?: (x: number) => void;
  compareName?: string | null;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-64 bg-raise sm:h-80" />;

  return (
    <div className="h-64 sm:h-80">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={points}
          margin={{ top: 12, right: 8, left: 0, bottom: 0 }}
          onClick={(state) => {
            const label = (state as { activeLabel?: number | string } | undefined)?.activeLabel;
            if (label !== undefined && onPick) onPick(Number(label));
          }}
        >
          <CartesianGrid stroke="var(--color-line)" vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fill: "var(--color-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "var(--color-line)" }}
            tickLine={false}
            allowDecimals={false}
            minTickGap={28}
          />
          <YAxis
            domain={[-8, 18]}
            ticks={[-8, -4, 0, 4, 8, 12, 16]}
            width={32}
            tick={{ fill: "var(--color-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceLine y={0} stroke="var(--color-danger)" strokeOpacity={0.45} />
          <ReferenceLine y={4} stroke="var(--color-mute)" strokeDasharray="3 4" strokeOpacity={0.7} />
          {cursor !== null ? (
            <ReferenceLine x={cursor} stroke="var(--color-brass)" strokeWidth={1.25} />
          ) : null}
          {points.length > 2 ? (
            <Tooltip content={<TickTip />} cursor={{ stroke: "var(--color-line)" }} />
          ) : null}
          <Area
            type="linear"
            dataKey="y"
            stroke="var(--color-brass)"
            fill="var(--color-brass)"
            fillOpacity={0.12}
            strokeWidth={1.75}
            dot={(props) => {
              const point = props.payload as ChartPoint | undefined;
              const cx = props.cx as number | undefined;
              const cy = props.cy as number | undefined;
              if (!point?.crisis || cx === undefined || cy === undefined) {
                return <g key={`dot-${point?.x ?? "x"}`} />;
              }
              return <circle key={`crisis-${point.x}`} cx={cx} cy={cy} r={2.5} fill="var(--color-danger)" />;
            }}
            activeDot={{ r: 3, fill: "var(--color-bone)", stroke: "none" }}
            name="V̄"
          />
          {compareName ? (
            <Line
              type="linear"
              dataKey="compare"
              stroke="var(--color-bone)"
              strokeOpacity={0.7}
              strokeDasharray="4 4"
              strokeWidth={1.25}
              dot={false}
              connectNulls={false}
              name={compareName}
            />
          ) : null}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
