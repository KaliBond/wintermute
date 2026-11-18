import { INSTRUMENTS, formatSigned, type YearMetrics } from "@/lib/cams/model";
import { cn } from "@/lib/cn";

export function Instruments({ metrics }: { metrics: YearMetrics }) {
  return (
    <div className="grid grid-cols-3 gap-x-3 gap-y-4 sm:grid-cols-6">
      {INSTRUMENTS.map((item) => {
        const raw = metrics[item.key];
        const missing = raw === null || raw === undefined;
        const value = missing ? null : raw;
        const healthy = value !== null && item.good(value);
        const span = item.hi - item.lo;
        const pct =
          value === null ? 0 : Math.min(100, Math.max(0, ((value - item.lo) / span) * 100));
        return (
          <div key={item.key} title={item.hint} className="min-w-0">
            <div className="text-xs text-mute">{item.label}</div>
            <div
              className={cn(
                "font-mono text-lg tabular-nums",
                missing || item.neutral ? "text-bone" : healthy ? "text-bone" : "text-danger",
              )}
            >
              {value === null ? "—" : formatSigned(value, item.dp)}
            </div>
            <div className="mt-1.5 h-0.5 bg-line">
              <div
                className={cn("h-full", item.neutral ? "bg-mute" : healthy ? "bg-good" : "bg-danger")}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
