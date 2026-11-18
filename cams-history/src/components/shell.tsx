import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

const LINKS = [
  { to: "/", label: "Atlas" },
  { to: "/prophet", label: "Prophet" },
  { to: "/primer", label: "Primer" },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const path = useRouterState({ select: (state) => state.location.pathname });

  return (
    <div className="flex min-h-dvh flex-col bg-canvas text-bone lg:h-dvh lg:overflow-hidden">
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line px-4">
        <Link to="/" className="min-w-0">
          <span className="font-mono text-xs tracking-widest text-brass">CAMS</span>
          <span className="ml-2 font-display text-lg italic text-bone">History</span>
        </Link>
        <nav className="flex items-center gap-1" aria-label="Sections">
          {LINKS.map((link) => {
            const active = link.to === "/" ? path === "/" : path.startsWith(link.to);
            return (
              <Link
                key={link.to}
                to={link.to}
                className={cn(
                  "inline-flex h-11 items-center px-3 text-sm",
                  active ? "text-brass" : "text-mute hover:text-bone",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </header>
      <div className="min-h-0 lg:flex-1">{children}</div>
    </div>
  );
}
