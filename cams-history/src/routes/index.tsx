import { createFileRoute } from "@tanstack/react-router";
import { AtlasView } from "@/components/atlas-view";
import { AppShell } from "@/components/shell";

type AtlasSearch = { society?: string; year?: number };

export const Route = createFileRoute("/")({
  validateSearch: (search: Record<string, unknown>): AtlasSearch => {
    const yearRaw = search.year;
    const year =
      typeof yearRaw === "number"
        ? yearRaw
        : typeof yearRaw === "string" && yearRaw.trim() !== "" && Number.isFinite(Number(yearRaw))
          ? Number(yearRaw)
          : undefined;
    return {
      society: typeof search.society === "string" ? search.society : undefined,
      year,
    };
  },
  head: () => ({ meta: [{ title: "Atlas · CAMS History" }] }),
  component: AtlasPage,
});

function AtlasPage() {
  const search = Route.useSearch();
  return (
    <AppShell>
      <AtlasView society={search.society} year={search.year} />
    </AppShell>
  );
}
