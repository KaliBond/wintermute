import { createFileRoute } from "@tanstack/react-router";
import { ProphetView } from "@/components/prophet-view";
import { AppShell } from "@/components/shell";

export const Route = createFileRoute("/prophet")({
  head: () => ({ meta: [{ title: "Blind Prophet · CAMS History" }] }),
  component: ProphetPage,
});

function ProphetPage() {
  return (
    <AppShell>
      <ProphetView />
    </AppShell>
  );
}
