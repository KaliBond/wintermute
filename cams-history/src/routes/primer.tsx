import { createFileRoute } from "@tanstack/react-router";
import { PrimerView } from "@/components/primer-view";
import { AppShell } from "@/components/shell";

export const Route = createFileRoute("/primer")({
  head: () => ({ meta: [{ title: "Primer · CAMS History" }] }),
  component: PrimerPage,
});

function PrimerPage() {
  return (
    <AppShell>
      <PrimerView />
    </AppShell>
  );
}
