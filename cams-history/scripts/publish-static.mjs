// Copy the prerendered static build into ./app, which neuralnations.org
// serves at /cams-history/app/. Run via `npm run build:static`.
import { cpSync, rmSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const from = `${root}dist/client`;
const to = `${root}app`;
if (!existsSync(`${from}/index.html`)) throw new Error(`No build at ${from}; run vite build first`);
rmSync(to, { recursive: true, force: true });
cpSync(from, to, { recursive: true });
console.log(`Published ${from} -> ${to}`);
