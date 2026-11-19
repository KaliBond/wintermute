/**
 * Static build for neuralnations.org (Cloudflare Pages serves the repo as-is).
 *
 *   npm run build:static
 *
 * Prerenders every route to plain HTML under ../cams-history/app/ so the site
 * can serve it at https://neuralnations.org/cams-history/app/. No server, no
 * database, no sign-in: the app only reads public/data/*.json.
 * The Grok dev/Vercel setup in vite.config.ts is left unchanged.
 */
import { defineConfig } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export const BASE = "/cams-history/app/";

export default defineConfig({
  base: BASE,
  resolve: { tsconfigPaths: true },
  plugins: [
    tailwindcss(),
    tanstackStart({
      prerender: { enabled: true, crawlLinks: true, failOnError: true },
      pages: [{ path: "/" }, { path: "/primer" }, { path: "/prophet" }],
    }),
    viteReact(),
  ],
});
