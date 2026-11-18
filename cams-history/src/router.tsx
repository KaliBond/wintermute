import { createRouter } from "@tanstack/react-router";
import { AppErrorComponent } from "@/lib/error-component";
import { previewPathRewrite } from "@/lib/preview-path-prefix";
import { routeTree } from "./routeTree.gen";

export function getRouter() {
  return createRouter({
    routeTree,
    // "/" in dev; "/cams-history/app" in the static build for neuralnations.org.
    basepath: import.meta.env.BASE_URL.replace(/\/$/, "") || "/",
    rewrite: previewPathRewrite(),
    defaultErrorComponent: AppErrorComponent,
  });
}
