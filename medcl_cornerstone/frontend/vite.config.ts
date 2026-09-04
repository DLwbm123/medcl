import { defineConfig } from "vite";

export default defineConfig({
  resolve: { alias: { events: "events/", url: "url/" } },
  build: {
    lib: { entry: "src/index.ts", formats: ["es"], fileName: "index", cssFileName: "style" },
    target: "es2022",
    sourcemap: false,
    emptyOutDir: true,
    outDir: "build",
  },
});
