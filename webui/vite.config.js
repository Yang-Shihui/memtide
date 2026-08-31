import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI is served by the memtide REST server at the ROOT in production.
// In dev, `npm run dev` proxies API calls to a locally running server.
export default defineConfig({
  plugins: [react()],
  base: "/console/",
  server: {
    port: 5173,
    proxy: {
      "/memories": "http://localhost:8300",
      "/search": "http://localhost:8300",
      "/context": "http://localhost:8300",
      "/history": "http://localhost:8300",
      "/stats": "http://localhost:8300",
      "/consolidate": "http://localhost:8300",
      "/reset": "http://localhost:8300",
    },
  },
  build: { outDir: "dist" },
});
