import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the api runs on :8000 (framehound-api); nginx does the same /api rewrite in compose.
const API = process.env.FH_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api": { target: API, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
});
