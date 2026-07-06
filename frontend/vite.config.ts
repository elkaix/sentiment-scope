import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev-server proxy: the UI calls relative /api/... paths and Vite
    // forwards them to FastAPI. Same-origin from the browser's view, so no
    // CORS gymnastics; in Docker, nginx plays this exact role.
    proxy: { "/api": "http://127.0.0.1:8002" },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    globals: true,
  },
});
