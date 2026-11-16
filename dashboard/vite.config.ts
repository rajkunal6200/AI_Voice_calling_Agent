import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy to the FastAPI dev server so the browser sees one origin and the
    // CORS config only has to matter if you run the UI from somewhere else.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
