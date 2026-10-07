import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // single-page 3D viewer: three.js + react vendor chunk is inherently large
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three", "@react-three/fiber", "@react-three/drei"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.MMDE_STUDIO_API ?? "http://localhost:8010",
        changeOrigin: true,
      },
      "/ws": {
        target: process.env.MMDE_STUDIO_API ?? "http://localhost:8010",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
