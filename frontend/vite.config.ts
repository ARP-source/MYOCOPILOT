import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// public/hand.html is a straight copy of assets/hand.html: a standalone
// page that loads three.js via a browser-native <script type="importmap">
// pointing at a CDN, with no bundler involved. Vite's dev server otherwise
// intercepts *every* .html request and runs its own import-analysis on
// embedded module scripts, which fails trying to resolve "three" as an
// npm package. This plugin serves that one file byte-for-byte, ahead of
// Vite's internal HTML middleware, so it behaves the same in dev as it
// does when opened statically or served from the production build (where
// files under public/ are copied verbatim and never touch this problem).
function rawHandHtml(): Plugin {
  return {
    name: "myocopilot-raw-hand-html",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === "/hand.html" || req.url?.startsWith("/hand.html?")) {
          res.setHeader("Content-Type", "text/html; charset=utf-8");
          res.end(readFileSync(path.resolve(__dirname, "public/hand.html")));
          return;
        }
        next();
      });
    },
  };
}

// Backend (web/server.py) is started separately with:
//   py -m uvicorn web.server:app --reload --port 8000
// Dev-server proxy forwards REST + WebSocket traffic so the SPA can use
// same-origin relative URLs (/stream, /history, /stats) in both dev and prod.
export default defineConfig({
  plugins: [rawHandHtml(), react()],
  // Without this, Vite's dependency scanner globs every .html file in the
  // project, including public/hand.html, and tries to pre-bundle its
  // CDN-importmap-resolved "three" imports as if they were npm packages,
  // logging a spurious (non-fatal) error on every dev-server start.
  optimizeDeps: {
    entries: ["index.html"],
  },
  server: {
    port: 5173,
    proxy: {
      "/stream": {
        target: "ws://localhost:8000",
        ws: true,
      },
      "/history": "http://localhost:8000",
      "/stats": "http://localhost:8000",
    },
  },
});
