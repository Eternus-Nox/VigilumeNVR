import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Dev-only: mirror the nginx routing so the SPA works against a running stack.
const devProxy = {
  '/api': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
  '/go2rtc': {
    target: 'http://localhost:1984',
    changeOrigin: true,
    ws: true,
    rewrite: (p: string) => p.replace(/^\/go2rtc/, ''),
  },
};

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.ts',
      registerType: 'autoUpdate',
      injectRegister: false,
      manifest: {
        name: 'Vigilume NVR',
        short_name: 'Vigilume',
        description: 'Self-hosted security camera NVR',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        orientation: 'any',
        background_color: '#0b1017',
        theme_color: '#0b1017',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
          { src: '/icons/maskable-192.png', sizes: '192x192', type: 'image/png', purpose: 'maskable' },
          { src: '/icons/maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      injectManifest: {
        globPatterns: ['**/*.{js,css,html,svg,png,woff2,webmanifest}'],
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
      },
      devOptions: { enabled: false },
    }),
  ],
  build: {
    target: 'es2020',
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: devProxy,
  },
  // The `vite preview` server (built dist) needs its own proxy — `server.proxy`
  // only applies to the dev server. Mirror it so previews reach a local stack.
  preview: {
    proxy: devProxy,
  },
});
