import { build } from 'vite';
import { execFileSync } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function runBuild() {
  console.log('Running TypeScript check...');
  execFileSync(
    process.platform === 'win32' ? 'npx.cmd' : 'npx',
    ['tsc', '--noEmit'],
    { cwd: __dirname, stdio: 'inherit' }
  );

  console.log('Starting Second extension build...');

  // 1. Build Sidepanel and Background Worker (ES modules)
  console.log('Building sidepanel and background service worker...');
  await build({
    configFile: false,
    publicDir: resolve(__dirname, 'public'),
    esbuild: {
      legalComments: 'none',
      drop: ['debugger'],
    },
    build: {
      outDir: resolve(__dirname, 'dist'),
      emptyOutDir: true,
      target: 'esnext',
      minify: 'esbuild',
      cssMinify: 'esbuild',
      rollupOptions: {
        input: {
          sidepanel: resolve(__dirname, 'sidepanel.html'),
          background: resolve(__dirname, 'src/background.ts'),
        },
        output: {
          entryFileNames: '[name].js',
          chunkFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]',
          format: 'es',
        },
      },
    },
  });

  // 2. Build Content Script (IIFE for isolated Chrome DOM world)
  console.log('Building content script (IIFE)...');
  await build({
    configFile: false,
    publicDir: false,
    esbuild: {
      legalComments: 'none',
      drop: ['debugger'],
    },
    build: {
      outDir: resolve(__dirname, 'dist'),
      emptyOutDir: false,
      target: 'esnext',
      minify: 'esbuild',
      rollupOptions: {
        input: {
          content: resolve(__dirname, 'src/content.ts'),
        },
        output: {
          entryFileNames: 'content.js',
          format: 'iife',
          name: 'SecondContentScript',
        },
      },
    },
  });

  // 3. Copy manifest.json into dist
  console.log('Copying manifest.json to dist...');
  fs.copyFileSync(
    resolve(__dirname, 'manifest.json'),
    resolve(__dirname, 'dist/manifest.json')
  );

  // 4. Remove placeholder .gitkeep copied from public/
  fs.rmSync(resolve(__dirname, 'dist/.gitkeep'), { force: true });

  console.log('Second extension build completed successfully!');
}

runBuild().catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
});
