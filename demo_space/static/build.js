import { createRequire } from 'module';
import fs from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../..');
const extNodeModules = resolve(repoRoot, 'extension/node_modules');
const requireFromExt = createRequire(resolve(extNodeModules, 'package.json'));
const { build } = requireFromExt('vite');

const outDir = resolve(repoRoot, 'demo_space/dist_static');

async function buildStaticSpace() {
  await build({
    root: __dirname,
    base: './',
    publicDir: resolve(repoRoot, 'extension/public'),
    build: {
      outDir,
      emptyOutDir: true,
      target: 'esnext',
      minify: 'esbuild',
    },
  });

  fs.copyFileSync(resolve(__dirname, 'README.md'), resolve(outDir, 'README.md'));
  fs.rmSync(resolve(outDir, '.gitkeep'), { force: true });
  console.log(`Built Static Space in ${outDir}`);
}

buildStaticSpace().catch((err) => {
  console.error('Static Space build failed:', err);
  process.exit(1);
});
