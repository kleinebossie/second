import { execFileSync } from 'child_process';
import fs from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, 'dist');
const releaseDir = resolve(__dirname, 'release');

const REQUIRED_DIST_FILES = [
  'manifest.json',
  'sidepanel.html',
  'sidepanel.js',
  'background.js',
  'content.js',
  'icons/icon16.png',
  'icons/icon32.png',
  'icons/icon48.png',
  'icons/icon128.png',
];

function packageExtension() {
  console.log('Building extension before packaging...');
  execFileSync(process.execPath, [resolve(__dirname, 'build.js')], {
    cwd: __dirname,
    stdio: 'inherit',
  });

  for (const relPath of REQUIRED_DIST_FILES) {
    const fullPath = resolve(distDir, relPath);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing required extension build file: ${relPath}`);
    }
  }

  const manifest = JSON.parse(
    fs.readFileSync(resolve(distDir, 'manifest.json'), 'utf8')
  );
  const version = String(manifest.version || '1.0.0');
  const zipName = `second-extension-v${version}.zip`;
  const zipPath = resolve(releaseDir, zipName);

  fs.mkdirSync(releaseDir, { recursive: true });
  fs.rmSync(zipPath, { force: true });

  console.log(`Creating store archive ${zipPath}...`);
  execFileSync('zip', ['-r', '-q', zipPath, '.'], {
    cwd: distDir,
    stdio: 'inherit',
  });

  const stats = fs.statSync(zipPath);
  const mb = (stats.size / (1024 * 1024)).toFixed(2);
  console.log(`Packaged ${zipName} (${mb} MiB) in extension/release/.`);
}

try {
  packageExtension();
} catch (err) {
  console.error('Packaging failed:', err);
  process.exit(1);
}
