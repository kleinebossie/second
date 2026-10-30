import { defineConfig } from "vite";
import { resolve } from "path";
import fs from "fs";

export default defineConfig({
  build: {
    outDir: "dist-ext",
    emptyOutDir: true,
    target: "esnext",
    rollupOptions: {
      input: {
        sidepanel: resolve(__dirname, "sidepanel.html")
      },
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash].[ext]"
      }
    }
  },
  plugins: [
    {
      name: "copy-extension-assets",
      closeBundle() {
        const outDir = resolve(__dirname, "dist-ext");
        fs.copyFileSync(resolve(__dirname, "manifest.json"), resolve(outDir, "manifest.json"));
        fs.copyFileSync(resolve(__dirname, "background.js"), resolve(outDir, "background.js"));

        // Copy wllama wasm binary for Manifest V3 extension execution
        const wasmDir = resolve(outDir, "wasm");
        if (!fs.existsSync(wasmDir)) {
          fs.mkdirSync(wasmDir, { recursive: true });
        }
        const wllamaWasmSource = resolve(__dirname, "node_modules/@wllama/wllama/esm/wasm/wllama.wasm");
        if (fs.existsSync(wllamaWasmSource)) {
          fs.copyFileSync(wllamaWasmSource, resolve(wasmDir, "wllama.wasm"));
          console.log("Copied wllama.wasm to dist-ext/wasm/ successfully.");
        }

        // Copy onnxruntime wasm/mjs binaries for Manifest V3 extension execution
        const ortDir = resolve(outDir, "ort");
        if (!fs.existsSync(ortDir)) {
          fs.mkdirSync(ortDir, { recursive: true });
        }
        const onnxSourceDirs = [
          resolve(__dirname, "node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist"),
          resolve(__dirname, "node_modules/onnxruntime-web/dist")
        ];
        let copiedOrtCount = 0;
        for (const dir of onnxSourceDirs) {
          if (fs.existsSync(dir)) {
            const files = fs.readdirSync(dir);
            for (const file of files) {
              if (file.startsWith("ort-wasm") && (file.endsWith(".wasm") || file.endsWith(".mjs"))) {
                fs.copyFileSync(resolve(dir, file), resolve(ortDir, file));
                copiedOrtCount++;
              }
            }
            if (copiedOrtCount > 0) {
              console.log(`Copied ${copiedOrtCount} onnxruntime files from ${dir} to dist-ext/ort/ successfully.`);
              break;
            }
          }
        }

        console.log("Copied manifest.json and background.js to dist-ext/ successfully.");

        // Copy reference and prompt JSON files
        for (const file of ["reference_output.json", "prompt.json"]) {
          const src = resolve(__dirname, file);
          if (fs.existsSync(src)) {
            fs.copyFileSync(src, resolve(outDir, file));
            console.log(`Copied ${file} to dist-ext/ successfully.`);
          }
        }
      }
    }
  ]
});
