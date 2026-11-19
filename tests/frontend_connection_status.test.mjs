import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const source = await readFile(new URL('../src/api/connectionStatus.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const module = { exports: {} };
new Function('exports', 'require', 'module', compiled)(module.exports, require, module);

const { getBackendConnectionLabel, markBackendConnected, markLocalFallback } = module.exports;

assert.equal(getBackendConnectionLabel(true, false), 'LIVE BACKEND');
assert.equal(getBackendConnectionLabel(true, true), 'DEMO CONTROLLER');
assert.equal(getBackendConnectionLabel(false, true), 'DISCONNECTED');
assert.equal(markBackendConnected({ isDemoMode: true }).isDemoMode, true);
assert.equal(markBackendConnected({ isDemoMode: false }).isDemoMode, false);
assert.deepEqual(markLocalFallback({ isDemoMode: false, isConnectedToBackend: true }), {
  isDemoMode: true,
  isConnectedToBackend: false,
});

console.log('frontend connection status: 6 assertions passed');
