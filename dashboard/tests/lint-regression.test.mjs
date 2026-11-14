// Run from dashboard/: node --test tests/lint-regression.test.mjs
// Browser storage/network below are test fixtures; no analytics service is called.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import test from 'node:test';
import { build, transform } from 'esbuild';
import { ESLint } from 'eslint';

const source = await readFile('src/lib/consent.js', 'utf8');
const consentCode = (await transform(source, { format: 'cjs' })).code;
function consentFixture({ country = 'DE', blocked = false, response = 'loc=US\n', failFetch = false } = {}) {
  const values = new Map(country ? [['up_geo_cc', country]] : []);
  const requests = [];
  let starts = 0;
  const storage = {
    getItem(key) { if (blocked) throw new Error('blocked'); return values.get(key) ?? null; },
    setItem(key, value) { if (blocked) throw new Error('blocked'); values.set(key, value); },
  };
  const fetch = async (...args) => {
    requests.push(args);
    if (failFetch) throw new Error('offline');
    return { ok: true, text: async () => response };
  };
  const context = vm.createContext({
    module: { exports: {} }, localStorage: storage, fetch,
    window: { localStorage: storage, fetch, __osAnalyticsInit: () => { starts++; return true; } },
    Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Europe/Berlin' }) }) },
  });
  vm.runInContext(consentCode, context);
  return { api: context.module.exports, values, requests, starts: () => starts };
}
const plain = (value) => JSON.parse(JSON.stringify(value));
const settle = () => new Promise((resolve) => setImmediate(resolve));

test('dashboard lint is clean and authored scripts remain checked', async () => {
  const eslint = new ESLint();
  assert.equal(await eslint.isPathIgnored('public/op1.js'), true);
  const results = await eslint.lintFiles(['.']);
  assert.equal(results.reduce((n, r) => n + r.errorCount + r.warningCount, 0), 0,
    (await eslint.loadFormatter('stylish')).format(results));
  for (const filePath of ['src/lint-probe.js', 'public/first-party-probe.js']) {
    assert.equal(await eslint.isPathIgnored(filePath), false);
    const [result] = await eslint.lintText('function probe(unused) { return; neverRuns(); }\nprobe();', { filePath });
    assert.ok(result.messages.some((m) => m.ruleId === 'no-unused-vars'));
    assert.ok(result.messages.some((m) => m.ruleId === 'no-unreachable'));
  }
});

test('vendored OpenPanel bytes are unchanged', async () => {
  // Pinned to the asset introduced by 96f0139, not to a mutable production constant.
  assert.equal(createHash('sha256').update(await readFile('public/op1.js')).digest('hex'),
    'dcb457dff892d17f6c795c1a933fa7506a6f0067dbba9f8267b3432d460c27ec');
});

test('consent persistence, listeners, and always-on analytics retain current semantics', () => {
  // Characterization of current policy, not a legal/compliance endorsement.
  const f = consentFixture();
  assert.deepEqual(plain(f.api.getConsent()), { necessary: true, analytics: true, marketing: false });
  assert.equal(f.api.hasDecided(), false);
  const seen = [];
  const unsubscribe = f.api.onConsentChange((value) => seen.push(plain(value)));
  f.api.onConsentChange(() => { throw new Error('listener failure'); });
  assert.deepEqual(plain(f.api.acceptAll()), { necessary: true, analytics: true, marketing: true });
  assert.deepEqual(plain(f.api.rejectAll()), { necessary: true, analytics: true, marketing: false });
  const stored = JSON.parse(f.values.get('openshorts_consent'));
  assert.equal(stored.v, 1);
  assert.equal(stored.necessary, true);
  assert.equal(stored.analytics, true);
  assert.equal(stored.marketing, false);
  assert.ok(Number.isFinite(Date.parse(stored.ts)));
  assert.equal(f.api.hasDecided(), true);
  assert.deepEqual(seen.map((v) => v.marketing), [true, false]);
  unsubscribe();
  f.api.acceptAll();
  assert.equal(seen.length, 2);
  assert.equal(f.starts(), 3);
  assert.equal(f.requests.length, 0);
  f.values.set('openshorts_consent', '{broken');
  assert.equal(f.api.getConsentRecord(), null);
  f.values.set('openshorts_consent', '{"v":2,"marketing":true}');
  assert.equal(f.api.getConsentRecord(), null);
  assert.equal(consentFixture({ country: 'US' }).api.hasDecided(), true);
});

test('geo lookup keeps same-origin no-store requests and tolerates unavailable storage/network', async () => {
  for (const options of [{ country: null }, { blocked: true }, { country: null, failFetch: true }, { country: null, response: 'loc=invalid\n' }]) {
    const f = consentFixture(options);
    assert.doesNotThrow(() => f.api.rejectAll());
    await settle();
    assert.deepEqual(plain(f.requests), [['/cdn-cgi/trace', { cache: 'no-store' }]]);
    assert.equal(f.starts(), 1);
    assert.equal(f.values.get('up_geo_cc'), options.blocked ? 'DE' : options.failFetch || options.response ? undefined : 'US');
  }
});

test('analytics loader retains configuration, hostname, and once-only gates', async () => {
  const html = await readFile('index.html', 'utf8');
  const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).find((s) => s.includes('window.__osAnalyticsInit'));
  assert.ok(script);
  for (const [configured, hostname, expected] of [[false, 'openshorts.app', false], [true, 'preview.example', false], [true, 'openshorts.app', true]]) {
    const appended = [];
    const code = configured ? script.replaceAll('%VITE_OPENPANEL_API_URL%', 'https://analytics.example').replaceAll('%VITE_OPENPANEL_CLIENT_ID%', 'test-client') : script;
    const context = vm.createContext({ window: {}, location: { hostname }, document: { createElement: () => ({}), head: { appendChild: (s) => appended.push(s) } } });
    vm.runInContext(code, context);
    assert.equal(appended.length, 0);
    assert.equal(context.window.__osAnalyticsInit(), expected);
    assert.equal(context.window.__osAnalyticsInit(), false);
    assert.deepEqual(appended.map((s) => s.src), expected ? ['/op1.js'] : []);
  }
});

test('legal and pricing pages render every row icon with original dimensions', async () => {
  const { outputFiles } = await build({
    stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server';
      import Legal from './src/Legal.jsx'; import PricingPage from './src/components/PricingPage.jsx';
      import { AuthProvider } from './src/contexts/AuthContext.jsx';
      export const legal = renderToStaticMarkup(React.createElement(Legal));
      export const pricing = renderToStaticMarkup(React.createElement(AuthProvider, null, React.createElement(PricingPage)));`,
      resolveDir: process.cwd() },
    bundle: true, write: false, jsx: 'automatic', platform: 'node', format: 'cjs', define: { 'import.meta.env': '{}' },
  });
  const context = vm.createContext({ module: { exports: {} }, console, TextEncoder, TextDecoder, process, require: createRequire(import.meta.url) });
  // Use real React, components, and icon library. SSR does not run browser effects.
  vm.runInContext(outputFiles[0].text, context);
  const { legal, pricing } = context.module.exports;
  for (const icon of ['file-text', 'shield', 'landmark', 'rotate-ccw', 'flag']) {
    assert.match(legal, new RegExp(`<svg[^>]*width="18"[^>]*class="[^"]*lucide-${icon} `));
  }
  for (const icon of ['cpu', 'key-round', 'send', 'server', 'hard-drive']) {
    assert.match(pricing, new RegExp(`<svg[^>]*width="15"[^>]*class="[^"]*lucide-${icon} `));
  }
});
