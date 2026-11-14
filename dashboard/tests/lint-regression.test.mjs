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
import { renderPage } from '../seo/render.js';
import { legalPages } from '../seo/legal.js';
import { buildPages } from '../seo/pages.js';

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

// Intentional policy change: the earlier lint-only characterization pinned always-on
// analytics. The SDK cannot cancel pending sends safely, so optional analytics is
// now unavailable even after a saved grant; this is not a relaxed expectation.
test('optional categories stay disabled across all saved choices and storage failures', async () => {
  for (const blocked of [false, true]) {
    for (const saved of [null, '{broken', '{"v":1,"analytics":true,"marketing":true}', '{"v":2,"analytics":true,"marketing":true}', '{"v":1,"analytics":false,"marketing":false}']) {
      const f = consentFixture({ country: 'US', blocked });
      if (saved) f.values.set('openshorts_consent', saved);
      f.values.set('session_token', 'fixture-session');
      f.values.set('media_token', 'fixture-media');
      const before = [...f.values];
      assert.deepEqual(plain(f.api.getConsent()), { necessary: true, analytics: false, marketing: false });
      assert.equal(f.api.applyConsent(), false);
      await settle();
      assert.equal(f.starts(), 0);
      assert.deepEqual(f.requests, []);
      assert.deepEqual([...f.values], before, 'no consent or operational storage changes');
    }
  }
});

test('track, identify, and reset cannot queue or send analytics', async () => {
  const { outputFiles } = await build({ entryPoints: ['src/lib/analytics.js'], bundle: true, write: false, platform: 'node', format: 'cjs' });
  const queued = [];
  const values = new Map([['openshorts_consent', '{"v":1,"analytics":true,"marketing":true}']]);
  const op = (...args) => queued.push(args);
  op.q = queued;
  const context = vm.createContext({ module: { exports: {} }, window: { op }, localStorage: { getItem: (key) => values.get(key) ?? null }, fetch: (...args) => queued.push(args) });
  vm.runInContext(outputFiles[0].text, context);
  for (const saved of [values.get('openshorts_consent'), null, '{broken']) {
    values.set('openshorts_consent', saved);
    context.module.exports.track('Signup', { props: { source: 'fixture' } });
    context.module.exports.identify({ id: 'fixture-account' }, { plan: 'fixture' });
    context.module.exports.reset();
  }
  assert.deepEqual(queued, []);
});

test('attribution never reads stale storage or submits data, including when storage throws', async () => {
  const { outputFiles } = await build({ entryPoints: ['src/lib/attribution.js'], bundle: true, write: false, platform: 'node', format: 'cjs' });
  for (const blocked of [false, true]) {
    const accesses = [];
    const posts = [];
    const stored = new Map([['os_attrib', '{"referrer":"https://example.test","landing_path":"/?campaign=fixture"}'], ['os_attrib_sent', '0'], ['session_token', 'fixture-session']]);
    const before = [...stored];
    const context = vm.createContext({ module: { exports: {} },
      localStorage: { getItem(key) { accesses.push(key); if (blocked) throw new Error('blocked'); return stored.get(key); }, setItem(key, value) { accesses.push(key); stored.set(key, value); } },
    });
    vm.runInContext(outputFiles[0].text, context);
    context.module.exports.capture();
    assert.deepEqual(plain(context.module.exports.firstTouchProps()), {});
    // This awaited callback is the one AuthContext uses after sign-in; it must resolve.
    await context.module.exports.report(async (...args) => posts.push(args));
    assert.deepEqual(posts, []);
    assert.deepEqual(accesses, []);
    assert.deepEqual([...stored], before);
  }
});

test('SPA and rendered static pages have no analytics loader even with configuration', async () => {
  const pages = [await readFile('index.html', 'utf8'), ...legalPages().map((page) => renderPage(page))];
  for (const html of pages) {
    const appended = [];
    const storageAccesses = [];
    const values = new Map([['openshorts_consent', '{"v":1,"analytics":true,"marketing":true}']]);
    const context = vm.createContext({ window: {}, location: new URL('https://openshorts.app/privacy'), URLSearchParams,
      localStorage: { getItem: (key) => { storageAccesses.push(key); return values.get(key) ?? null; }, setItem: (key, value) => { storageAccesses.push(key); values.set(key, value); } },
      document: { referrer: '', querySelector: () => null, createElement: () => ({}), head: { appendChild: (node) => appended.push(node) } } });
    for (const [, script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) {
      vm.runInContext(script.replaceAll('%VITE_OPENPANEL_API_URL%', 'https://analytics.example').replaceAll('%VITE_OPENPANEL_CLIENT_ID%', 'fixture-client'), context);
    }
    assert.equal(context.window.op, undefined);
    assert.equal(context.window.__osAnalyticsInit, undefined);
    assert.deepEqual(appended, []);
    assert.deepEqual(storageAccesses, [], 'static pages must not capture attribution');
    assert.doesNotMatch(html, /src=["'](?:https?:[^"']*)?\/op1\.js/);
  }
});

test('static CTA handlers and tool events cannot use a stale analytics stub', async () => {
  const queued = [];
  const context = vm.createContext({ window: { op: (...args) => queued.push(args) }, module: { exports: {} } });
  for (const page of buildPages()) {
    const html = renderPage(page);
    for (const [, handler] of html.matchAll(/onclick="([^"]*window\.op[^"]*)"/g)) {
      vm.runInContext(handler.replaceAll('&amp;', '&').replaceAll('&#39;', "'"), context);
    }
  }
  const { outputFiles } = await build({ entryPoints: ['tools/common.js'], bundle: true, write: false, format: 'cjs', define: { 'import.meta.env': '{}' } });
  vm.runInContext(outputFiles[0].text, context);
  context.module.exports.track('tool_cta_click', { tool: 'fixture' });
  assert.deepEqual(queued, []);
});

test('English and Spanish privacy pages explain analytics is unavailable', () => {
  const pages = legalPages();
  const en = renderPage(pages.find((page) => page.path === '/privacy'));
  const es = renderPage(pages.find((page) => page.path === '/privacidad'));
  assert.match(en, /Optional browser analytics is disabled/);
  assert.match(es, /La analítica opcional del navegador está desactivada/);
  for (const html of [en, es]) assert.doesNotMatch(html, /exempt.*prior consent|exenta de\s+consentimiento|Audience measurement — always on|Medición de audiencia — siempre activa/);
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
