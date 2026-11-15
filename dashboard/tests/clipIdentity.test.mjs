import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8');
const scheduler = readFileSync(new URL('../src/components/ScheduleWeekModal.jsx', import.meta.url), 'utf8');

// Execute the actual component callbacks without a DOM or backend. API responses
// and React state setters below are test fixtures; this is not browser coverage.
function callback(source, name, end, bindings) {
  const start = source.indexOf(`const ${name} = `);
  assert.notEqual(start, -1, `missing ${name}`);
  const stop = source.indexOf(end, start);
  assert.notEqual(stop, -1, `missing end of ${name}`);
  return new Function(...Object.keys(bindings), `${source.slice(start, stop)}; return ${name};`)(...Object.values(bindings));
}

const filtered = [{ clip_index: 1, video_url: '/videos/original-1.mp4', video_title_for_youtube_short: 'Original one' }];
const noop = () => {};

test('ranked cards retain original identity after the first clip is filtered out', () => {
  const ranked = callback(app, 'rankedClips', '  // Bulk subtitles:', {
    results: { clips: filtered }, useMemo: (fn) => fn(),
  });
  assert.equal(ranked[0].index, 1);
});

test('ranking preserves explicit zero after sorting', () => {
  const ranked = callback(app, 'rankedClips', '  // Bulk subtitles:', {
    results: { clips: [{ clip_index: 1, predicted_score: 1 }, { clip_index: 0, predicted_score: 9 }] }, useMemo: (fn) => fn(),
  });
  assert.deepEqual(ranked.map(({ index }) => index), [0, 1]);
});

test('legacy responses retain array identity after sorting', () => {
  const ranked = callback(app, 'rankedClips', '  // Bulk subtitles:', {
    results: { clips: [{ predicted_score: 1 }, { predicted_score: 9 }] }, useMemo: (fn) => fn(),
  });
  assert.deepEqual(ranked.map(({ index }) => index), [1, 0]);
});

test('bulk subtitles pair the original API identity with the delivered file', async () => {
  const requests = [];
  const run = callback(app, 'handleBulkSubtitles', '  const handleDownloadAll', {
    results: { clips: filtered }, jobId: 'fixture-job', setBulkSub: noop, refreshMe: noop,
    pollJob: async () => ({}), setResults: noop,
    apiFetch: async (url, options) => { requests.push([url, JSON.parse(options.body)]); return { ok: true }; },
  });
  await run({});
  assert.equal(requests[0][0], '/api/subtitle');
  assert.equal(requests[0][1].clip_index, 1);
  assert.equal(requests[0][1].input_filename, 'original-1.mp4');
});

test('rerender updates the delivered clip by original identity', () => {
  let results = { clips: filtered };
  const run = callback(app, 'handleClipRerendered', '  // Reopen an archived project', {
    setResults: (fn) => { results = fn(results); }, setProjectState: noop,
    setDurableClips: noop, handleClipStateChange: noop,
  });
  run(1, { new_video_url: '/videos/recut-1.mp4', start: 12, end: 30 });
  assert.equal(results.clips.length, 1);
  assert.equal(results.clips[0].video_url, '/videos/recut-1.mp4');
  assert.equal(results.clips[0].clip_index, 1);
});

for (const editor of ['ClipEditor', 'ReframeEditor']) {
  test(`${editor} opens the filtered clip with its original title`, () => {
    const state = app.slice(app.indexOf('  const [editingClip,'), app.indexOf('  // Silent-success'));
    const guard = app.match(new RegExp(`\\{([^\\n]+) && \\(\\n        <${editor}`))[1];
    const title = app.slice(app.indexOf(`<${editor}\n`)).match(/clipTitle=\{([^\n]+)\}/)[1];
    const bindings = {
      results: { clips: filtered }, rankedClips: [{ clip: filtered[0], index: 1 }],
      useState: () => [1, noop],
    };
    const evaluate = (expression) => new Function(...Object.keys(bindings), `${state}; return ${expression};`)(...Object.values(bindings));
    assert.ok(evaluate(guard), 'editor must open for original index 1 in a one-clip response');
    assert.equal(evaluate(title), 'Original one');
  });
}

test('schedule payload uses original identity while retaining display position', async () => {
  const requests = [];
  const progress = [];
  const run = callback(scheduler, 'handleScheduleAll', '    const successCount', {
    canPost: true, selectedPlatforms: ['youtube'], setScheduling: noop, setDone: noop,
    schedule: [{ clip: filtered[0], index: 0, date: new Date(2026, 0, 1) }],
    setProgress: (value) => progress.push(value), time: '12:00', timezone: 'UTC',
    jobId: 'fixture-job', uploadPostKey: '', uploadUserId: 'fixture-user',
    apiFetch: async (url, options) => { requests.push([url, JSON.parse(options.body)]); return { ok: true }; },
  });
  await run();
  assert.equal(requests[0][0], '/api/social/post');
  assert.equal(requests[0][1].clip_index, 1);
  assert.deepEqual(progress.at(-1).results, [{ index: 0, success: true }]);
});
