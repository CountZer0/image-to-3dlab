// --- Generate mode -------------------------------------------------------------------
import { JobProgressPanel, formatDuration } from '../components/job-progress.js';
import { presentTrellisAdvice } from '../components/trellis-input-advice.js';
import { alphaBadge } from '../components/alpha-badge.js';

// Backend metadata (stages, labels, requires_alpha) is server-owned truth (viewer/generate_api.py
// BACKENDS registry) -- fetched once so the frontend never hardcodes a second copy that can drift.
let backendMeta = {
  trellis: {
    requires_alpha: true,
    stages: ['load', 'sparse_structure', 'shape_slat_coarse', 'shape_slat_fine', 'texture_slat', 'decode', 'bake'],
    stage_labels: { load: 'Load', sparse_structure: 'Sparse structure', shape_slat_coarse: 'Shape SLat coarse',
      shape_slat_fine: 'Shape SLat fine', texture_slat: 'Texture SLat', decode: 'Decode', bake: 'Bake / remesh' },
  },
}; // placeholder until /api/backends resolves; keeps the page usable if that fetch is slow/fails
const jobProgress = new JobProgressPanel({
  stages: document.getElementById('generate-stages'),
  bar: document.getElementById('generate-overall-bar'),
  label: document.getElementById('generate-overall-label'),
  eta: document.getElementById('generate-overall-eta'),
});
function buildStageRows(backendId) {
  const meta = backendMeta[backendId] || { stages: ['running'], stage_labels: { running: 'Running' } };
  jobProgress.configure(meta);
}
buildStageRows('trellis');
const gen = {
  file: null, objectUrl: null, hasAlpha: false, jobId: null, source: null,
  running: false, outputDir: null, pollTimer: null, adviceRequest: 0,
};
const setupState = { ready: false };
function currentBackend() { return g('generate-backend').value; }
// Hide the controls the server says this machine's route ignores (e.g. the Mac-only
// attention and rembg options on NVIDIA). A control is hidden with its wrapping row.
function applyHiddenFields() {
  const hidden = new Set(backendMeta[currentBackend()]?.hidden_fields || []);
  for (const el of document.querySelectorAll('.backend-fields [id]')) {
    const row = el.closest('.field, label.check');
    if (row && (el.tagName === 'SELECT' || el.tagName === 'INPUT')) row.hidden = hidden.has(el.id);
  }
}
function backendRequiresAlpha() {
  const meta = backendMeta[currentBackend()];
  return meta ? meta.requires_alpha : true; // fail conservative if metadata hasn't loaded yet
}
const g = (id) => document.getElementById(id);
const updateGenerateButton = () => {
  const needsRembg = backendRequiresAlpha() && !gen.hasAlpha && !g('generate-rembg').checked;
  g('generate-submit').disabled = !gen.file || gen.running || needsRembg || !setupState.ready;
};
async function inspectAlpha(file) {
  // JPEG and opaque formats cannot carry a useful foreground alpha. For PNG/WebP, inspect the
  // decoded pixels in-browser so the guardrail is visible before a long job is submitted.
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const pixels = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    let min = 255;
    for (let i = 3; i < pixels.length; i += 4) { if (pixels[i] < min) min = pixels[i]; if (min === 0) break; }
    bitmap.close();
    return file.type === 'image/png' || file.type === 'image/webp' ? min < 255 : false;
  } catch (_) { return false; }
}
function renderAlphaBadge() {
  const badge = g('generate-alpha');
  const { tone, text } = alphaBadge(gen.hasAlpha, backendRequiresAlpha());
  badge.className = `alpha-badge ${tone}`;
  badge.textContent = text;
}
function resetTrellisAdvice() {
  gen.adviceRequest += 1;
  const result = g('trellis-input-result');
  result.hidden = true;
  result.className = '';
  result.textContent = '';
}
async function inspectTrellisInput(file) {
  resetTrellisAdvice();
  if (currentBackend() !== 'trellis') return;
  const request = gen.adviceRequest;
  const result = g('trellis-input-result');
  result.hidden = false;
  result.className = 'advice-uncertain';
  result.textContent = 'TinyCLIP is checking image style and lighting… First use may download ~94 MB.';
  const body = new FormData(); body.append('image', file);
  try {
    const response = await fetch('/api/trellis/input-advice', { method: 'POST', body });
    const payload = await response.json();
    if (request !== gen.adviceRequest || gen.file !== file || currentBackend() !== 'trellis') return;
    if (!response.ok) throw new Error(payload.error || `request failed (${response.status})`);
    const advice = presentTrellisAdvice(payload);
    result.className = `advice-${advice.tone}`;
    result.textContent = advice.message;
  } catch (error) {
    if (request !== gen.adviceRequest || gen.file !== file || currentBackend() !== 'trellis') return;
    result.className = 'advice-uncertain';
    result.textContent = `Automatic check unavailable (${error.message || error}). Please eyeball the input; generation is still allowed.`;
  }
}
function setGenerateFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  if (gen.objectUrl) URL.revokeObjectURL(gen.objectUrl);
  gen.file = file; gen.objectUrl = URL.createObjectURL(file); gen.hasAlpha = false;
  g('generate-preview').src = gen.objectUrl;
  g('generate-drop').classList.add('has-image');
  const badge = g('generate-alpha'); badge.hidden = false; badge.className = 'alpha-badge';
  badge.textContent = 'checking alpha…';
  inspectAlpha(file).then((hasAlpha) => {
    if (gen.file !== file) return;
    gen.hasAlpha = hasAlpha;
    renderAlphaBadge();
    updateGenerateButton();
  });
  inspectTrellisInput(file);
  updateGenerateButton();
  g('generate-status').textContent = '';
}
const generateFile = g('generate-file');
const generateDrop = g('generate-drop');
generateDrop.onclick = () => generateFile.click();
generateFile.onchange = () => { if (generateFile.files[0]) setGenerateFile(generateFile.files[0]); generateFile.value = ''; };
for (const event of ['dragenter', 'dragover']) generateDrop.addEventListener(event, (e) => { e.preventDefault(); generateDrop.classList.add('drag'); });
for (const event of ['dragleave', 'drop']) generateDrop.addEventListener(event, (e) => { e.preventDefault(); generateDrop.classList.remove('drag'); });
generateDrop.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) setGenerateFile(e.dataTransfer.files[0]); });
g('generate-rembg').onchange = updateGenerateButton;

function applyGenerateProgress(event) {
  const phase = event.phase;
  jobProgress.apply(event);
  if (phase === 'done') {
    g('generate-overall-bar').style.width = '100%';
    g('generate-overall-label').textContent = 'Generation complete';
    g('generate-overall-eta').textContent = '';
    g('generate-status').textContent = 'Done — the generated GLB is ready below.';
    const durationEl = g('generate-duration');
    durationEl.style.display = 'block';
    durationEl.textContent = `Generated in ${formatDuration(event.elapsed_seconds)}`;
    g('generate-progress-box').style.display = 'none';
    g('generate-viewer').style.display = 'block';
    g('generate-downloads').style.display = 'flex';
    g('generate-summary').style.display = 'block';
    const src = `${event.result_url}?t=${Date.now()}`;
    // No add-pane inside the embedded preview -- a Compare-capable iframe crammed into this
    // sidebar-sized panel makes every pane too small to see once a second model is added.
    // Full multi-model comparison belongs in the dedicated Compare tab, which gets the whole
    // screen.
    g('generate-frame').src = `/viewer/index.html?a=${encodeURIComponent(src)}&la=Generated&restricted=1`;
    g('generate-glb').href = event.result_url;
    const savedNote = g('generate-saved-note');
    if (gen.outputDir) {
      savedNote.style.display = 'block';
      savedNote.textContent = `Already saved to ${gen.outputDir}/ — these buttons are just an extra browser-download option.`;
    }
    g('generate-manifest').hidden = !event.manifest_url;
    if (event.manifest_url) {
      g('generate-manifest').href = event.manifest_url;
      fetch(event.manifest_url).then((response) => response.json()).then((manifest) => {
        // Manifest shape differs per backend (TRELLIS: run_stages_1_3/to_glb; Hunyuan:
        // shape/remesh/paint; SF3D writes none at all) -- render whatever stage timings exist
        // generically rather than hardcoding one backend's field names.
        const timings = manifest.timings_seconds || {};
        const parts = Object.entries(timings)
          .filter(([name]) => name !== 'total')
          .map(([name, value]) => `${name} ${formatDuration(value)}`);
        g('generate-summary').textContent = `${manifest.backend || currentBackend()} manifest · ` +
          `total ${formatDuration(timings.total)}` + (parts.length ? ` · ${parts.join(' · ')}` : '');
      }).catch(() => {});
    } else {
      g('generate-summary').textContent = '';
    }
  } else if (phase === 'error') {
    g('generate-status').textContent = event.message || 'Generation failed';
  }
}
function stopStatusPoll() { if (gen.pollTimer) { clearInterval(gen.pollTimer); gen.pollTimer = null; } }
// Fallback for when the SSE stream (source.onerror, below) drops and EventSource's own
// auto-reconnect doesn't recover in time -- e.g. a sub-minute SF3D run that finished
// server-side while the browser's connection was down, with no other way to find out.
// Polls the same single-shot snapshot the stream itself is built from, so a job that
// already finished is discovered on the very next tick instead of leaving the progress
// bar stuck indefinitely.
function startStatusPoll(jobId) {
  if (gen.pollTimer) return; // already polling
  gen.pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/generate/${jobId}/status`);
      if (!res.ok) return; // server restarted and lost this job; keep waiting/reconnecting
      const data = await res.json();
      if (data.last_event) applyGenerateProgress(data.last_event);
      if (data.status === 'done' || data.status === 'error' || data.status === 'cancelled') {
        gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
        stopGenerateStream();
      }
    } catch (e) { /* still offline; next tick will retry */ }
  }, 4000);
}
function stopGenerateStream() {
  if (gen.source) { gen.source.close(); gen.source = null; }
  stopStatusPoll();
}
function startGenerateStream(jobId) {
  stopGenerateStream();
  const source = new EventSource(`/api/generate/${jobId}/events`); gen.source = source;
  source.onmessage = (message) => {
    const event = JSON.parse(message.data); applyGenerateProgress(event);
    if (event.phase === 'done' || event.phase === 'error') {
      gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
      source.close();
    }
  };
  source.onopen = () => {
    stopStatusPoll();
    if (gen.running) g('generate-status').textContent = 'Generating…';
  };
  source.onerror = () => {
    if (gen.running) {
      g('generate-status').textContent = 'Progress connection lost; reconnecting…';
      startStatusPoll(jobId);
    }
  };
}
g('generate-submit').onclick = async () => {
  if (!gen.file || gen.running) return;
  gen.running = true; jobProgress.reset();
  g('generate-progress-box').style.display = 'block'; g('generate-viewer').style.display = 'none';
  g('generate-downloads').style.display = 'none'; g('generate-summary').style.display = 'none';
  g('generate-duration').style.display = 'none';
  g('generate-status').textContent = 'Submitting job…';
  g('generate-cancel').style.display = 'block'; updateGenerateButton();
  const backendId = currentBackend();
  const perBackendSettings = {
    trellis: () => ({
      resolution: g('generate-resolution').value,
      seed: Number(g('generate-seed').value),
      decimation_target: Number(g('generate-decimation').value),
      texture_size: Number(g('generate-texture').value),
      allow_rembg: g('generate-rembg').checked,
      sparse_attn_backend: g('generate-attention').value,
    }),
    sf3d: () => ({
      texture_resolution: Number(g('sf3d-texture').value),
      foreground_ratio: Number(g('sf3d-foreground').value),
      remesh: g('sf3d-remesh').value,
      target_vertices: Number(g('sf3d-vertices').value),
    }),
    'hunyuan-mlx': () => ({
      octree_resolution: Number(g('hunyuan-octree').value),
      seed: Number(g('hunyuan-seed').value),
      decimation_target: Number(g('hunyuan-decimation').value),
      paint_seed: Number(g('hunyuan-paint-seed').value),
      paint_res: Number(g('hunyuan-paint-res').value),
      paint_steps: Number(g('hunyuan-paint-steps').value),
      paint_tex: Number(g('hunyuan-paint-tex').value),
    }),
    pixal3d: () => ({
      res: Number(g('pixal3d-res').value),
      seed: Number(g('pixal3d-seed').value),
      fov: Number(g('pixal3d-fov').value),
      steps: g('pixal3d-steps').value === 'auto' ? 'auto' : Number(g('pixal3d-steps').value),
    }),
    'hunyuan-mlx-xiong': () => ({
      model: g('xiong-model').value,
      octree_resolution: Number(g('xiong-octree').value),
      seed: Number(g('xiong-seed').value),
      quantize: Number(g('xiong-quantize').value),
      decimation_target: Number(g('xiong-decimation').value),
      paint_seed: Number(g('xiong-paint-seed').value),
      paint_res: Number(g('xiong-paint-res').value),
      paint_steps: Number(g('xiong-paint-steps').value),
      paint_tex: Number(g('xiong-paint-tex').value),
    }),
  };
  const settings = {
    backend: backendId,
    output_dir: g('generate-output-dir').value.trim() || undefined,
    output_name: g('generate-output-name').value.trim() || undefined,
    debug: g('generate-debug').checked,
    ...perBackendSettings[backendId](),
  };
  const body = new FormData(); body.append('image', gen.file); body.append('settings', JSON.stringify(settings));
  try {
    const response = await fetch('/api/generate', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `request failed (${response.status})`);
    gen.jobId = payload.job_id; gen.outputDir = payload.output_dir;
    g('generate-status').textContent = 'Job started. MPS is serialized; this may take a while.';
    startGenerateStream(gen.jobId);
  } catch (error) {
    gen.running = false; g('generate-cancel').style.display = 'none'; updateGenerateButton();
    g('generate-status').textContent = error.message || String(error);
  }
};
g('generate-cancel').onclick = async () => {
  if (!gen.jobId) return;
  try { await fetch(`/api/generate/${gen.jobId}/cancel`, { method: 'POST' }); }
  catch (_) { /* the SSE stream will surface the process outcome */ }
};

async function refreshSetup() {
  const el = g('setup-checks');
  const backendId = currentBackend();
  try {
    const res = await fetch('/api/setup?backend=' + encodeURIComponent(backendId));
    const s = await res.json();
    setupState.ready = !!s.ready;
    const rows = [];
    const buildLabel = backendMeta[backendId]?.label || backendId;
    if (s.build && s.build.present) {
      const interp = s.build.interpreter ? '<code>' + s.build.interpreter + '</code>' : '';
      rows.push('<div class="setup-check"><span class="ok">✓</span><span>' + buildLabel + ' build</span>' + interp + '</div>');
    } else {
      rows.push('<div class="setup-check"><span class="bad">✗</span><span>' + buildLabel + ' build missing</span></div>');
      if (s.build && s.build.hint) rows.push('<div class="setup-check"><span class="hint">→</span><span class="hint">' + s.build.hint + '</span></div>');
    }
    for (const repo of Object.values(s.weights || {})) {
      rows.push(repo.present
        ? '<div class="setup-check"><span class="ok">✓</span><span>' + repo.label + '</span><code>' + repo.human + ' on disk</code></div>'
        : '<div class="setup-check"><span class="warn">⚠</span><span>' + repo.label + '</span><code>not on disk — first use downloads</code></div>');
    }
    // The mlx attention backend is optional: generation works without it on the stock
    // sdpa path. But offering it when the checkout is unpatched or mlx is missing would
    // crash a run partway through, so the options are disabled rather than left to fail.
    // Same treatment the mlx attention options get: offering a model whose weights are
    // absent lets a run fail minutes in, so it is disabled and labelled instead. Only the
    // default route is downloaded now, so this is the ordinary case, not a broken install.
    if (backendId === 'hunyuan-mlx-xiong' && s.model_availability) {
      const select = g('xiong-model');
      if (select) {
        for (const option of select.querySelectorAll('option')) {
          const present = s.model_availability[option.value] !== false;
          option.disabled = !present;
          const suffix = ' — not downloaded';
          option.textContent = option.textContent.replace(suffix, '') + (present ? '' : suffix);
        }
        if (select.selectedOptions[0]?.disabled) {
          const first = [...select.options].find((o) => !o.disabled);
          if (first) select.value = first.value;
        }
      }
    }
    if (backendId === 'trellis' && s.mlx_attention) { // Mac only; NVIDIA reports none
      const mlx = s.mlx_attention;
      const sel = g('generate-attention');
      if (sel) {
        for (const opt of sel.querySelectorAll('option')) {
          if (opt.value.startsWith('mlx')) opt.disabled = !mlx.ready;
        }
        if (!mlx.ready && sel.value.startsWith('mlx')) sel.value = 'sdpa';
      }
      rows.push(mlx.ready
        ? '<div class="setup-check"><span class="ok">✓</span><span>MLX fused attention</span><code>available — pick it under Attention backend</code></div>'
        : '<div class="setup-check"><span class="warn">⚠</span><span>MLX fused attention</span><code>optional; stock sdpa still works</code></div>');
      if (!mlx.ready && mlx.hint) {
        rows.push('<div class="setup-check"><span class="hint">→</span><span class="hint">' + mlx.hint + '</span></div>');
      }
    }
    // One line, not a list of checks. The full picture lives on Setup & Status; here the
    // only question is whether this backend can run right now.
    const missing = Object.values(s.weights || {}).filter((w) => !w.present).length;
    setHealth(
      setupState.ready && !missing ? 'ok' : (setupState.ready ? 'warn' : 'bad'),
      setupState.ready && !missing
        ? `${buildLabel} is ready`
        : setupState.ready
          ? `${buildLabel} is ready, but ${missing} weight set${missing === 1 ? '' : 's'} `
            + 'would download on first use'
          : `${buildLabel} is not installed yet`,
      !(setupState.ready && !missing),
    );
  } catch (e) {
    setHealth('bad', `Could not read machine status: ${e}`, true);
  }
  updateGenerateButton();
}

function setHealth(level, text, showLink) {
  const dot = g('health-dot');
  dot.className = `health-dot ${level}`;
  dot.textContent = level === 'ok' ? '●' : (level === 'warn' ? '◐' : '○');
  g('health-text').textContent = text;
  g('health-goto').hidden = !showLink;
  // A badge rather than a forced redirect: the user may have asked not to land on the
  // setup page, and overriding that because something needs attention would ignore them.
  document.getElementById('mode-setup')?.classList.toggle('needs-attention', level !== 'ok');
}
async function loadBackendMeta() {
  try {
    const res = await fetch('/api/backends');
    const data = await res.json();
    const merged = {};
    for (const b of data.backends) merged[b.id] = b;
    backendMeta = merged;
  } catch (e) { /* keep the trellis-only placeholder; page stays usable */ }
  buildStageRows(currentBackend());
  applyHiddenFields();
  refreshSetup();
}
g('generate-backend').onchange = () => {
  const backendId = currentBackend();
  for (const block of document.querySelectorAll('.backend-fields')) {
    block.hidden = block.dataset.backend !== backendId;
  }
  buildStageRows(backendId);
  applyHiddenFields();
  jobProgress.reset();
  if (gen.file) renderAlphaBadge(); // the image is unchanged; only the wording depends on the backend
  updateGenerateButton();
  g('trellis-input-guidance').hidden = backendId !== 'trellis';
  resetTrellisAdvice();
  if (backendId === 'trellis' && gen.file) inspectTrellisInput(gen.file);
  refreshSetup();
};
g('health-goto').onclick = () => document.getElementById('mode-setup').click();

loadBackendMeta();
