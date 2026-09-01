let isGenerating = false;
let currentBubble = null;
// The conversation actually sent to the engine. The transcript is only a rendering of it --
// clearing the transcript without clearing this would show an empty screen while the model
// still remembered everything.
let conversation = [];
let telemetryTimer = null;
// Aborts the in-flight SSE reader. POSTing /api/stop only tells the engine to stop; without
// this the fetch loop kept appending tokens until the server closed the stream, and the
// STOPPED badge was then overwritten by the normal completion path.
let currentAbort = null;
// The model name /v1/chat/completions requires. It is `--model-id` on the server and was
// hardcoded here, so any custom id made every message 404 while the HUD still read READY.
let serverModelId = 'gemma-4-26b-a4b-it';
// Wall-clock deadline during which the telemetry poll must not rewrite the status badge.
// ERROR and STOPPED were previously unreadable: the 1.5s poll reset them to READY.
let statusHoldUntil = 0;
let configDebounce = null;
// The active model as /api/models describes it. Every architecture figure in the sidebar is
// read from here; they used to be literals in index.html that no code ever wrote.
let activeModel = null;
// The engine's resolved context window, and the exact prompt-token count the last response
// reported. Before any response has landed the readout falls back to a character estimate.
let liveContext = 0;
let lastPromptTokens = null;
// Populated from GET /api/server_info; `max_stop` bounds the stop-sequence box.
let maxStopSequences = 4;
// Server environment for the download flow, and the poll handle while one is running.
let hostEnv = null;
let downloadTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    initHeatmapGrid();
    restoreSystemPrompt();
    hydrateServerInfo();
    resumeDownloadWatch();
    hydrateModelId();
    refreshModelList();
    hydrateConfig();
    fetchTelemetry();
    telemetryTimer = setInterval(fetchTelemetry, 1500);
});

// Single writer for the status badge. `holdMs` reserves the badge against the telemetry
// poll, which runs every 1.5s and otherwise erases any transient state instantly.
function setStatus(text, color, holdMs) {
    const el = document.getElementById('hud-status');
    if (!el) return;
    el.innerText = text;
    el.style.color = color;
    statusHoldUntil = holdMs ? Date.now() + holdMs : 0;
}

function statusHeld() {
    return Date.now() < statusHoldUntil;
}

// Fixed-at-launch settings. None of these was visible anywhere in the GUI, so a 404 from a
// mismatched model id or a 429 from a full queue had no explanation on screen.
function hydrateServerInfo() {
    fetch('/api/server_info')
        .then(r => r.json())
        .then(d => {
            const sv = d && d.server;
            if (!sv) return;
            const set = (id, text) => {
                const el = document.getElementById(id);
                if (el) el.innerText = text;
            };
            set('srv-version', sv.version || '\u2014');
            set('srv-address', `${sv.host}:${sv.port}`);
            set('srv-model-id', sv.model_id || '\u2014');
            set('srv-queue', sv.queue_limit != null ? String(sv.queue_limit) : '\u2014');
            set('srv-context-max', sv.context_max != null
                ? `${sv.context_max} tokens` : '\u2014');
            set('srv-gpu', sv.gpu_name || '\u2014');

            const warn = document.getElementById('srv-lan-warn');
            if (warn) warn.style.display = sv.lan_accessible ? '' : 'none';

            hostEnv = sv;
            renderDownloadPrereqs();

            if (sv.max_stop_sequences) {
                maxStopSequences = sv.max_stop_sequences;
                const limit = document.getElementById('stop-limit');
                if (limit) limit.innerText = String(maxStopSequences);
            }
        })
        .catch(err => console.warn('Server info fetch failed:', err));
}

// Says up front whether a download can run, and why not when it cannot. Without this the
// button would start, fail several seconds later inside a subprocess, and report something
// the user has no way to act on.
function renderDownloadPrereqs() {
    const box = document.getElementById('download-prereqs');
    if (!box || !hostEnv) return;

    const needGb = hostEnv.download_needs_gb || 29;
    const freeGb = hostEnv.free_disk_gb || 0;
    const checks = [
        {
            ok: !!hostEnv.python_available,
            label: hostEnv.python_available
                ? `Python ${hostEnv.python_version}`
                : 'Python 3.10+ not found on PATH',
        },
        {
            ok: !!hostEnv.converter_present,
            label: hostEnv.converter_present
                ? 'Converter script found'
                : 'tools/convert_hf_to_gturbo.py not found',
        },
        {
            ok: freeGb >= needGb,
            label: `${freeGb.toFixed(1)} GB free (about ${needGb} GB needed)`,
        },
    ];

    box.innerHTML = '';
    for (const c of checks) {
        const row = document.createElement('div');
        row.className = 'prereq' + (c.ok ? ' ok' : ' bad');
        row.innerText = (c.ok ? '\u2713  ' : '\u2717  ') + c.label;
        box.appendChild(row);
    }

    // Disk space is a warning, not a block: the figure is for the whole conversion and a
    // resumed run may need far less, so the decision stays with the user.
    const blocked = !hostEnv.python_available || !hostEnv.converter_present;
    const btn = document.getElementById('dl-start');
    if (btn) {
        btn.disabled = blocked;
        btn.title = blocked
            ? 'Install Python 3.10+ and keep tools/ beside the executable.'
            : 'Stream the pinned checkpoint and repack it into a .gturbo bundle';
    }
}

function startDownload() {
    const output = document.getElementById('dl-output').value.trim();
    const token = document.getElementById('dl-token').value;
    const resume = document.getElementById('dl-resume').checked;

    setDownloadResult('', false);
    fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ output: output, token: token, resume: resume })
    })
    .then(async r => {
        const data = await r.json().catch(() => null);
        if (!r.ok || !data || data.status !== 'SUCCESS') {
            setDownloadResult((data && data.message) || `HTTP ${r.status}`, true);
            return;
        }
        // The token has done its job. Keeping it in a live DOM node for the hours a
        // conversion runs serves no purpose.
        document.getElementById('dl-token').value = '';
        showDownloadRunning(true);
        pollDownload();
        downloadTimer = setInterval(pollDownload, 1000);
    })
    .catch(err => setDownloadResult('Could not reach the engine: ' + err, true));
}

function cancelDownload() {
    fetch('/api/download/cancel', { method: 'POST' })
        .then(() => pollDownload())
        .catch(err => setDownloadResult('Cancel failed: ' + err, true));
}

function pollDownload() {
    fetch('/api/download/status')
        .then(r => r.json())
        .then(d => {
            const dl = d && d.download;
            if (!dl) return;
            renderDownloadProgress(dl);

            if (dl.state === 'running') return;

            // Terminal state: stop polling and settle the UI.
            if (downloadTimer) {
                clearInterval(downloadTimer);
                downloadTimer = null;
            }
            showDownloadRunning(false);

            if (dl.state === 'done') {
                setDownloadResult(`${dl.output} is ready. Select it above and press Load Model.`,
                                  false);
                refreshModelList();
            } else if (dl.state === 'failed') {
                // The converter's own tail is the only useful detail when it dies without
                // reporting a reason.
                const tail = (dl.log || []).slice(-3).join(' / ');
                setDownloadResult(dl.message + (tail ? ` \u2014 ${tail}` : ''), true);
            } else if (dl.state === 'cancelled') {
                setDownloadResult(dl.message, false);
            }
        })
        .catch(err => console.warn('Download status poll failed:', err));
}

function renderDownloadProgress(dl) {
    const box = document.getElementById('dl-progress');
    if (box) box.style.display = (dl.state === 'running') ? '' : 'none';
    if (dl.state !== 'running') return;

    const stage = document.getElementById('dl-stage');
    if (stage) {
        stage.innerText = dl.steps
            ? `Step ${dl.step} of ${dl.steps}: ${dl.stage}`
            : (dl.stage || 'Working...');
    }

    const fill = document.getElementById('dl-fill');
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, dl.pct || 0)).toFixed(1)}%`;

    const meta = document.getElementById('dl-meta');
    if (meta) {
        const parts = [];
        if (dl.bytes_total) {
            parts.push(`${gbytes(dl.bytes_done)} / ${gbytes(dl.bytes_total)}`);
            parts.push(`${(dl.pct || 0).toFixed(1)}%`);
        }
        if (dl.rate_mbs) parts.push(`${dl.rate_mbs.toFixed(0)} MB/s`);
        if (dl.eta_s) parts.push(`ETA ${formatDuration(dl.eta_s)}`);
        meta.innerText = parts.length ? parts.join('  \u00b7  ') : dl.label || '';
    }
}

function gbytes(n) {
    if (!n) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
    if (m) return `${m}m ${String(sec).padStart(2, '0')}s`;
    return `${sec}s`;
}

function showDownloadRunning(running) {
    const start = document.getElementById('dl-start');
    const cancel = document.getElementById('dl-cancel');
    const progress = document.getElementById('dl-progress');
    if (start) start.style.display = running ? 'none' : '';
    if (cancel) cancel.style.display = running ? '' : 'none';
    if (progress) progress.style.display = running ? '' : 'none';
    // Changing the bundle name mid-run would not move the download.
    for (const id of ['dl-output', 'dl-token', 'dl-resume']) {
        const el = document.getElementById(id);
        if (el) el.disabled = running;
    }
}

function setDownloadResult(message, isError) {
    const el = document.getElementById('dl-result');
    if (!el) return;
    el.style.display = message ? '' : 'none';
    el.className = 'dl-result' + (isError ? ' error' : ' ok');
    el.innerText = message;
}

// A conversion outlives the page: it runs in the server, not the tab. Reattach to one that
// is already going rather than showing an idle panel over a running download.
function resumeDownloadWatch() {
    fetch('/api/download/status')
        .then(r => r.json())
        .then(d => {
            if (d && d.download && d.download.state === 'running') {
                showDownloadRunning(true);
                renderDownloadProgress(d.download);
                if (!downloadTimer) downloadTimer = setInterval(pollDownload, 1000);
            }
        })
        .catch(() => { /* endpoint absent or unreachable; leave the panel idle */ });
}

// Reads the model id the server actually answers to, rather than assuming the default.
function hydrateModelId() {
    fetch('/v1/models')
        .then(r => r.json())
        .then(d => {
            const first = d && d.data && d.data[0];
            if (first && first.id) serverModelId = first.id;
        })
        .catch(err => console.warn('Model id fetch failed, using default:', err));
}

// One in-page error surface. alert() blocks the whole UI and looks nothing like the rest of
// the app; several fetches previously had no failure path at all.
function reportError(message) {
    appendBanner('⚠', message, true);
}

function notify(message) {
    appendBanner('ℹ', message, false);
}

function appendBanner(icon, message, isError) {
    const transcript = document.getElementById('transcript');
    if (!transcript) return;
    const banner = document.createElement('div');
    banner.className = 'msg assistant';
    banner.innerHTML =
        '<div class="avatar"></div><div class="msg-content"><div class="bubble"></div></div>';
    banner.querySelector('.avatar').innerText = icon;
    const bubble = banner.querySelector('.bubble');
    if (isError) bubble.classList.add('error');
    bubble.innerText = message;
    transcript.appendChild(banner);
    scrollToBottom(transcript);
}

// Seeds every control from the engine's live configuration. The values in index.html are
// only placeholders: without this the sidebar showed whatever was typed into the HTML
// (context "62000", top-K 16) regardless of what the engine had actually initialized with.
function hydrateConfig() {
    fetch('/api/config')
        .then(r => r.json())
        .then(data => {
            const c = data.config;
            if (!c) return;
            setValue('cfg-temp', c.temperature);
            setValue('cfg-topp', c.top_p);
            setValue('cfg-topk', c.top_k);
            setValue('cfg-maxtoks', c.max_tokens);
            setValue('cfg-reppen', c.repetition_penalty);
            // null means "fresh seed each request"; 0 is a real seed, so the two cannot
            // share an empty-string representation on the way in either.
            const seedBox = document.getElementById('cfg-seed');
            if (seedBox) seedBox.value = (c.seed === null || c.seed === undefined) ? '' : c.seed;
            const stopBox = document.getElementById('cfg-stop');
            if (stopBox) stopBox.value = (c.stop || []).join('\n');
            if (c.max_stop) maxStopSequences = c.max_stop;
            setValue('cfg-slots', c.slots);
            setValue('cfg-eviction', c.eviction_policy);
            applyContextBounds(c.context_max, c.context_len);
            setValue('cfg-context', c.context_len);
            // The banner reflects the engine's own view of whether a reload is outstanding,
            // rather than lingering from whatever the last slider drag happened to report.
            showReloadNotice(!!c.reload_pending);
            liveContext = c.context_len_active || c.context_len || 0;
            updateContextUsage();
            applySlotBounds();
            renderConfigLabels();
            markConfigStale(false);
        })
        .catch(err => {
            // The sidebar's HTML values are placeholders. Leaving them on screen after a
            // failed hydrate presents confident numbers that the engine never confirmed.
            console.warn('Config hydrate failed:', err);
            markConfigStale(true);
        });
}

// Rebuilds the context dropdown from the ceiling the engine reports (`context_max`,
// ATTN_MAX_SPAN -- requesting more than it throws at load), so the list can never offer a
// value that fails to initialize, and it widens on its own if that kernel limit is ever
// raised. `live` is the engine's current context: it auto-sizes from installed RAM and may
// land on a value the ladder does not contain, so it is added rather than silently snapped
// to a listed one.
function applyContextBounds(maxContext, live) {
    const sel = document.getElementById('cfg-context');
    if (!sel) return;
    const ceiling = maxContext || 4096;
    const ladder = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768];
    const values = ladder.filter(v => v <= ceiling);
    if (!values.includes(ceiling)) values.push(ceiling);
    if (live && !values.includes(live) && live <= ceiling) values.push(live);
    values.sort((a, b) => a - b);

    const previous = +sel.value;
    sel.innerHTML = '';
    for (const v of values) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.innerText = `${v} Tokens`;
        sel.appendChild(opt);
    }
    // Keep whatever was selected if it survived the rebuild; otherwise fall back to the
    // engine's live value rather than to the first entry.
    sel.value = values.includes(previous) ? previous
              : (values.includes(live) ? live : values[values.length - 1]);
}

// The slots slider used to be a hardcoded min=1 max=32. Both ends were wrong: 1..8 are
// below the engine's hard floor of top_k + 1 and would throw on load, and 32 sat well under
// what the descriptor heap now allows. Drive both ends off the loaded model instead.
//
// This reads `activeModel` rather than issuing its own GET /api/models: it used to duplicate
// the request refreshModelList() was already making on the same page load.
function applySlotBounds() {
    const slider = document.getElementById('cfg-slots');
    if (!slider || !activeModel || activeModel.top_k == null || activeModel.experts == null) {
        return;
    }
    slider.min = activeModel.top_k + 1;
    slider.max = activeModel.experts;
    if (+slider.value < +slider.min) slider.value = slider.min;
    if (+slider.value > +slider.max) slider.value = slider.max;
    renderConfigLabels();
}

// Writes the Model Repository metadata rows from the bundle the engine actually opened.
// Architecture, routed top-K and quantization were hardcoded in index.html and never
// written by any code path, so they described the default model no matter what was loaded.
function updateModelMeta() {
    const m = activeModel;
    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.innerText = text;
    };
    const row = (id, shown) => {
        const el = document.getElementById(id);
        if (el && el.parentElement) el.parentElement.style.display = shown ? '' : 'none';
    };

    if (!m) {
        set('meta-id', '\u2014');
        set('meta-arch', '\u2014');
        set('meta-topk', '\u2014');
        row('meta-quant', false);
        return;
    }

    set('meta-id', m.model_id || m.name || '\u2014');
    set('meta-arch', (m.layers != null && m.experts != null)
        ? `${m.layers} Layers \u2022 ${m.experts} Experts` : '\u2014');
    set('meta-topk', m.top_k != null ? `${m.top_k} Experts / Token` : '\u2014');
    // A bundle whose manifest carries no quant block gets no row at all, rather than the
    // previous literal that would have been a guess.
    row('meta-quant', !!m.quantization);
    if (m.quantization) set('meta-quant', m.quantization);

    resizeHeatmap(m.experts);
}

// Greys the configuration panels and shows a banner when the sidebar could not be seeded.
function markConfigStale(stale) {
    for (const id of ['panel-memory', 'panel-generation']) {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('stale', stale);
    }
    const banner = document.getElementById('config-stale-notice');
    if (banner) banner.style.display = stale ? 'block' : 'none';
}

// Approximate token count for the conversation as it will next be sent. This is a rough
// 4-chars-per-token rule, NOT the tokenizer -- it is labelled as an estimate for exactly
// that reason, and is replaced by the engine's own prompt_tokens once a response reports it.
function estimateConversationTokens() {
    let chars = 0;
    const sys = document.getElementById('sys-prompt-input');
    if (sys && sys.value.trim()) chars += sys.value.trim().length;
    for (const m of conversation) chars += (m.content || '').length;
    const pending = document.getElementById('prompt-input');
    if (pending) chars += pending.value.length;
    return Math.ceil(chars / 4);
}

function updateContextUsage() {
    const el = document.getElementById('context-usage');
    if (!el) return;
    if (!liveContext) {
        el.innerText = '\u2014';
        el.classList.remove('warn', 'over');
        return;
    }

    const estimated = estimateConversationTokens();
    // Prefer the engine's own count for the history it has already seen, and estimate only
    // the part it has not.
    const used = lastPromptTokens !== null ? Math.max(lastPromptTokens, estimated) : estimated;
    const pct = (used / liveContext) * 100;
    const exact = lastPromptTokens !== null ? '' : '~';
    el.innerText = `${exact}${used} / ${liveContext} context tokens used`;
    el.classList.toggle('warn', pct >= 75 && pct < 100);
    el.classList.toggle('over', pct >= 100);
}

function setValue(id, value) {
    if (value === null || value === undefined) return;
    const el = document.getElementById(id);
    if (el) el.value = value;
}

// Shows the startup model-load failure once, as a banner above the transcript.
let loadErrorShown = null;
function showLoadError(message) {
    if (!message || loadErrorShown === message) return;
    loadErrorShown = message;
    const transcript = document.getElementById('transcript');
    if (!transcript) return;
    const banner = document.createElement('div');
    banner.className = 'msg assistant';
    banner.innerHTML =
        '<div class="avatar">⚠</div><div class="msg-content"><div class="bubble error"></div></div>';
    banner.querySelector('.bubble').innerText = `Model failed to load: ${message}`;
    transcript.appendChild(banner);
    transcript.scrollTop = transcript.scrollHeight;
}

// Renders a numeric metric, or an em dash when the engine reported nothing for it.
function fmt(value, digits, suffix) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return `${value.toFixed(digits)}${suffix}`;
}

// Telemetry Polling for Real-Time RAM and Engine Metrics
function fetchTelemetry() {
    fetch('/api/telemetry')
        .then(r => r.json())
        .then(data => {
            if (data.status === 'OK') {
                // Update GPU Name
                if (data.gpu_name) document.getElementById('hud-gpu').innerText = data.gpu_name;

                // Memory Telemetry.
                // Every field below uses ?? and renders EM DASH when the engine did not
                // report a value. `||` was previously used here, which treats 0 as falsy --
                // so a genuinely zero cache hit rate displayed as 78.4%, and zero VRAM as
                // 2850 MB. Absent data must look absent.
                const mem = data.memory || {};
                const resMB = mem.resident_weights_mb ?? null;
                const kvMB = mem.kv_cache_mb ?? null;
                const expMB = mem.expert_cache_mb ?? null;
                const totalModelMB = mem.total_model_ram_mb ?? null;
                const procSetMB = mem.process_working_set_mb ?? null;
                const sysAvailGB = mem.system_avail_ram_gb ?? null;
                const sysTotalGB = mem.system_total_ram_gb ?? null;
                const vramMB = mem.gpu_vram_used_mb ?? null;

                const totalModelGB = totalModelMB === null ? null : totalModelMB / 1024.0;
                document.getElementById('hud-ram-usage').innerText = fmt(totalModelGB, 2, ' GB');
                document.getElementById('hud-ram-sub').innerText =
                    sysTotalGB === null ? '/ — UMA' : `/ ${sysTotalGB.toFixed(1)} GB UMA`;
                document.getElementById('ram-total-badge').innerText = fmt(totalModelGB, 2, ' GB');

                const ramPct = (totalModelMB === null || !sysTotalGB)
                    ? 0 : Math.min(100, (totalModelMB / (sysTotalGB * 1024.0)) * 100);
                document.getElementById('hud-ram-fill').style.width = `${ramPct.toFixed(0)}%`;

                // Update Segmented RAM Bar
                const seg = (v) => (v === null || !totalModelMB) ? 0 : Math.min(100, (v / totalModelMB) * 100);
                document.getElementById('seg-resident').style.width = `${seg(resMB)}%`;
                document.getElementById('seg-kv').style.width = `${seg(kvMB)}%`;
                document.getElementById('seg-expert').style.width = `${seg(expMB)}%`;

                // Legend figures track the bar. These were hardcoded in the HTML.
                const gb = (v) => v === null ? '—' : `${(v / 1024.0).toFixed(2)} GB`;
                document.getElementById('legend-resident').innerText = gb(resMB);
                document.getElementById('legend-kv').innerText = gb(kvMB);
                document.getElementById('legend-expert').innerText = gb(expMB);

                // Metric Grid Labels
                document.getElementById('ram-val-resident').innerText = fmt(resMB, 0, ' MB');
                document.getElementById('ram-val-kv').innerText = fmt(kvMB, 0, ' MB');
                document.getElementById('ram-val-expert').innerText = fmt(expMB, 0, ' MB');
                document.getElementById('ram-val-process').innerText = fmt(procSetMB, 0, ' MB');
                document.getElementById('ram-val-sysavail').innerText = fmt(sysAvailGB, 2, ' GB');
                document.getElementById('ram-val-vram').innerText = fmt(vramMB, 0, ' MB');

                // Performance Metrics
                const perf = data.performance || {};
                document.getElementById('hud-toks').innerText = fmt(perf.decode_toks_sec ?? null, 1, ' t/s');
                document.getElementById('hud-io').innerText = fmt(perf.total_io_mbs ?? null, 1, ' MB/s');

                // Cache Metrics
                const cache = data.cache || {};
                document.getElementById('hud-cache').innerText = fmt(cache.hit_rate_pct ?? null, 1, '%');

                // Per-phase decode breakdown. These are cumulative totals over
                // tokens_measured, so they are shown as per-token averages -- the same shape
                // the CLI footer prints.
                const ms = (v) => (typeof v === 'number' ? `${v.toFixed(1)} ms` : '\u2014');
                const per = (v) => (typeof v === 'number' && perf.tokens_measured
                    ? `${(v / perf.tokens_measured).toFixed(1)} ms` : '\u2014');
                const setd = (id, text) => {
                    const el = document.getElementById(id);
                    if (el) el.innerText = text;
                };
                setd('diag-prefill', fmt(perf.prefill_toks_sec ?? null, 1, ' t/s'));
                setd('diag-expert-io', per(perf.expert_io_ms));
                setd('diag-gpu-wait', per(perf.gpu_wait_ms));
                setd('diag-lm-head', per(perf.lm_head_ms));
                setd('diag-cpu-other', per(perf.cpu_other_ms));
                setd('diag-gpu-waits', perf.gpu_waits != null ? String(perf.gpu_waits) : '\u2014');
                setd('diag-io-calls', perf.total_io_calls != null
                    ? perf.total_io_calls.toLocaleString() : '\u2014');
                setd('diag-tokens', perf.tokens_measured != null
                    ? String(perf.tokens_measured) : '\u2014');
                setd('diag-model-dir', data.model_dir || '\u2014');

                // server.cpp: "any value above 0 is worth acting on". Hidden at zero so it
                // reads as an alert rather than another metric.
                const fallbacks = mem.uma_upload_fallbacks ?? 0;
                const umaWarn = document.getElementById('diag-uma-warn');
                if (umaWarn) umaWarn.style.display = fallbacks > 0 ? '' : 'none';
                setd('diag-uma', String(fallbacks));

                // Active Experts Heatmap
                if (data.active_experts) {
                    updateHeatmap(data.active_experts);
                }

                // Status Badge. A held status (ERROR/STOPPED) survives the poll; the
                // model-status row underneath is always refreshed.
                if (!isGenerating) {
                    if (data.model_active) {
                        if (!statusHeld()) setStatus('READY', '#34d399', 0);
                        document.getElementById('meta-status').innerText = 'ACTIVE';
                        document.getElementById('meta-status').style.color = 'var(--accent-green)';
                    } else {
                        if (!statusHeld()) setStatus('NO MODEL', '#f87171', 0);
                        // Show *why* the model is unloaded rather than just that it is.
                        document.getElementById('meta-status').innerText =
                            data.load_error ? 'LOAD FAILED' : 'UNLOADED';
                        document.getElementById('meta-status').style.color = '#f87171';
                        document.getElementById('meta-status').title = data.load_error || '';
                        showLoadError(data.load_error);
                    }
                }
            }
        })
        .catch(err => {
            console.warn('Telemetry error:', err);
        });
}

// Model Repository Management
function refreshModelList() {
    return fetch('/api/models')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('model-select');
            const models = (data && data.models) || [];
            select.innerHTML = '';

            for (const m of models) {
                const opt = document.createElement('option');
                opt.value = m.path;
                opt.innerText = m.name + (m.is_active ? ' (Active)' : '');
                if (m.is_active) opt.selected = true;
                select.appendChild(opt);
            }

            // No invented fallback entry. There used to be a hardcoded
            // "gemma-4-26b-a4b.gturbo (Default)" option here, which looked like an installed
            // model and was guaranteed to fail on Load -- the one case where the user most
            // needs to be told the truth is when they have no bundle at all.
            if (models.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.innerText = 'No model bundles found';
                opt.disabled = true;
                opt.selected = true;
                select.appendChild(opt);
            }

            activeModel = models.find(m => m.is_active) || null;
            updateModelMeta();
            applySlotBounds();
            showNoModelHelp(models.length === 0);
        })
        .catch(err => console.warn('Model list fetch failed:', err));
}

// Surfaces the "you have no model" case in the panel itself rather than leaving a dropdown
// that looks populated. Phase 3 replaces the body of this card with the download flow.
function showNoModelHelp(show) {
    const card = document.getElementById('no-model-help');
    if (card) card.style.display = show ? '' : 'none';
}

function loadSelectedModel() {
    const modelPath = document.getElementById('model-select').value;
    setStatus('LOADING', '#60a5fa', 0);

    fetch('/api/load_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath })
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'SUCCESS') {
            document.getElementById('meta-id').innerText = modelPath.replace('.gturbo', '');
            refreshModelList();
            fetchTelemetry();
            // Re-read the resolved configuration. The reload now genuinely applies the
            // requested slots/context, and the engine may still have clamped them -- so show
            // what it settled on rather than what was asked for. This also clears the reload
            // banner, which used to stay up after the very reload that satisfied it.
            hydrateConfig();
        } else {
            reportError('Failed to load model: ' + (data.message || 'Unknown error'));
            fetchTelemetry();
        }
    })
    .catch(err => {
        reportError('Could not reach the engine while loading: ' + err);
        fetchTelemetry();
    });
}

function unloadModel() {
    fetch('/api/unload_model', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            fetchTelemetry();
            refreshModelList();
        })
        .catch(err => reportError('Unload failed: ' + err));
}

// Generation & Prompt Execution
function usePreset(promptText) {
    document.getElementById('prompt-input').value = promptText;
    sendPrompt();
}

const SYSTEM_PRESETS = {
    default: 'You are a helpful, knowledgeable assistant. Answer the question that was asked, directly and without preamble. Match length to the question: one sentence when that is enough, more when the topic needs it. If you do not know something or are uncertain, say so plainly instead of guessing.',
    code: 'You are an experienced software engineer. Give complete, runnable code: real imports, error paths handled, no placeholder comments standing in for logic. State the language version and any assumptions you made. Prefer the clear solution over the clever one. When reviewing or explaining existing code, describe what it actually does before suggesting changes, and flag bugs and edge cases you notice along the way.',
    explain: 'You explain technical subjects to a competent reader who is new to this particular topic. Lead with the core idea in plain language, then add the mechanism and the details that matter. Use concrete examples and numbers over analogies. Define a term the first time you use it. Say explicitly where your explanation simplifies something, and where the real behavior differs.',
    creative: 'You are a skilled creative writer. Write with concrete, specific detail and a distinct voice. Avoid cliche, filler adjectives, and tidy closing morals. Follow the requested form, length, and tone exactly. When a brief is vague, make a specific choice and commit to it rather than hedging across several options.',
    // Empty on purpose: sendPrompt() omits the system message entirely when the box is
    // blank, so this is the only setting that costs zero prefill tokens. With no prompt
    // cache, the system prompt is re-prefilled every turn, so that is worth ~10s of
    // time-to-first-token on every message.
    none: '',
};

function applySystemPreset() {
    const preset = document.getElementById('sys-preset-select').value;
    const input = document.getElementById('sys-prompt-input');
    // Compare against undefined, not truthiness: the 'none' preset is a legitimate empty
    // string, and `||` would silently substitute the default for it.
    const value = SYSTEM_PRESETS[preset];
    input.value = typeof value === 'string' ? value : SYSTEM_PRESETS.default;
    persistSystemPrompt();
    updateContextUsage();
}

// The system prompt is the one setting the server does not hold: it is read out of the DOM
// per message and was reset to the default persona on every page reload, silently discarding
// whatever the user had written. localStorage is per-viewer and the right home for it --
// unlike the sampling knobs, which are engine-wide and belong on the server.
const SYS_PROMPT_KEY = 'turbo.systemPrompt';
const SYS_PRESET_KEY = 'turbo.systemPreset';

function persistSystemPrompt() {
    try {
        const input = document.getElementById('sys-prompt-input');
        const select = document.getElementById('sys-preset-select');
        if (input) localStorage.setItem(SYS_PROMPT_KEY, input.value);
        if (select) localStorage.setItem(SYS_PRESET_KEY, select.value);
    } catch (e) {
        // Private windows and blocked site data throw on access, not just on read.
        console.warn('Could not persist the system prompt:', e);
    }
}

function restoreSystemPrompt() {
    try {
        const saved = localStorage.getItem(SYS_PROMPT_KEY);
        const preset = localStorage.getItem(SYS_PRESET_KEY);
        const input = document.getElementById('sys-prompt-input');
        const select = document.getElementById('sys-preset-select');
        // An empty saved prompt is meaningful (the 'none' persona), so test for null
        // rather than truthiness -- otherwise choosing 'none' would come back as the
        // default persona on the next load.
        if (input) input.value = (saved !== null) ? saved : SYSTEM_PRESETS.default;
        if (select && preset !== null) select.value = preset;
    } catch (e) {
        console.warn('Could not restore the system prompt:', e);
    }
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendPrompt();
    }
}

function sendPrompt() {
    // The generating check must come FIRST. Sending clears the textarea, so during a
    // generation the box is empty -- an empty-input guard above this returned early and made
    // the stop button (the same element, relabelled) impossible to click.
    if (isGenerating) {
        stopGeneration();
        return;
    }

    const input = document.getElementById('prompt-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    appendUserMessage(prompt);
    input.value = '';
    isGenerating = true;
    document.getElementById('btn-send').innerText = '■';
    document.getElementById('btn-send').title = 'Stop Generation';
    setStatus('GENERATING', '#3b82f6', 0);

    currentBubble = createAssistantMessage();
    conversation.push({ role: 'user', content: prompt });
    updateContextUsage();

    const systemPrompt = document.getElementById('sys-prompt-input').value.trim();
    // Send the whole conversation, not just the latest turn. Only the current textarea used
    // to go out, so every message started a fresh conversation with no memory of the last.
    const messages = [];
    if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
    for (const m of conversation) messages.push(m);

    // Built from the same reader the config form uses. Reading the number boxes with a raw
    // parseInt here produced NaN for an empty box, which JSON.stringify emits as null, and
    // the engine then silently substituted its own default.
    const payload = Object.assign(readGenerationParams(), {
        messages: messages,
        model: serverModelId,
        stream: true,
        // Without this the server sends no usage chunk, so token counts and TTFT -- which
        // the engine measures on every request -- never reached the client at all.
        stream_options: { include_usage: true }
    });

    streamCompletion(payload);
}

// Consumes the SSE stream from /v1/chat/completions, appending each delta as it arrives.
// The whole completion used to land in one call to appendToken, so the UI sat blank for the
// entire generation and then filled instantly.
function streamCompletion(payload) {
    let assistantText = '';
    const stats = { usage: null, turbo: null, startedAt: Date.now() };
    const bubbleForStats = currentBubble;
    currentAbort = new AbortController();

    return (async () => {
        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: currentAbort.signal
            });

            if (!response.ok) {
                const body = await response.json().catch(() => null);
                renderError((body && body.error && body.error.message) ||
                            `Engine returned HTTP ${response.status}.`);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                // SSE events are separated by a blank line; a partial tail stays buffered.
                let split;
                while ((split = buffer.indexOf('\n\n')) !== -1) {
                    const event = buffer.slice(0, split);
                    buffer = buffer.slice(split + 2);

                    for (const line of event.split('\n')) {
                        // ": ping" keepalives are comments and carry no data.
                        if (!line.startsWith('data: ')) continue;
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;

                        let obj;
                        try { obj = JSON.parse(data); } catch (e) { continue; }
                        if (obj.error) { renderError(obj.error.message || 'Engine error.'); return; }

                        // The usage-only chunk carries no choices.
                        if (obj.usage) {
                            stats.usage = obj.usage;
                            if (typeof obj.usage.prompt_tokens === 'number') {
                                lastPromptTokens = obj.usage.prompt_tokens;
                            }
                        }
                        // Vendor extension on the final chunk: the true stop reason, which
                        // finish_reason deliberately collapses for OpenAI compatibility.
                        if (obj.x_turbo) stats.turbo = obj.x_turbo;

                        const choice = obj.choices && obj.choices[0];
                        if (choice && choice.delta && choice.delta.content) {
                            assistantText += choice.delta.content;
                            appendToken(choice.delta.content);
                        }
                    }
                }
            }

            finishGeneration();
        } catch (err) {
            // An abort is a user action, not a failure: stopGeneration() has already set the
            // badge, and the partial reply already on screen stands.
            if (err && err.name === 'AbortError') return;
            renderError(`Could not reach the engine: ${err}`);
        } finally {
            currentAbort = null;
            renderStats(bubbleForStats, stats);
            settleConversation(assistantText);
        }
    })();
}

// A per-response footer under the assistant bubble. Everything here is measured by the
// engine and was already crossing the wire or available for the asking; none of it had
// anywhere to go in the UI.
function renderStats(bubble, stats) {
    if (!bubble || (!stats.usage && !stats.turbo)) return;
    const parts = [];

    if (stats.usage) {
        parts.push(`${stats.usage.prompt_tokens} prompt`);
        parts.push(`${stats.usage.completion_tokens} completion`);
        if (stats.turbo && typeof stats.turbo.ttft_ms === 'number') {
            const ttft = stats.turbo.ttft_ms;
            parts.push(`TTFT ${ttft >= 1000 ? (ttft / 1000).toFixed(2) + ' s'
                                            : ttft.toFixed(0) + ' ms'}`);
            // Decode rate over the post-prefill window, which is the number the performance
            // notes quote. Deriving it from total elapsed would blend in the prefill.
            const decodeMs = (Date.now() - stats.startedAt) - ttft;
            if (decodeMs > 0 && stats.usage.completion_tokens > 1) {
                parts.push(`${(stats.usage.completion_tokens / (decodeMs / 1000)).toFixed(1)} tok/s`);
            }
        }
    }

    if (stats.turbo && stats.turbo.stop_reason) {
        parts.push(stats.turbo.matched_stop
            ? `stopped at ${JSON.stringify(stats.turbo.matched_stop)}`
            : stats.turbo.stop_reason.replace(/_/g, ' '));
    }

    if (!parts.length) return;
    const line = document.createElement('div');
    line.className = 'msg-stats';
    line.innerText = parts.join('  \u00b7  ');
    if (bubble.parentElement) bubble.parentElement.appendChild(line);
}

// Keeps `conversation` a well-formed alternating transcript. A stopped or failed turn used
// to leave its user message with no reply, so the next request sent two consecutive user
// turns -- accepted by the server validator but a malformed chat template for the model.
function settleConversation(assistantText) {
    if (assistantText) {
        conversation.push({ role: 'assistant', content: assistantText });
    } else if (conversation.length &&
               conversation[conversation.length - 1].role === 'user') {
        conversation.pop();
    }
    updateContextUsage();
}

function finishGeneration() {
    resetSendButton();
    setStatus('READY', '#34d399', 0);
    fetchTelemetry();
}

function resetSendButton() {
    isGenerating = false;
    const btn = document.getElementById('btn-send');
    btn.innerText = '➔';
    btn.title = 'Send Prompt';
}

// Replaces the pending assistant bubble with a visibly-styled error.
function renderError(message) {
    if (currentBubble) {
        currentBubble.classList.add('error');
        currentBubble.innerText = `⚠ ${message}`;
    }
    resetSendButton();
    setStatus('ERROR', '#f87171', 6000);
}

function stopGeneration() {
    // Reset the UI first, then abort the reader, then tell the engine. Without the abort the
    // stream kept arriving and the normal completion path overwrote the STOPPED badge.
    resetSendButton();
    setStatus('STOPPED', '#f59e0b', 6000);
    if (currentAbort) currentAbort.abort();
    fetch('/api/stop', { method: 'POST' })
        .then(() => fetchTelemetry())
        .catch(err => console.warn('Stop request failed:', err));
}

function clearTranscript() {
    // Clear the real conversation too. This used to only rewrite the DOM, so "clear" wiped
    // the screen while the model kept the full history.
    conversation = [];
    lastPromptTokens = null;
    updateContextUsage();
    // Let a load-error banner reappear after a clear; showLoadError() dedupes on this.
    loadErrorShown = null;
    const transcript = document.getElementById('transcript');
    transcript.innerHTML = `
        <div class="msg assistant">
            <div class="avatar">⚡</div>
            <div class="msg-content">
                <div class="bubble">Transcript cleared. Engine ready for new instructions!</div>
            </div>
        </div>
    `;
}

// Autoscroll only while the reader is already at the bottom. Forcing scrollTop on every
// token makes the transcript impossible to scroll back through during generation.
function isPinnedToBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

function appendUserMessage(text) {
    const transcript = document.getElementById('transcript');
    const msg = document.createElement('div');
    msg.className = 'msg user';
    msg.innerHTML = `
        <div class="avatar">👤</div>
        <div class="msg-content">
            <div class="bubble">${escapeHtml(text)}</div>
        </div>
    `;
    transcript.appendChild(msg);
    scrollToBottom(transcript);
}

function createAssistantMessage() {
    const transcript = document.getElementById('transcript');
    const msg = document.createElement('div');
    msg.className = 'msg assistant';
    msg.innerHTML = `
        <div class="avatar">⚡</div>
        <div class="msg-content">
            <div class="bubble"></div>
        </div>
    `;
    transcript.appendChild(msg);
    scrollToBottom(transcript);
    return msg.querySelector('.bubble');
}

function appendToken(text) {
    if (currentBubble) {
        const transcript = document.getElementById('transcript');
        // Sample before mutating: appending grows scrollHeight, which would make an
        // already-scrolled-up reader look "at the bottom" on the very next token.
        const stick = isPinnedToBottom(transcript);
        currentBubble.innerText += text;
        if (stick) scrollToBottom(transcript);
    }
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Fallback geometry, used only before /api/models has answered or when no model is loaded.
// These were the sole source of the slot-pool figure, so the label described the default
// bundle's geometry regardless of which bundle was open. The live values now come from
// activeModel.expert_stride / activeModel.layers.
const DEFAULT_EXPERT_STRIDE_BYTES = 3358720;
const DEFAULT_NUM_LAYERS = 30;

// Only the two <input type="number"> boxes can be left empty; a range or select always holds
// a valid value. An empty box parses as NaN, which JSON.stringify emits as null -- so fall
// back to the element's own default rather than posting a null for the engine to interpret.
function numberBox(id) {
    const el = document.getElementById(id);
    const v = parseInt(el.value, 10);
    return Number.isNaN(v) ? parseInt(el.defaultValue, 10) : v;
}

// The per-request sampling knobs, and the only place they are read. sendPrompt() used to
// parse the number boxes itself and skipped the numberBox() guard below.
function readGenerationParams() {
    const params = {
        temperature: parseFloat(document.getElementById('cfg-temp').value),
        top_p: parseFloat(document.getElementById('cfg-topp').value),
        top_k: numberBox('cfg-topk'),
        max_tokens: numberBox('cfg-maxtoks'),
        repetition_penalty: parseFloat(document.getElementById('cfg-reppen').value)
    };

    // An empty box means "no seed": send an explicit null so the server clears any stored
    // one, rather than omitting the key and leaving the previous seed in force.
    const raw = document.getElementById('cfg-seed').value.trim();
    params.seed = raw === '' ? null : parseInt(raw, 10);
    if (Number.isNaN(params.seed)) params.seed = null;

    params.stop = readStopSequences();
    return params;
}

// One stop sequence per line. Blank lines are dropped -- an empty stop matches immediately
// and would halt every generation at token zero -- but interior and trailing spaces are
// preserved, because whitespace is a legitimate part of a stop string.
function readStopSequences() {
    const box = document.getElementById('cfg-stop');
    if (!box) return [];
    return box.value.split('\n')
        .filter(line => line.length > 0)
        .slice(0, maxStopSequences);
}

function readConfigForm() {
    return Object.assign(readGenerationParams(), {
        context_len: parseInt(document.getElementById('cfg-context').value, 10),
        slots: parseInt(document.getElementById('cfg-slots').value, 10),
        eviction_policy: document.getElementById('cfg-eviction').value || 'LFU'
    });
}

function renderConfigLabels() {
    const c = readConfigForm();
    document.getElementById('val-temp').innerText = c.temperature.toFixed(2);
    document.getElementById('val-topp').innerText = c.top_p.toFixed(2);
    const stride = (activeModel && activeModel.expert_stride) || DEFAULT_EXPERT_STRIDE_BYTES;
    const layers = (activeModel && activeModel.layers) || DEFAULT_NUM_LAYERS;
    const slotsMB = ((c.slots * layers * stride) / (1024 * 1024)).toFixed(0);
    // Labelled an estimate: it is the pool allocation, not the process footprint the RAM
    // panel reports, and the two are not the same number.
    document.getElementById('val-slots').innerText = `${c.slots} Slots (~${slotsMB} MB est.)`;
    document.getElementById('val-topk').innerText = c.top_k;
    document.getElementById('val-context').innerText = c.context_len;
    document.getElementById('val-maxtoks').innerText = c.max_tokens;
    document.getElementById('val-reppen').innerText =
        c.repetition_penalty.toFixed(2) + (c.repetition_penalty === 1 ? ' (off)' : '');
    document.getElementById('val-seed').innerText = c.seed === null ? 'random' : String(c.seed);
    const stops = c.stop || [];
    document.getElementById('val-stop').innerText =
        stops.length === 0 ? 'none' : `${stops.length} / ${maxStopSequences}`;
    document.getElementById('val-eviction').innerText = c.eviction_policy;
}

// The label tracks the drag immediately; the POST is debounced. Dragging a range control
// with oninput previously fired one request per pixel of travel.
function updateConfig() {
    renderConfigLabels();
    if (configDebounce) clearTimeout(configDebounce);
    configDebounce = setTimeout(pushConfig, 250);
}

function pushConfig() {
    configDebounce = null;
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(readConfigForm())
    })
    .then(async r => {
        const data = await r.json().catch(() => null);
        // A rejected change is not a successful one. The server answers 400 with a message
        // for an out-of-range context or slot count; reading only requires_reload turned
        // that into `undefined` and *hid* the banner, so the rejection was invisible and the
        // control kept showing a value the engine had refused.
        if (!r.ok || !data || data.status !== 'SUCCESS') {
            reportError('Configuration rejected: ' +
                        ((data && data.message) || ('HTTP ' + r.status)));
            hydrateConfig();
            return;
        }
        // Slot count and context size are fixed at initialize(); the server says so via
        // requires_reload.
        showReloadNotice(!!data.requires_reload);
        fetchTelemetry();
    })
    .catch(err => reportError('Configuration sync failed: ' + err));
}

function showReloadNotice(needed) {
    const el = document.getElementById('reload-notice');
    if (el) el.style.display = needed ? 'block' : 'none';
}

function updateEvictionPolicy() {
    updateConfig();
}

function clearSeed() {
    const box = document.getElementById('cfg-seed');
    if (!box) return;
    box.value = '';
    updateConfig();
}

function flushExpertCache() {
    fetch('/api/clear_cache', { method: 'POST' })
        .then(async r => {
            const data = await r.json().catch(() => null);
            if (!r.ok || !data || data.status !== 'SUCCESS') {
                reportError('Could not flush the expert cache: ' +
                            ((data && data.message) || ('HTTP ' + r.status)));
                return;
            }
            notify(data.message || 'Expert cache flushed.');
            fetchTelemetry();
        })
        .catch(err => reportError('Could not reach the engine: ' + err));
}

/* Modals & Active Expert Heatmap */
// One cell per expert in ONE layer, because that is all the engine reports.
//
// This was a 30x32 = 960 grid indexed `idx % 960`, as if it showed every layer. The model
// has 128 experts per layer and last_active_experts() returns only layer 0's top-8, so the
// old grid was decorative: expert 100 lit a cell in the wrong row and 872 of 960 cells could
// never light at all. Showing layer 0 honestly beats showing all 30 layers wrongly.
//
// The count is the loaded model's expert count, not a constant -- a bundle with a different
// expert count would otherwise silently drop every index past 127.
const DEFAULT_HEATMAP_EXPERTS = 128;
let heatmapExperts = DEFAULT_HEATMAP_EXPERTS;

function initHeatmapGrid() {
    const grid = document.getElementById('heatmap-grid');
    grid.innerHTML = '';
    for (let i = 0; i < heatmapExperts; i++) {
        const cell = document.createElement('div');
        cell.className = 'expert-cell';
        cell.id = `expert-cell-${i}`;
        cell.title = `Layer 0, expert ${i}`;
        grid.appendChild(cell);
    }
    const title = document.getElementById('heatmap-title');
    if (title) {
        title.innerText = `Active Expert Routing \u2014 Layer 0 (${heatmapExperts} Experts)`;
    }
}

function resizeHeatmap(experts) {
    const next = experts || DEFAULT_HEATMAP_EXPERTS;
    if (next === heatmapExperts) return;
    heatmapExperts = next;
    initHeatmapGrid();
}

function updateHeatmap(activeIndices) {
    for (let i = 0; i < heatmapExperts; i++) {
        const cell = document.getElementById(`expert-cell-${i}`);
        if (cell) cell.classList.remove('active');
    }
    if (Array.isArray(activeIndices)) {
        activeIndices.forEach(idx => {
            // Out-of-range indices are dropped rather than wrapped into a wrong cell.
            if (idx < 0 || idx >= heatmapExperts) return;
            const cell = document.getElementById(`expert-cell-${idx}`);
            if (cell) cell.classList.add('active');
        });
    }
}

function openHeatmapModal() { document.getElementById('heatmap-modal').classList.add('active'); }
function closeHeatmapModal() { document.getElementById('heatmap-modal').classList.remove('active'); }
// The repacker UI is gone along with /api/repack -- see the note in index.html. Building a
// bundle is `python tools/convert_hf_to_gturbo.py`.
