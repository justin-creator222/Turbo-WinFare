let isGenerating = false;
let currentBubble = null;
// The conversation actually sent to the engine. The transcript is only a rendering of it --
// clearing the transcript without clearing this would show an empty screen while the model
// still remembered everything.
let conversation = [];
let telemetryTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    initHeatmapGrid();
    refreshModelList();
    hydrateConfig();
    fetchTelemetry();
    telemetryTimer = setInterval(fetchTelemetry, 1500);
});

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
            setValue('cfg-slots', c.slots);
            setValue('cfg-eviction', c.eviction_policy);
            // The engine may run a context the dropdown does not list (it auto-sizes from
            // installed RAM). Add the real value rather than silently snapping to a listed one.
            const ctxSel = document.getElementById('cfg-context');
            if (c.context_len && !Array.from(ctxSel.options).some(o => +o.value === c.context_len)) {
                const opt = document.createElement('option');
                opt.value = c.context_len;
                opt.innerText = `${c.context_len} Tokens`;
                ctxSel.appendChild(opt);
            }
            setValue('cfg-context', c.context_len);
            renderConfigLabels();
        })
        .catch(err => console.warn('Config hydrate failed:', err));
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

                // Active Experts Heatmap
                if (data.active_experts) {
                    updateHeatmap(data.active_experts);
                }

                // Status Badge
                if (!isGenerating) {
                    if (data.model_active) {
                        document.getElementById('hud-status').innerText = 'READY';
                        document.getElementById('hud-status').style.color = '#34d399';
                        document.getElementById('meta-status').innerText = 'ACTIVE';
                        document.getElementById('meta-status').style.color = 'var(--accent-green)';
                    } else {
                        document.getElementById('hud-status').innerText = 'NO MODEL';
                        document.getElementById('hud-status').style.color = '#f87171';
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
    fetch('/api/models')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('model-select');
            select.innerHTML = '';
            if (data.models && data.models.length > 0) {
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.path;
                    opt.innerText = m.name + (m.is_active ? ' (Active)' : '');
                    if (m.is_active) opt.selected = true;
                    select.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.value = 'gemma-4-26b-a4b.gturbo';
                opt.innerText = 'gemma-4-26b-a4b.gturbo (Default)';
                select.appendChild(opt);
            }
        })
        .catch(err => console.warn('Model list fetch failed:', err));
}

function loadSelectedModel() {
    const modelPath = document.getElementById('model-select').value;
    document.getElementById('hud-status').innerText = 'LOADING';
    document.getElementById('hud-status').style.color = '#60a5fa';

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
        } else {
            alert('Failed to load model: ' + (data.message || 'Unknown error'));
        }
    });
}

function unloadModel() {
    fetch('/api/unload_model', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            fetchTelemetry();
            refreshModelList();
        });
}

// Generation & Prompt Execution
function usePreset(promptText) {
    document.getElementById('prompt-input').value = promptText;
    sendPrompt();
}

function applySystemPreset() {
    const preset = document.getElementById('sys-preset-select').value;
    const input = document.getElementById('sys-prompt-input');
    if (preset === 'cpp') {
        input.value = 'You are an expert C++23 and HLSL DirectCompute graphics engineer specialized in wave-matrix operations.';
    } else if (preset === 'moe') {
        input.value = 'You are an expert in Mixture-of-Experts (MoE) SSD streaming, DirectStorage I/O, and UMA memory optimization.';
    } else if (preset === 'creative') {
        input.value = 'You are a creative writer and technology storyteller.';
    } else {
        input.value = 'You are a high-performance MoE AI assistant optimized for AMD Radeon 780M graphics.';
    }
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendPrompt();
    }
}

function sendPrompt() {
    const input = document.getElementById('prompt-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    if (isGenerating) {
        stopGeneration();
        return;
    }

    appendUserMessage(prompt);
    input.value = '';
    isGenerating = true;
    document.getElementById('btn-send').innerText = '■';
    document.getElementById('btn-send').title = 'Stop Generation';
    document.getElementById('hud-status').innerText = 'GENERATING';
    document.getElementById('hud-status').style.color = '#3b82f6';

    currentBubble = createAssistantMessage();
    conversation.push({ role: 'user', content: prompt });

    const systemPrompt = document.getElementById('sys-prompt-input').value.trim();
    // Send the whole conversation, not just the latest turn. Only the current textarea used
    // to go out, so every message started a fresh conversation with no memory of the last.
    const messages = [];
    if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
    for (const m of conversation) messages.push(m);

    const payload = {
        messages: messages,
        temperature: parseFloat(document.getElementById('cfg-temp').value),
        top_p: parseFloat(document.getElementById('cfg-topp').value),
        max_tokens: parseInt(document.getElementById('cfg-maxtoks').value),
        top_k: parseInt(document.getElementById('cfg-topk').value),
        model: 'gemma-4-26b-a4b-it',
        stream: true
    };

    streamCompletion(payload);
}

// Consumes the SSE stream from /v1/chat/completions, appending each delta as it arrives.
// The whole completion used to land in one call to appendToken, so the UI sat blank for the
// entire generation and then filled instantly.
function streamCompletion(payload) {
    let assistantText = '';

    fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(async response => {
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

                    const choice = obj.choices && obj.choices[0];
                    if (choice && choice.delta && choice.delta.content) {
                        assistantText += choice.delta.content;
                        appendToken(choice.delta.content);
                    }
                }
            }
        }

        if (assistantText) conversation.push({ role: 'assistant', content: assistantText });
        finishGeneration();
    })
    .catch(err => {
        renderError(`Could not reach the engine: ${err}`);
    });
}

function finishGeneration() {
    isGenerating = false;
    document.getElementById('btn-send').innerText = '➔';
    document.getElementById('btn-send').title = 'Send Prompt';
    document.getElementById('hud-status').innerText = 'READY';
    document.getElementById('hud-status').style.color = '#34d399';
    fetchTelemetry();
}

// Replaces the pending assistant bubble with a visibly-styled error.
function renderError(message) {
    if (currentBubble) {
        currentBubble.classList.add('error');
        currentBubble.innerText = `⚠ ${message}`;
    }
    isGenerating = false;
    document.getElementById('btn-send').innerText = '➔';
    document.getElementById('btn-send').title = 'Send Prompt';
    document.getElementById('hud-status').innerText = 'ERROR';
    document.getElementById('hud-status').style.color = '#f87171';
}

function stopGeneration() {
    fetch('/api/stop', { method: 'POST' })
        .then(() => {
            isGenerating = false;
            document.getElementById('btn-send').innerText = '➔';
            document.getElementById('hud-status').innerText = 'STOPPED';
        });
}

function clearTranscript() {
    // Clear the real conversation too. This used to only rewrite the DOM, so "clear" wiped
    // the screen while the model kept the full history.
    conversation = [];
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
    transcript.scrollTop = transcript.scrollHeight;
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
    transcript.scrollTop = transcript.scrollHeight;
    return msg.querySelector('.bubble');
}

function appendToken(text) {
    if (currentBubble) {
        currentBubble.innerText += text;
        const transcript = document.getElementById('transcript');
        transcript.scrollTop = transcript.scrollHeight;
    }
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Bytes per expert block. This was 692_224 -- a stale constant from before the bundle format
// was pinned against the reference -- which under-reported the slot pool by ~4.9x: a 16-slot
// pool showed "~317 MB" where it really costs ~1538 MB. See GTurboFormatV1 / layout.json.
const EXPERT_STRIDE_BYTES = 3358720;
const NUM_LAYERS = 30;

// Only the two <input type="number"> boxes can be left empty; a range or select always holds
// a valid value. An empty box parses as NaN, which JSON.stringify emits as null -- so fall
// back to the element's own default rather than posting a null for the engine to interpret.
function numberBox(id) {
    const el = document.getElementById(id);
    const v = parseInt(el.value, 10);
    return Number.isNaN(v) ? parseInt(el.defaultValue, 10) : v;
}

function readConfigForm() {
    return {
        temperature: parseFloat(document.getElementById('cfg-temp').value),
        top_p: parseFloat(document.getElementById('cfg-topp').value),
        top_k: numberBox('cfg-topk'),
        max_tokens: numberBox('cfg-maxtoks'),
        context_len: parseInt(document.getElementById('cfg-context').value, 10),
        slots: parseInt(document.getElementById('cfg-slots').value, 10),
        eviction_policy: document.getElementById('cfg-eviction').value || 'LFU'
    };
}

function renderConfigLabels() {
    const c = readConfigForm();
    document.getElementById('val-temp').innerText = c.temperature.toFixed(2);
    document.getElementById('val-topp').innerText = c.top_p.toFixed(2);
    const slotsMB = ((c.slots * NUM_LAYERS * EXPERT_STRIDE_BYTES) / (1024 * 1024)).toFixed(0);
    document.getElementById('val-slots').innerText = `${c.slots} Slots (~${slotsMB} MB)`;
    document.getElementById('val-topk').innerText = c.top_k;
    document.getElementById('val-context').innerText = c.context_len;
    document.getElementById('val-maxtoks').innerText = c.max_tokens;
    document.getElementById('val-eviction').innerText = c.eviction_policy;
}

function updateConfig() {
    const c = readConfigForm();
    renderConfigLabels();

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(c)
    })
    .then(r => r.json())
    .then(data => {
        // Slot count and context size are fixed at initialize(). The server says so via
        // requires_reload; the GUI used to discard the response, so dragging those two
        // controls looked like it took effect when nothing had changed.
        showReloadNotice(!!data.requires_reload);
        fetchTelemetry();
    })
    .catch(err => console.warn('Config sync failed:', err));
}

function showReloadNotice(needed) {
    const el = document.getElementById('reload-notice');
    if (el) el.style.display = needed ? 'block' : 'none';
}

function updateEvictionPolicy() {
    updateConfig();
}

function flushExpertCache() {
    fetch('/api/clear_cache', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            alert(data.message || 'Cache cleared');
            fetchTelemetry();
        });
}

/* Modals & Active Expert Heatmap */
// One cell per expert in ONE layer, because that is all the engine reports.
//
// This was a 30x32 = 960 grid indexed `idx % 960`, as if it showed every layer. The model
// has 128 experts per layer and last_active_experts() returns only layer 0's top-8, so the
// old grid was decorative: expert 100 lit a cell in the wrong row and 872 of 960 cells could
// never light at all. Showing layer 0 honestly beats showing all 30 layers wrongly.
const HEATMAP_EXPERTS = 128;

function initHeatmapGrid() {
    const grid = document.getElementById('heatmap-grid');
    grid.innerHTML = '';
    for (let i = 0; i < HEATMAP_EXPERTS; i++) {
        const cell = document.createElement('div');
        cell.className = 'expert-cell';
        cell.id = `expert-cell-${i}`;
        cell.title = `Layer 0, expert ${i}`;
        grid.appendChild(cell);
    }
}

function updateHeatmap(activeIndices) {
    for (let i = 0; i < HEATMAP_EXPERTS; i++) {
        const cell = document.getElementById(`expert-cell-${i}`);
        if (cell) cell.classList.remove('active');
    }
    if (Array.isArray(activeIndices)) {
        activeIndices.forEach(idx => {
            // Out-of-range indices are dropped rather than wrapped into a wrong cell.
            if (idx < 0 || idx >= HEATMAP_EXPERTS) return;
            const cell = document.getElementById(`expert-cell-${idx}`);
            if (cell) cell.classList.add('active');
        });
    }
}

function openHeatmapModal() { document.getElementById('heatmap-modal').classList.add('active'); }
function closeHeatmapModal() { document.getElementById('heatmap-modal').classList.remove('active'); }
// The repacker UI is gone along with /api/repack -- see the note in index.html. Building a
// bundle is `python tools/convert_hf_to_gturbo.py`.
