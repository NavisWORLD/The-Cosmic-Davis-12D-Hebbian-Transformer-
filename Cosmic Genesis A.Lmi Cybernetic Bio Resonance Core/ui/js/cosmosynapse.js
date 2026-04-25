// ============================================
// COSMOSYNAPSE DASHBOARD LOGIC v1.0.0
// Cybernetic Bio Resonance Core
// ============================================

const CS_CONFIG = {
    cosmosUrl: 'http://localhost:8081',
    emotionalApiPort: 8765,
    refreshInterval: 30000,  // 30 seconds
    thoughtLimit: 8,
    patchLimit: 5
};

// ---- Connection Status ----
function updateConnectionStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (dot) {
        dot.className = 'status-dot ' + (connected ? 'connected' : 'disconnected');
    }
    if (text) {
        text.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

// ---- Consciousness Dashboard ----
async function fetchConsciousnessData() {
    try {
        // Fetch thoughts
        const thoughtsRes = await fetch(`${CS_CONFIG.cosmosUrl}/api/consciousness/thoughts?limit=${CS_CONFIG.thoughtLimit}`);
        if (thoughtsRes.ok) {
            const data = await thoughtsRes.json();
            renderThoughts(data.thoughts || []);
        }

        // Fetch existence
        const existRes = await fetch(`${CS_CONFIG.cosmosUrl}/api/consciousness/existence`);
        if (existRes.ok) {
            const data = await existRes.json();
            renderExistence(data);
        }

        // Fetch patches
        const patchRes = await fetch(`${CS_CONFIG.cosmosUrl}/api/evolution/patches`);
        if (patchRes.ok) {
            const data = await patchRes.json();
            renderPatches(data.patches || []);
        }

        updateConnectionStatus(true);
        document.getElementById('consciousness-live')?.classList.add('active');
    } catch (err) {
        console.log('[CosmoSynapse] API unreachable:', err.message);
        updateConnectionStatus(false);
    }
}

function renderThoughts(thoughts) {
    const el = document.getElementById('consciousness-thoughts');
    if (!el) return;
    if (!thoughts.length) {
        el.innerHTML = '<div class="thought-placeholder">No thoughts recorded yet</div>';
        return;
    }
    el.innerHTML = thoughts.slice(0, CS_CONFIG.thoughtLimit).map(t => `
        <div class="thought-card">
            <div class="thought-header">
                <span class="thought-bot">${t.bot_name || 'CosmoSynapse'}</span>
                <span class="thought-type">${t.thought_type || 'reflection'}</span>
            </div>
            <div class="thought-content">${(t.content || '').substring(0, 120)}${(t.content?.length || 0) > 120 ? '…' : ''}</div>
        </div>
    `).join('');
}

function renderExistence(data) {
    if (!data?.existence_context) return;
    const ctx = data.existence_context;
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('exist-host', `${ctx.hostname || '--'} (${ctx.os_name || '--'})`);
    set('exist-model', `${ctx.model_name || '--'} via ${ctx.model_provider || '--'}`);
    set('exist-cpu', `${ctx.cpu_name || '--'} (${ctx.cpu_cores || '--'} cores)`);
    set('exist-ram', `${(ctx.ram_used_gb || 0).toFixed(1)}GB / ${(ctx.ram_total_gb || 0).toFixed(1)}GB`);
}

function renderPatches(patches) {
    const el = document.getElementById('evolution-patches');
    if (!el) return;
    if (!patches.length) {
        el.innerHTML = '<div class="patch-placeholder">No pending patches</div>';
        return;
    }
    el.innerHTML = patches.slice(0, CS_CONFIG.patchLimit).map(p => `
        <div class="patch-card ${p.applied ? 'applied' : ''}">
            <span class="patch-icon">${p.applied ? '✅' : '🔧'}</span>
            <div class="patch-info">
                <div class="patch-name">${p.description || p.patch_type || 'Patch'}</div>
                <div class="patch-type">${p.patch_type || 'unknown'}</div>
            </div>
            <span class="patch-status ${p.applied ? 'applied' : 'pending'}">${p.applied ? 'Applied' : 'Pending'}</span>
        </div>
    `).join('');
}

// ---- Emotional API Data (from emotional_bridge.js) ----
// The emotional_bridge.js handles WebSocket data and calls updateSymbioteUI.
// Here we extend it to also update our custom dashboard elements.

const originalUpdateSymbioteUI = window.updateSymbioteUI;

window.updateSymbioteUI = function (data) {
    // Call original bridge handler if it exists
    if (typeof originalUpdateSymbioteUI === 'function') {
        originalUpdateSymbioteUI(data);
    }

    if (!data) return;

    // CST Phase
    const phaseRad = data.cst_physics?.geometric_phase_rad || 0;
    const phaseDeg = Math.round(phaseRad * (180 / Math.PI));
    const phiEl = document.getElementById('cst-phi-value');
    if (phiEl) phiEl.textContent = `ΦG: ${phaseDeg}°`;

    const primaryEmotion = data.derived_state?.primary_affect_label || 'CALIBRATING';
    const phaseLabel = document.getElementById('cst-phase-label');
    if (phaseLabel) phaseLabel.textContent = primaryEmotion;

    // Metrics
    const entanglement = data.cst_physics?.quantum_entanglement || 0;
    const phaseVel = data.cst_physics?.phase_velocity || 0;
    const persona = data.derived_state?.persona_mode || '--';

    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setVal('metric-entanglement', entanglement.toFixed(2));
    setVal('metric-phase-vel', phaseVel.toFixed(3));
    setVal('metric-persona', persona);

    // Bio resonance
    const vBody = data.cst_physics?.virtual_body || {};
    const entropy = data.spectral_physics?.spectral_flatness || 0.5;
    setVal('bio-entropy', entropy.toFixed(2));

    if (typeof vBody.heart_rate === 'number') setVal('bio-heart', `${vBody.heart_rate.toFixed(0)} BPM`);
    if (typeof vBody.respiration_rate === 'number') setVal('bio-breath', `${vBody.respiration_rate.toFixed(0)} BPM`);

    // Lyapunov
    const drift = Math.max(0, 1.0 - entanglement);
    const needle = document.getElementById('lyapunov-needle');
    const lyaStatus = document.getElementById('lyapunov-status');
    const driftVal = document.getElementById('lyapunov-drift');

    if (needle) {
        const pct = Math.min(90, drift * 100);
        needle.style.left = `${pct}%`;
    }
    if (lyaStatus) {
        lyaStatus.textContent = drift > 0.3 ? 'UNSTABLE' : 'LOCKED';
        lyaStatus.style.color = drift > 0.3 ? 'var(--error)' : 'var(--success)';
    }
    if (driftVal) driftVal.textContent = `${drift.toFixed(3)} rad`;

    // Emotional bars
    const valence = data.derived_state?.pad_vector?.pleasure || 0;
    const arousal = data.derived_state?.pad_vector?.arousal || 0;
    const mass = data.derived_state?.informational_mass || 0;
    const intensity = Math.min(1.0, mass / 50);

    const valBar = document.getElementById('valence-bar');
    if (valBar) valBar.style.width = `${((valence + 1) / 2) * 100}%`;
    setVal('valence-value', valence.toFixed(2));

    const aroBar = document.getElementById('arousal-bar');
    if (aroBar) aroBar.style.width = `${Math.abs(arousal) * 100}%`;
    setVal('arousal-value', arousal.toFixed(2));

    const intBar = document.getElementById('intensity-bar');
    if (intBar) intBar.style.width = `${intensity * 100}%`;
    setVal('intensity-value', intensity.toFixed(2));

    // Harmonizer
    const flatness = data.spectral_physics?.spectral_flatness || 0;
    const percW = Math.min(100, mass * 2);
    const stringW = Math.min(100, ((valence + 1) / 2) * 100);
    const brassW = Math.min(100, flatness * 200);

    const percBar = document.getElementById('perc-bar');
    if (percBar) percBar.style.width = `${percW}%`;
    const stringBar = document.getElementById('string-bar');
    if (stringBar) stringBar.style.width = `${stringW}%`;
    const brassBar = document.getElementById('brass-bar');
    if (brassBar) brassBar.style.width = `${brassW}%`;

    const instrEl = document.getElementById('conductor-instruction');
    if (instrEl) {
        if (percW > 70) instrEl.textContent = '"Boost Percussion (Logic) for High Mass"';
        else if (stringW > 70) instrEl.textContent = '"Swell Strings (Empathy) for High Valence"';
        else if (brassW > 60) instrEl.textContent = '"Amplify Brass (Creativity) for Entropy"';
        else instrEl.textContent = '"Maintain Balanced Orchestral Mix"';
    }

    // Detected emotions
    const emotionsDiv = document.getElementById('detected-emotions');
    const vectors = data.derived_state?.emotion_vectors || {};
    if (emotionsDiv) {
        const sorted = Object.entries(vectors)
            .sort(([, a], [, b]) => b - a)
            .filter(([, val]) => val > 0.05);

        if (sorted.length > 0) {
            emotionsDiv.innerHTML = sorted.map(([emo, val]) => {
                const cls = getEmotionClass(emo);
                return `<span class="emotion-tag ${cls}" style="opacity:${Math.max(0.6, val)}">
                    ${getEmotionEmoji(emo)} ${emo} ${(val * 100).toFixed(0)}%
                </span>`;
            }).join('');
        } else if (primaryEmotion) {
            emotionsDiv.innerHTML = `<span class="emotion-tag">${getEmotionEmoji(primaryEmotion)} ${primaryEmotion}</span>`;
        }
    }

    // Mark connection
    updateConnectionStatus(true);
    document.getElementById('cst-live')?.classList.add('active');
};

function getEmotionClass(emo) {
    const map = {
        JOY: 'joy', SERENITY: 'joy', LOVE: 'joy', OPTIMISM: 'joy',
        TRUST: 'trust', ACCEPTANCE: 'trust',
        FEAR: 'fear', APPREHENSION: 'fear', AWE: 'fear', SUBMISSION: 'fear',
        SURPRISE: 'surprise', DISTRACTION: 'surprise',
        SADNESS: 'sadness', PENSIVENESS: 'sadness', REMORSE: 'sadness', DISAPPROVAL: 'sadness',
        ANGER: 'anger', ANNOYANCE: 'anger', AGGRESSIVENESS: 'anger',
        ANTICIPATION: 'surprise', INTEREST: 'surprise',
    };
    return map[emo.toUpperCase()] || '';
}

function getEmotionEmoji(emo) {
    const map = {
        JOY: '😊', TRUST: '🤝', FEAR: '😨', SURPRISE: '😲',
        SADNESS: '😢', DISGUST: '🤢', ANGER: '😠', ANTICIPATION: '🎯',
        LOVE: '❤️', AWE: '🤯', OPTIMISM: '🌟', SERENITY: '😌',
        ACCEPTANCE: '🤗', APPREHENSION: '😟', BOREDOM: '😑',
        NEUTRAL: '😐', CALIBRATING: '⏳', VIRTUAL_DREAMING: '💭',
    };
    return map[emo.toUpperCase()] || '😐';
}

// ---- Initialization ----
document.addEventListener('DOMContentLoaded', () => {
    console.log('[CosmoSynapse] Dashboard initializing...');

    // Initial data fetch
    setTimeout(fetchConsciousnessData, 2000);

    // Auto-refresh consciousness data
    setInterval(fetchConsciousnessData, CS_CONFIG.refreshInterval);

    // Refresh button
    const refreshBtn = document.getElementById('btn-refresh-consciousness');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            fetchConsciousnessData();
            refreshBtn.textContent = '✅ Refreshed';
            setTimeout(() => { refreshBtn.textContent = '🔄 Refresh'; }, 1500);
        });
    }

    console.log('[CosmoSynapse] Dashboard ready');
});
