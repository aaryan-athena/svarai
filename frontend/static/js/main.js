'use strict';

/* ══════════════════════════════════════════════════════════════════════════
   SvarAI — main.js
   Single-page app: navigation + practice logic + raga education + progress
═══════════════════════════════════════════════════════════════════════════ */

// ── Constants ───────────────────────────────────────────────────────────────
const NAV_ITEMS = ['home', 'practice', 'learn', 'progress', 'settings'];
const PAGE_SUBTITLES = {
  home:     'AI Tutor for Indian Classical Music',
  practice: 'Submit Your Performance',
  learn:    'Raga Guide',
  progress: 'Your Progress',
  settings: 'Customize Your Experience',
};
const STORAGE_KEYS = {
  currentPage: 'svarai_page',
  sessions:    'svarai_sessions',
  theme:       'svarai_theme',
  prefs:       'svarai_prefs',
};

// ── State ───────────────────────────────────────────────────────────────────
let currentPage      = 'home';
let activeTradition  = 'carnatic';
let learnSearchQuery = '';

// Practice state
let selectedFile   = null;
let recordedBlob   = null;
let activeTab      = 'upload';
let mediaRecorder  = null;
let recChunks      = [];
let recInterval    = null;
let recSeconds     = 0;
let audioCtx       = null;
let analyserNode   = null;
let animFrameId    = null;

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadPreferences();
  loadTheme();
  initNavigation();
  initPractice();
  initLearn();
  initProgress();
  initSettings();

  // Restore last page (default: home)
  const saved = sessionStorage.getItem(STORAGE_KEYS.currentPage) || 'home';
  navigateTo(saved, false);
});

// ══════════════════════════════════════════════════════════════════════════════
//  NAVIGATION
// ══════════════════════════════════════════════════════════════════════════════
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.page));
  });

  // Home page CTA → practice
  document.getElementById('heroStartBtn').addEventListener('click', () => navigateTo('practice'));
}

function navigateTo(page, save = true) {
  if (!NAV_ITEMS.includes(page)) page = 'home';
  currentPage = page;

  // Show/hide pages
  document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
  const pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.classList.add('active');

  // Update nav items
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === page);
  });

  // Slide indicator
  updateNavIndicator(page);

  // Update subtitle
  const subtitleEl = document.getElementById('pageSubtitle');
  if (subtitleEl) subtitleEl.textContent = PAGE_SUBTITLES[page] || '';

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Persist
  if (save) sessionStorage.setItem(STORAGE_KEYS.currentPage, page);

  // Page-specific refresh
  if (page === 'progress') renderProgress();
  if (page === 'learn') renderLearnPage();
}

function updateNavIndicator(page) {
  const items = document.querySelectorAll('.nav-item');
  const inner = document.querySelector('.nav-items');
  const indicator = document.getElementById('navIndicator');
  if (!indicator || !inner) return;

  const idx = NAV_ITEMS.indexOf(page);
  if (idx === -1) return;

  const totalItems = items.length;
  const widthPct   = 100 / totalItems;
  indicator.style.width = widthPct + '%';
  indicator.style.left  = (idx * widthPct) + '%';
}

// ══════════════════════════════════════════════════════════════════════════════
//  PRACTICE PAGE
// ══════════════════════════════════════════════════════════════════════════════
function initPractice() {
  loadRagaList();
  setupTabs();
  setupDropZone();
  setupRecord();
  document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
}

// ── Raga dropdown ────────────────────────────────────────────────────────────
async function loadRagaList() {
  try {
    const res = await fetch('/api/ragas');
    if (!res.ok) return;
    const ragas = await res.json();
    const sel = document.getElementById('targetRaga');
    ragas.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = r.name;
      sel.appendChild(opt);
    });
  } catch (_) {}
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll('#page-practice .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#page-practice .tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('#page-practice .tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeTab = btn.dataset.tab;
      document.getElementById('tab-' + activeTab).classList.add('active');
      refreshAnalyzeBtn();
    });
  });
}

// ── Drop zone ────────────────────────────────────────────────────────────────
function setupDropZone() {
  const browseBtn = document.getElementById('browseBtn');
  const fileInput = document.getElementById('fileInput');
  const dropZone  = document.getElementById('dropZone');

  browseBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0]);
  });
}

function setSelectedFile(file) {
  selectedFile = file;
  const info = document.getElementById('fileInfo');
  info.textContent = file.name + '  (' + (file.size / 1024).toFixed(0) + ' KB)';
  info.classList.remove('hidden');
  refreshAnalyzeBtn();
}

// ── Recording ────────────────────────────────────────────────────────────────
function setupRecord() {
  document.getElementById('recordBtn').addEventListener('click', toggleRecord);
}

async function toggleRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx    = new AudioContext();
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    audioCtx.createMediaStreamSource(stream).connect(analyserNode);
    drawWaveform();

    recChunks     = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => recChunks.push(e.data);
    mediaRecorder.onstop = finaliseRecording;
    mediaRecorder.start();

    recSeconds  = 0;
    recInterval = setInterval(() => {
      recSeconds++;
      document.getElementById('recTimer').textContent = formatTime(recSeconds);
    }, 1000);

    const btn = document.getElementById('recordBtn');
    btn.textContent = '⏹ Stop Recording';
    btn.classList.add('recording');
    document.getElementById('recAudioPreview').classList.add('hidden');
    recordedBlob = null;
    refreshAnalyzeBtn();
  } catch (err) {
    showStatus('Microphone access denied: ' + err.message, 'error');
  }
}

function stopRecording() {
  if (mediaRecorder) mediaRecorder.stop();
  clearInterval(recInterval);
  cancelAnimationFrame(animFrameId);
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  const btn = document.getElementById('recordBtn');
  btn.textContent = '⏺ Start Recording';
  btn.classList.remove('recording');
}

function finaliseRecording() {
  recordedBlob = new Blob(recChunks, { type: 'audio/webm' });
  const url = URL.createObjectURL(recordedBlob);
  document.getElementById('recAudioEl').src = url;
  document.getElementById('recAudioPreview').classList.remove('hidden');
  refreshAnalyzeBtn();
}

function drawWaveform() {
  const canvas = document.getElementById('waveCanvas');
  const ctx    = canvas.getContext('2d');
  const buf    = new Uint8Array(analyserNode.frequencyBinCount);

  function frame() {
    animFrameId = requestAnimationFrame(frame);
    analyserNode.getByteTimeDomainData(buf);

    ctx.fillStyle = 'hsl(20 18% 6%)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth   = 2;
    ctx.strokeStyle = 'hsl(32, 85%, 54%)'; // saffron
    ctx.shadowBlur  = 6;
    ctx.shadowColor = 'hsl(32, 85%, 54%)';
    ctx.beginPath();

    const sliceW = canvas.width / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const y = (buf[i] / 128) * (canvas.height / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  frame();
}

// ── Analyse ──────────────────────────────────────────────────────────────────
function refreshAnalyzeBtn() {
  document.getElementById('analyzeBtn').disabled =
    activeTab === 'upload' ? !selectedFile : !recordedBlob;
}

async function runAnalysis() {
  const formData   = new FormData();
  const targetRaga = document.getElementById('targetRaga').value;
  const aiToggle   = document.getElementById('aiToggle').checked;

  if (activeTab === 'upload') {
    formData.append('file', selectedFile);
  } else {
    formData.append('file', recordedBlob, 'recording.webm');
  }
  if (targetRaga) formData.append('target_raga', targetRaga);
  formData.append('include_ai_feedback', aiToggle ? 'true' : 'false');

  document.getElementById('analyzeBtn').disabled = true;
  showStatus('Extracting audio features…', 'loading');
  document.getElementById('resultsSection').classList.add('hidden');

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      showStatus('Error: ' + (data.detail || 'Analysis failed'), 'error');
      return;
    }
    hideStatus();
    renderResults(data);
    saveSession(data);
  } catch (err) {
    showStatus('Network error: ' + err.message, 'error');
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const resultsSection = document.getElementById('resultsSection');
  resultsSection.classList.remove('hidden');
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);

  const bestMatch = data.best_match || 'Unknown';
  document.getElementById('resultRagaName').textContent = bestMatch;

  const conf   = data.identification_confidence;
  const confEl = document.getElementById('resultConfidence');
  confEl.textContent = conf === 'high' ? 'High confidence' : conf === 'medium' ? 'Medium confidence' : 'Low confidence';
  confEl.className   = 'result-confidence conf-' + conf;

  const banner = document.getElementById('targetMatchBanner');
  if (data.target_raga) {
    const sel        = document.getElementById('targetRaga');
    const targetName = sel.options[sel.selectedIndex]?.text || data.target_raga;
    const matched    = bestMatch.toLowerCase() === (data.ai_feedback_raga || '').toLowerCase()
                    || bestMatch.toLowerCase() === targetName.toLowerCase();
    banner.textContent = matched
      ? '✓ Matches your target: ' + targetName
      : 'Practising: ' + targetName + ' — Best match: ' + bestMatch;
    banner.className = 'target-match-banner ' + (matched ? 'match-yes' : 'match-no');
    banner.classList.remove('hidden');
  } else {
    banner.classList.add('hidden');
  }

  // AI Tutor
  const tutorCard = document.getElementById('tutorCard');
  if (data.ai_feedback && !data.ai_feedback.error) {
    const fb = data.ai_feedback;
    tutorCard.classList.remove('hidden');

    document.getElementById('tutorRagaLabel').textContent =
      'Feedback for: ' + (data.ai_feedback_raga || bestMatch);
    document.getElementById('tutorAssessment').textContent = fb.overall_assessment || '';

    const score    = Math.round(fb.score || 0);
    document.getElementById('ringValue').textContent = score;
    const ringFill = document.getElementById('ringFill');
    ringFill.setAttribute('stroke-dasharray', score + ', 100');
    ringFill.style.stroke = score >= 75 ? 'var(--green)' : score >= 50 ? 'var(--yellow)' : 'var(--red)';

    const ctxBox = document.getElementById('ragaContextBox');
    if (fb.raga_context) {
      document.getElementById('ragaContextText').textContent = fb.raga_context;
      ctxBox.classList.remove('hidden');
    } else {
      ctxBox.classList.add('hidden');
    }

    const posSection = document.getElementById('positiveSection');
    const posList    = document.getElementById('positiveList');
    if (fb.positive_aspects?.length) {
      posList.innerHTML = fb.positive_aspects.map(p => '<li>' + escHtml(p) + '</li>').join('');
      posSection.classList.remove('hidden');
    } else {
      posSection.classList.add('hidden');
    }

    const devSection = document.getElementById('deviationsSection');
    const devList    = document.getElementById('deviationsList');
    if (fb.deviations?.length) {
      devList.innerHTML = fb.deviations.map(d =>
        '<div class="deviation-item sev-' + d.severity + '">' +
          '<div class="dev-header">' +
            '<span class="dev-param">' + escHtml(d.parameter) + '</span>' +
            '<span class="dev-badge sev-' + d.severity + '">' + d.severity + '</span>' +
          '</div>' +
          '<p class="dev-issue">' + escHtml(d.issue) + '</p>' +
          '<p class="dev-suggestion"><strong>Fix:</strong> ' + escHtml(d.suggestion) + '</p>' +
        '</div>'
      ).join('');
      devSection.classList.remove('hidden');
    } else {
      devSection.classList.add('hidden');
    }

    const tipsSection = document.getElementById('tipsSection');
    const tipsList    = document.getElementById('tipsList');
    if (fb.practice_tips?.length) {
      tipsList.innerHTML = fb.practice_tips.map(t => '<li>' + escHtml(t) + '</li>').join('');
      tipsSection.classList.remove('hidden');
    } else {
      tipsSection.classList.add('hidden');
    }
  } else {
    tutorCard.classList.add('hidden');
  }

  // Rankings
  const ranked     = data.ranked_matches || [];
  const rankingsEl = document.getElementById('rankingsList');
  rankingsEl.innerHTML = ranked.map((r, i) =>
    '<div class="ranking-row ' + (i === 0 ? 'top-rank' : '') + '">' +
      '<span class="rank-pos">' + (i + 1) + '</span>' +
      '<span class="rank-name">' + escHtml(r.name) + '</span>' +
      '<div class="rank-scores">' +
        '<span class="rank-score-pill" title="Overall">' + r.overall_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Chroma">C:' + r.chroma_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Pitch">P:' + r.pitch_params_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Note">N:' + r.note_prominence_score.toFixed(0) + '</span>' +
      '</div>' +
      '<div class="rank-bar-wrap"><div class="rank-bar" style="width:' + r.overall_score + '%"></div></div>' +
    '</div>'
  ).join('');

  // Per-param breakdown
  if (ranked.length) {
    const top     = ranked[0];
    const details = top.param_details || {};
    document.getElementById('paramCardRagaName').textContent = top.name;
    document.getElementById('paramBreakdown').innerHTML = Object.entries(details).map(([key, d]) =>
      '<div class="param-item ' + (d.in_range ? 'param-ok' : 'param-off') + '">' +
        '<div class="param-name">' + escHtml(d.label || key) + '</div>' +
        '<div class="param-value">' + d.value + ' <small>' + escHtml(d.unit || '') + '</small></div>' +
        '<div class="param-range">Range: ' + d.min + ' – ' + d.max + '</div>' +
        '<div class="param-status">' + (d.in_range ? '✓ In range' : '✗ Out of range') + '</div>' +
      '</div>'
    ).join('');
  }

  // Acoustic features
  const features    = data.features || {};
  const featureKeys = [
    ['mean_pitch',         'Mean Pitch (Hz)'],
    ['std_pitch',          'Pitch Std Dev (Hz)'],
    ['min_pitch',          'Min Pitch (Hz)'],
    ['max_pitch',          'Max Pitch (Hz)'],
    ['oscillation_depth',  'Gamaka Depth (Hz)'],
    ['oscillation_rate',   'Ornament Rate (Hz)'],
    ['pitch_continuity',   'Pitch Continuity'],
    ['pitch_drift',        'Pitch Drift (Hz/s)'],
    ['spectral_centroid',  'Spectral Centroid (Hz)'],
    ['rms_energy',         'RMS Energy'],
    ['zero_crossing_rate', 'Zero Crossing Rate'],
    ['duration_sec',       'Duration (s)'],
  ];
  document.getElementById('featuresGrid').innerHTML = featureKeys
    .filter(([k]) => features[k] !== undefined)
    .map(([k, label]) =>
      '<div class="feat-item">' +
        '<div class="feat-label">' + escHtml(label) + '</div>' +
        '<div class="feat-value">' + (typeof features[k] === 'number' ? features[k].toFixed(3) : features[k]) + '</div>' +
      '</div>'
    ).join('');
}

// ══════════════════════════════════════════════════════════════════════════════
//  LEARN PAGE
// ══════════════════════════════════════════════════════════════════════════════
function initLearn() {
  // Tradition tabs
  document.getElementById('carnaticTab').addEventListener('click', () => {
    setTradition('carnatic');
  });
  document.getElementById('hindustaniTab').addEventListener('click', () => {
    setTradition('hindustani');
  });

  // Search
  document.getElementById('ragaSearch').addEventListener('input', e => {
    learnSearchQuery = e.target.value;
    renderLearnPage();
  });

  renderLearnPage();
}

function setTradition(t) {
  activeTradition  = t;
  learnSearchQuery = '';
  document.getElementById('ragaSearch').value = '';

  document.querySelectorAll('#page-learn .tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tradition === t);
  });
  renderLearnPage();
}

function renderLearnPage() {
  const data  = (window.RAGA_EDUCATION || {})[activeTradition] || [];
  const query = learnSearchQuery.toLowerCase().trim();

  const filtered = query
    ? data.filter(r =>
        r.name.toLowerCase().includes(query) ||
        (r.alternateNames || []).some(n => n.toLowerCase().includes(query)) ||
        r.mood.toLowerCase().includes(query)
      )
    : data;

  const container = document.getElementById('ragaCardsList');
  if (!filtered.length) {
    container.innerHTML = '<div class="raga-no-results">No ragas found matching "' + escHtml(query) + '"</div>';
    return;
  }

  container.innerHTML = filtered.map(r => buildRagaCard(r)).join('');

  // Attach accordion toggles
  container.querySelectorAll('.acc-trigger').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const item = trigger.closest('.acc-item');
      item.classList.toggle('open');
    });
  });
}

function buildRagaCard(r) {
  const altNames = r.alternateNames?.length
    ? '<p class="rec-alt">Also known as: ' + escHtml(r.alternateNames.join(', ')) + '</p>'
    : '';

  const timeRow = r.timeOfDay
    ? `<div class="rec-time">
        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        ${escHtml(r.timeOfDay)}
      </div>`
    : '';

  const movements = (r.characteristicMovements || []).map(m =>
    '<li><span class="acc-bullet">•</span><span>' + escHtml(m) + '</span></li>'
  ).join('');

  const compositions = (r.famousCompositions || []).map(c => {
    let line = '<strong>' + escHtml(c.title) + '</strong>';
    if (c.composer || c.type) {
      line += ' — ';
      if (c.composer) line += escHtml(c.composer);
      if (c.composer && c.type) line += ', ';
      if (c.type) line += '<em>' + escHtml(c.type) + '</em>';
    }
    return '<p class="acc-comp-item">' + line + '</p>';
  }).join('');

  return `
<div class="raga-edu-card">
  <div class="rec-header">
    <div class="rec-header-top">
      <div>
        <h3 class="rec-name">${escHtml(r.name)}</h3>
        ${altNames}
      </div>
      <span class="rec-tradition-badge">${escHtml(r.tradition)}</span>
    </div>
    ${timeRow}
  </div>

  <div class="rec-mood">
    <div class="rec-mood-label">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
      Mood &amp; Character
    </div>
    <p class="rec-mood-text">${escHtml(r.mood)}</p>
  </div>

  <div class="rec-accordion">
    <div class="acc-item">
      <button class="acc-trigger">
        <span class="acc-trigger-label">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          Historical Background
        </span>
        <svg class="acc-chevron" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="acc-content"><p>${escHtml(r.historicalBackground)}</p></div>
    </div>

    <div class="acc-item">
      <button class="acc-trigger">
        <span class="acc-trigger-label">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
          Characteristic Movements
        </span>
        <svg class="acc-chevron" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="acc-content"><ul>${movements}</ul></div>
    </div>

    <div class="acc-item">
      <button class="acc-trigger">
        <span class="acc-trigger-label">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          Famous Compositions
        </span>
        <svg class="acc-chevron" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="acc-content">${compositions}</div>
    </div>

    <div class="acc-item">
      <button class="acc-trigger">
        <span class="acc-trigger-label">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/></svg>
          Listening Tips
        </span>
        <svg class="acc-chevron" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="acc-content"><p class="acc-listening">${escHtml(r.listeningTips)}</p></div>
    </div>
  </div>
</div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
//  PROGRESS PAGE
// ══════════════════════════════════════════════════════════════════════════════
function initProgress() {
  renderProgress();
}

function renderProgress() {
  const sessions = getSessions();

  document.getElementById('pmSessions').textContent = sessions.length;

  const uniqueRagas = new Set(sessions.map(s => s.raga));
  document.getElementById('pmRagas').textContent = uniqueRagas.size;

  const scored = sessions.filter(s => s.score != null);
  if (scored.length) {
    const avg = scored.reduce((sum, s) => sum + s.score, 0) / scored.length;
    document.getElementById('pmAvgScore').textContent = Math.round(avg);
  } else {
    document.getElementById('pmAvgScore').textContent = '—';
  }

  const listEl = document.getElementById('recentSessionsList');
  if (!sessions.length) {
    listEl.innerHTML = '<div class="empty-state"><p>No sessions yet — complete your first analysis on the Practice page!</p></div>';
    return;
  }

  // Show most recent 10, newest first
  const recent = [...sessions].reverse().slice(0, 10);
  listEl.innerHTML = recent.map(s => {
    const date = new Date(s.timestamp);
    const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
                    ' · ' + date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    const scoreHtml = s.score != null
      ? '<span class="session-score">' + Math.round(s.score) + '</span>'
      : '';
    return `<div class="session-row">
      <div>
        <div class="session-raga">${escHtml(s.raga)}</div>
        <div class="session-date">${escHtml(dateStr)}</div>
      </div>
      ${scoreHtml}
    </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════════════════
//  SETTINGS PAGE
// ══════════════════════════════════════════════════════════════════════════════
function initSettings() {
  // Theme buttons
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setTheme(btn.dataset.theme);
    });
  });

  // Preference toggles
  const prefs = loadPreferences();
  const aiToggleEl  = document.getElementById('aiToggle');
  const prefAi      = document.getElementById('prefAiDefault');
  const prefWave    = document.getElementById('prefWaveform');
  const prefHistory = document.getElementById('prefSaveHistory');

  if (prefAi)      prefAi.checked      = prefs.aiDefault !== false;
  if (prefWave)    prefWave.checked    = prefs.waveform !== false;
  if (prefHistory) prefHistory.checked = prefs.saveHistory !== false;

  // Keep aiToggle in sync with the preference
  if (aiToggleEl && prefs.aiDefault !== false) aiToggleEl.checked = true;

  [prefAi, prefWave, prefHistory].forEach(el => {
    if (el) el.addEventListener('change', savePreferences);
  });

  // Clear data
  document.getElementById('clearDataBtn').addEventListener('click', () => {
    if (confirm('Clear all session history and preferences? This cannot be undone.')) {
      localStorage.clear();
      sessionStorage.clear();
      renderProgress();
      alert('All data cleared.');
    }
  });

  // Apply saved theme
  loadTheme();
}

function setTheme(theme) {
  document.documentElement.dataset.ragaTheme = theme === 'default' ? '' : theme;
  localStorage.setItem(STORAGE_KEYS.theme, theme);

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.theme === theme);
  });
}

function loadTheme() {
  const saved = localStorage.getItem(STORAGE_KEYS.theme) || 'default';
  document.documentElement.dataset.ragaTheme = saved === 'default' ? '' : saved;

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.theme === saved);
  });
}

function savePreferences() {
  const prefs = {
    aiDefault:   document.getElementById('prefAiDefault')?.checked ?? true,
    waveform:    document.getElementById('prefWaveform')?.checked ?? true,
    saveHistory: document.getElementById('prefSaveHistory')?.checked ?? true,
  };
  localStorage.setItem(STORAGE_KEYS.prefs, JSON.stringify(prefs));
}

function loadPreferences() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.prefs) || '{}');
  } catch (_) { return {}; }
}

// ══════════════════════════════════════════════════════════════════════════════
//  SESSION STORAGE
// ══════════════════════════════════════════════════════════════════════════════
function saveSession(data) {
  const prefs = loadPreferences();
  if (prefs.saveHistory === false) return;

  const sessions = getSessions();
  sessions.push({
    raga:      data.best_match || 'Unknown',
    score:     data.ai_feedback?.score ?? null,
    timestamp: Date.now(),
  });

  // Keep last 100 sessions
  if (sessions.length > 100) sessions.splice(0, sessions.length - 100);
  localStorage.setItem(STORAGE_KEYS.sessions, JSON.stringify(sessions));
}

function getSessions() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.sessions) || '[]');
  } catch (_) { return []; }
}

// ══════════════════════════════════════════════════════════════════════════════
//  UTILITY HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function showStatus(msg, type) {
  const bar = document.getElementById('analysisStatus');
  bar.textContent = msg;
  bar.className   = 'status-bar status-' + type;
  bar.classList.remove('hidden');
}
function hideStatus() {
  document.getElementById('analysisStatus').classList.add('hidden');
}
function formatTime(s) {
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
