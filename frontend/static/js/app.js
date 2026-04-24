'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let recordedBlob = null;
let activeTab = 'upload';
let mediaRecorder = null;
let recChunks = [];
let recInterval = null;
let recSeconds = 0;
let audioCtx = null;
let analyserNode = null;
let animFrameId = null;

// ── DOM refs ───────────────────────────────────────────────────────────────
const targetRagaEl    = document.getElementById('targetRaga');
const dropZone        = document.getElementById('dropZone');
const browseBtn       = document.getElementById('browseBtn');
const fileInput       = document.getElementById('fileInput');
const fileInfo        = document.getElementById('fileInfo');
const recordBtn       = document.getElementById('recordBtn');
const recTimer        = document.getElementById('recTimer');
const recAudioPreview = document.getElementById('recAudioPreview');
const recAudioEl      = document.getElementById('recAudioEl');
const waveCanvas      = document.getElementById('waveCanvas');
const aiToggle        = document.getElementById('aiToggle');
const analyzeBtn      = document.getElementById('analyzeBtn');
const statusBar       = document.getElementById('analysisStatus');
const resultsSection  = document.getElementById('resultsSection');

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await loadRagaList();
  setupTabs();
  setupDropZone();
  setupRecord();
})();

async function loadRagaList() {
  try {
    const res = await fetch('/api/ragas');
    if (!res.ok) return;
    const ragas = await res.json();
    ragas.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = r.name;
      targetRagaEl.appendChild(opt);
    });
  } catch (_) {}
}

// ── Tabs ───────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      activeTab = btn.dataset.tab;
      document.getElementById('tab-' + activeTab).classList.add('active');
      refreshAnalyzeBtn();
    });
  });
}

// ── Drop zone ──────────────────────────────────────────────────────────────
function setupDropZone() {
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
  fileInfo.textContent = file.name + '  (' + (file.size / 1024).toFixed(0) + ' KB)';
  fileInfo.classList.remove('hidden');
  refreshAnalyzeBtn();
}

// ── Recording ──────────────────────────────────────────────────────────────
function setupRecord() {
  recordBtn.addEventListener('click', toggleRecord);
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
    audioCtx = new AudioContext();
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    audioCtx.createMediaStreamSource(stream).connect(analyserNode);
    drawWaveform();

    recChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => recChunks.push(e.data);
    mediaRecorder.onstop = finaliseRecording;
    mediaRecorder.start();

    recSeconds = 0;
    recInterval = setInterval(() => {
      recSeconds++;
      recTimer.textContent = formatTime(recSeconds);
    }, 1000);

    recordBtn.textContent = '⏹ Stop Recording';
    recordBtn.classList.add('recording');
    recAudioPreview.classList.add('hidden');
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
  recordBtn.textContent = '⏺ Start Recording';
  recordBtn.classList.remove('recording');
}

function finaliseRecording() {
  recordedBlob = new Blob(recChunks, { type: 'audio/webm' });
  const url = URL.createObjectURL(recordedBlob);
  recAudioEl.src = url;
  recAudioPreview.classList.remove('hidden');
  refreshAnalyzeBtn();
}

function drawWaveform() {
  const ctx = waveCanvas.getContext('2d');
  const buf = new Uint8Array(analyserNode.frequencyBinCount);
  function frame() {
    animFrameId = requestAnimationFrame(frame);
    analyserNode.getByteTimeDomainData(buf);
    ctx.fillStyle = '#0f1117';
    ctx.fillRect(0, 0, waveCanvas.width, waveCanvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#a78bfa';
    ctx.beginPath();
    const sliceW = waveCanvas.width / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const y = (buf[i] / 128) * (waveCanvas.height / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.stroke();
  }
  frame();
}

// ── Analyse ────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

function refreshAnalyzeBtn() {
  analyzeBtn.disabled = activeTab === 'upload' ? !selectedFile : !recordedBlob;
}

async function runAnalysis() {
  const formData = new FormData();
  if (activeTab === 'upload') {
    formData.append('file', selectedFile);
  } else {
    formData.append('file', recordedBlob, 'recording.webm');
  }
  const targetRaga = targetRagaEl.value;
  if (targetRaga) formData.append('target_raga', targetRaga);
  formData.append('include_ai_feedback', aiToggle.checked ? 'true' : 'false');

  analyzeBtn.disabled = true;
  showStatus('Extracting audio features…', 'loading');
  resultsSection.classList.add('hidden');

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      showStatus('Error: ' + (data.detail || 'Analysis failed'), 'error');
      return;
    }
    hideStatus();
    renderResults(data);
  } catch (err) {
    showStatus('Network error: ' + err.message, 'error');
  } finally {
    analyzeBtn.disabled = false;
  }
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  resultsSection.classList.remove('hidden');
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);

  const bestMatch = data.best_match || 'Unknown';
  document.getElementById('resultRagaName').textContent = bestMatch;

  const conf = data.identification_confidence;
  const confEl = document.getElementById('resultConfidence');
  confEl.textContent = conf === 'high' ? 'High confidence' : conf === 'medium' ? 'Medium confidence' : 'Low confidence';
  confEl.className = 'result-confidence conf-' + conf;

  const banner = document.getElementById('targetMatchBanner');
  if (data.target_raga) {
    const sel = targetRagaEl.options[targetRagaEl.selectedIndex];
    const targetName = sel ? sel.text : data.target_raga;
    const matched = bestMatch.toLowerCase() === (data.ai_feedback_raga || '').toLowerCase()
                 || bestMatch.toLowerCase() === targetName.toLowerCase();
    banner.textContent = matched
      ? '✓ Matches your target: ' + targetName
      : 'Practising: ' + targetName + ' — Best match: ' + bestMatch;
    banner.className = 'target-match-banner ' + (matched ? 'match-yes' : 'match-no');
    banner.classList.remove('hidden');
  } else {
    banner.classList.add('hidden');
  }

  // ── AI Tutor ──
  const tutorCard = document.getElementById('tutorCard');
  if (data.ai_feedback && !data.ai_feedback.error) {
    const fb = data.ai_feedback;
    tutorCard.classList.remove('hidden');

    document.getElementById('tutorRagaLabel').textContent =
      'Feedback for: ' + (data.ai_feedback_raga || bestMatch);
    document.getElementById('tutorAssessment').textContent = fb.overall_assessment || '';

    const score = Math.round(fb.score || 0);
    document.getElementById('ringValue').textContent = score;
    const ringFill = document.getElementById('ringFill');
    ringFill.setAttribute('stroke-dasharray', score + ', 100');
    ringFill.style.stroke = score >= 75 ? '#22c55e' : score >= 50 ? '#f59e0b' : '#ef4444';

    const ctxBox = document.getElementById('ragaContextBox');
    if (fb.raga_context) {
      document.getElementById('ragaContextText').textContent = fb.raga_context;
      ctxBox.classList.remove('hidden');
    } else {
      ctxBox.classList.add('hidden');
    }

    const posSection = document.getElementById('positiveSection');
    const posList    = document.getElementById('positiveList');
    if (fb.positive_aspects && fb.positive_aspects.length) {
      posList.innerHTML = fb.positive_aspects.map(p => '<li>' + escHtml(p) + '</li>').join('');
      posSection.classList.remove('hidden');
    } else {
      posSection.classList.add('hidden');
    }

    const devSection = document.getElementById('deviationsSection');
    const devList    = document.getElementById('deviationsList');
    if (fb.deviations && fb.deviations.length) {
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
    if (fb.practice_tips && fb.practice_tips.length) {
      tipsList.innerHTML = fb.practice_tips.map(t => '<li>' + escHtml(t) + '</li>').join('');
      tipsSection.classList.remove('hidden');
    } else {
      tipsSection.classList.add('hidden');
    }
  } else {
    tutorCard.classList.add('hidden');
  }

  // ── Rankings ──
  const ranked = data.ranked_matches || [];
  const rankingsEl = document.getElementById('rankingsList');
  rankingsEl.innerHTML = ranked.map((r, i) =>
    '<div class="ranking-row ' + (i === 0 ? 'top-rank' : '') + '">' +
      '<span class="rank-pos">' + (i + 1) + '</span>' +
      '<span class="rank-name">' + escHtml(r.name) + '</span>' +
      '<div class="rank-scores">' +
        '<span class="rank-score-pill" title="Overall">' + r.overall_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Chroma">C:' + r.chroma_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Pitch params">P:' + r.pitch_params_score.toFixed(0) + '</span>' +
        '<span class="score-detail" title="Note prominence">N:' + r.note_prominence_score.toFixed(0) + '</span>' +
      '</div>' +
      '<div class="rank-bar-wrap"><div class="rank-bar" style="width:' + r.overall_score + '%"></div></div>' +
    '</div>'
  ).join('');

  // ── Per-param breakdown ──
  if (ranked.length) {
    const top = ranked[0];
    document.getElementById('paramCardRagaName').textContent = top.name;
    const details = top.param_details || {};
    document.getElementById('paramBreakdown').innerHTML = Object.entries(details).map(([key, d]) =>
      '<div class="param-item ' + (d.in_range ? 'param-ok' : 'param-off') + '">' +
        '<div class="param-name">' + escHtml(d.label || key) + '</div>' +
        '<div class="param-value">' + d.value + ' <small>' + escHtml(d.unit || '') + '</small></div>' +
        '<div class="param-range">Range: ' + d.min + ' – ' + d.max + '</div>' +
        '<div class="param-status">' + (d.in_range ? '✓ In range' : '✗ Out of range') + '</div>' +
      '</div>'
    ).join('');
  }

  // ── Acoustic features ──
  const features = data.features || {};
  const featureKeys = [
    ['mean_pitch', 'Mean Pitch (Hz)'], ['std_pitch', 'Pitch Std Dev (Hz)'],
    ['min_pitch', 'Min Pitch (Hz)'],   ['max_pitch', 'Max Pitch (Hz)'],
    ['oscillation_depth', 'Gamaka Depth (Hz)'], ['oscillation_rate', 'Ornament Rate (Hz)'],
    ['pitch_continuity', 'Pitch Continuity'],   ['pitch_drift', 'Pitch Drift (Hz/s)'],
    ['spectral_centroid', 'Spectral Centroid (Hz)'], ['rms_energy', 'RMS Energy'],
    ['zero_crossing_rate', 'Zero Crossing Rate'], ['duration_sec', 'Duration (s)'],
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

// ── Helpers ────────────────────────────────────────────────────────────────
function showStatus(msg, type) {
  statusBar.textContent = msg;
  statusBar.className = 'status-bar status-' + type;
  statusBar.classList.remove('hidden');
}
function hideStatus() { statusBar.classList.add('hidden'); }
function formatTime(s) {
  return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
}
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
