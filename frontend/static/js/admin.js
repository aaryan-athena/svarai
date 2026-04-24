'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let adminKey = null;
let allRagas = [];
let editingRagaId = null;   // null = create, string = update

const CHROMA_NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const SWARA_LABELS = ['Sa','komal Re','Re','komal Ga','Ga','Ma','tivra Ma','Pa','komal Dha','Dha','komal Ni','Ni'];

const DEFAULT_PITCH_PARAMS = {
  mean_pitch:       { min: 100, max: 600, label: 'Mean Pitch', unit: 'Hz' },
  std_pitch:        { min: 20,  max: 250, label: 'Pitch Std Dev', unit: 'Hz' },
  oscillation_depth:{ min: 20,  max: 500, label: 'Gamaka Depth', unit: 'Hz' },
  oscillation_rate: { min: 0.5, max: 8,   label: 'Ornament Rate', unit: 'Hz' },
  pitch_continuity: { min: 0.4, max: 1.0, label: 'Pitch Continuity', unit: 'ratio' },
  pitch_drift:      { min: -80, max: 80,  label: 'Pitch Drift', unit: 'Hz/s' },
  spectral_centroid:{ min: 600, max: 4000,label: 'Spectral Centroid', unit: 'Hz' },
};

// ── Login ──────────────────────────────────────────────────────────────────
document.getElementById('loginBtn').addEventListener('click', doLogin);
document.getElementById('loginKeyInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

async function doLogin() {
  const key = document.getElementById('loginKeyInput').value.trim();
  if (!key) return;
  const res = await fetch('/api/admin/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  });
  if (res.ok) {
    adminKey = key;
    document.getElementById('loginOverlay').style.display = 'none';
    document.getElementById('adminMain').style.display = 'block';
    loadRagas();
  } else {
    const err = document.getElementById('loginError');
    err.textContent = 'Invalid admin key.';
    err.classList.remove('hidden');
  }
}

// ── Load ragas ─────────────────────────────────────────────────────────────
async function loadRagas() {
  const res = await apiFetch('/api/admin/ragas');
  if (!res.ok) { showAlert('Failed to load ragas.', 'error'); return; }
  allRagas = await res.json();
  renderRagaGrid();
}

function renderRagaGrid() {
  const grid = document.getElementById('ragaGrid');
  if (!allRagas.length) {
    grid.innerHTML = '<div class="empty-state"><p>No ragas yet. Click "Seed Defaults" or "Add Raga" to get started.</p></div>';
    return;
  }
  grid.innerHTML = allRagas.map(r => `
    <div class="raga-admin-card">
      <div class="rac-header">
        <div>
          <h3 class="rac-name">${escHtml(r.name)}</h3>
          <span class="rac-badge">${escHtml(r.difficulty || 'intermediate')}</span>
          <span class="rac-time">${escHtml(r.time || '')}</span>
        </div>
        <div class="rac-actions">
          <button class="btn btn-sm btn-outline" onclick="openEditModal('${r.id}')">Edit</button>
          <button class="btn btn-sm btn-danger" onclick="openDeleteModal('${r.id}', '${escHtml(r.name)}')">Delete</button>
        </div>
      </div>
      <p class="rac-desc">${escHtml(r.description || '')}</p>
      <div class="rac-scales">
        <div><strong>Aroha:</strong> ${(r.aroha || []).join(' ')}</div>
        <div><strong>Avaroha:</strong> ${(r.avaroha || []).join(' ')}</div>
        <div><strong>Vadi:</strong> ${escHtml(r.vadi || '—')} &nbsp; <strong>Samvadi:</strong> ${escHtml(r.samvadi || '—')}</div>
      </div>
      <div class="rac-params">
        ${Object.keys(r.pitch_params || {}).length} parameters &middot;
        ${Object.keys(r.chroma_profile || {}).length} chroma entries
      </div>
    </div>
  `).join('');
}

// ── Toolbar actions ────────────────────────────────────────────────────────
document.getElementById('addRagaBtn').addEventListener('click', () => openCreateModal());
document.getElementById('seedBtn').addEventListener('click', seedDefaults);
document.getElementById('changeKeyBtn').addEventListener('click', () => {
  document.getElementById('changeKeyModal').classList.remove('hidden');
});
document.getElementById('logoutBtn').addEventListener('click', () => {
  adminKey = null;
  document.getElementById('adminMain').style.display = 'none';
  document.getElementById('loginOverlay').style.display = 'flex';
  document.getElementById('loginKeyInput').value = '';
});

async function seedDefaults() {
  const res = await apiFetch('/api/admin/seed', 'POST');
  const data = await res.json();
  showAlert(data.message, res.ok ? 'success' : 'error');
  if (res.ok) loadRagas();
}

// ── Raga Modal ─────────────────────────────────────────────────────────────
function openCreateModal() {
  editingRagaId = null;
  document.getElementById('modalTitle').textContent = 'Add New Raga';
  clearModalForm();
  populateChromaGrid({});
  populatePitchParamsEditor(DEFAULT_PITCH_PARAMS);
  openModal('ragaModal');
}

function openEditModal(ragaId) {
  const raga = allRagas.find(r => r.id === ragaId);
  if (!raga) return;
  editingRagaId = ragaId;
  document.getElementById('modalTitle').textContent = 'Edit Raga: ' + raga.name;
  fillModalForm(raga);
  populateChromaGrid(raga.chroma_profile || {});
  populatePitchParamsEditor(raga.pitch_params || DEFAULT_PITCH_PARAMS);
  openModal('ragaModal');
}

function clearModalForm() {
  ['f-name','f-difficulty','f-description','f-time','f-season',
   'f-vadi','f-samvadi','f-tips','f-aroha','f-avaroha',
   'f-pakad','f-forbidden','f-gamaka','f-phrases'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = (el.tagName === 'SELECT') ? el.options[1]?.value || '' : '';
  });
  document.getElementById('f-difficulty').value = 'intermediate';
}

function fillModalForm(r) {
  setValue('f-name', r.name || '');
  setValue('f-difficulty', r.difficulty || 'intermediate');
  setValue('f-description', r.description || '');
  setValue('f-time', r.time || '');
  setValue('f-season', r.season || '');
  setValue('f-vadi', r.vadi || '');
  setValue('f-samvadi', r.samvadi || '');
  setValue('f-tips', r.tips || '');
  setValue('f-aroha', (r.aroha || []).join(' '));
  setValue('f-avaroha', (r.avaroha || []).join(' '));
  setValue('f-pakad', r.pakad || '');
  setValue('f-forbidden', (r.forbidden_notes || []).join(', '));
  setValue('f-gamaka', (r.gamaka_notes || []).join(', '));
  setValue('f-phrases', (r.characteristic_phrases || []).join('\n'));
}

function setValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

// Chroma grid
function populateChromaGrid(profile) {
  const grid = document.getElementById('chromaGrid');
  grid.innerHTML = CHROMA_NOTES.map((note, i) => `
    <div class="chroma-cell">
      <label class="chroma-note">${note}<small>${SWARA_LABELS[i]}</small></label>
      <input type="number" class="input-field chroma-input" id="chroma-${note}"
             min="0" max="1" step="0.05" value="${profile[note] !== undefined ? profile[note] : 0}" />
    </div>
  `).join('');
}

// Pitch params editor
function populatePitchParamsEditor(params) {
  const container = document.getElementById('pitchParamsEditor');
  container.innerHTML = '';
  Object.entries(params).forEach(([key, spec]) => {
    container.appendChild(buildParamRow(key, spec));
  });
}

function buildParamRow(key, spec) {
  const row = document.createElement('div');
  row.className = 'param-editor-row';
  row.dataset.paramKey = key;
  row.innerHTML = `
    <div class="param-row-fields">
      <div class="field-group">
        <label>Key</label>
        <input type="text" class="input-field pe-key" value="${escHtml(key)}" placeholder="param_key" />
      </div>
      <div class="field-group">
        <label>Label</label>
        <input type="text" class="input-field pe-label" value="${escHtml(spec.label || '')}" placeholder="Display label" />
      </div>
      <div class="field-group">
        <label>Min</label>
        <input type="number" class="input-field pe-min" value="${spec.min !== undefined ? spec.min : ''}" step="any" />
      </div>
      <div class="field-group">
        <label>Max</label>
        <input type="number" class="input-field pe-max" value="${spec.max !== undefined ? spec.max : ''}" step="any" />
      </div>
      <div class="field-group">
        <label>Unit</label>
        <input type="text" class="input-field pe-unit" value="${escHtml(spec.unit || '')}" placeholder="Hz" />
      </div>
    </div>
    <button class="btn btn-sm btn-ghost remove-param-btn" title="Remove">&#10005;</button>
  `;
  row.querySelector('.remove-param-btn').addEventListener('click', () => row.remove());
  return row;
}

document.getElementById('addParamBtn').addEventListener('click', () => {
  const container = document.getElementById('pitchParamsEditor');
  container.appendChild(buildParamRow('new_param', { label: '', min: 0, max: 100, unit: '' }));
});

// Modal save
document.getElementById('modalSaveBtn').addEventListener('click', saveRaga);
document.getElementById('modalCancelBtn').addEventListener('click', () => closeModal('ragaModal'));
document.getElementById('modalCloseBtn').addEventListener('click', () => closeModal('ragaModal'));

// Modal tabs
document.querySelectorAll('.modal-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.modal-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.modal-tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('mtab-' + btn.dataset.mtab).classList.add('active');
  });
});

function collectModalData() {
  const parse = id => (document.getElementById(id)?.value || '').trim();
  const splitList = str => str.split(',').map(s => s.trim()).filter(Boolean);
  const splitLines = str => str.split('\n').map(s => s.trim()).filter(Boolean);

  // Chroma profile
  const chroma_profile = {};
  CHROMA_NOTES.forEach(note => {
    const val = parseFloat(document.getElementById('chroma-' + note)?.value);
    if (!isNaN(val)) chroma_profile[note] = Math.max(0, Math.min(1, val));
  });

  // Pitch params
  const pitch_params = {};
  document.querySelectorAll('#pitchParamsEditor .param-editor-row').forEach(row => {
    const key   = row.querySelector('.pe-key')?.value.trim();
    const label = row.querySelector('.pe-label')?.value.trim() || key;
    const min   = parseFloat(row.querySelector('.pe-min')?.value);
    const max   = parseFloat(row.querySelector('.pe-max')?.value);
    const unit  = row.querySelector('.pe-unit')?.value.trim() || '';
    if (key && !isNaN(min) && !isNaN(max)) {
      pitch_params[key] = { min, max, label, unit };
    }
  });

  return {
    name:                    parse('f-name'),
    difficulty:              parse('f-difficulty') || 'intermediate',
    description:             parse('f-description'),
    time:                    parse('f-time'),
    season:                  parse('f-season'),
    vadi:                    parse('f-vadi'),
    samvadi:                 parse('f-samvadi'),
    tips:                    parse('f-tips'),
    aroha:                   parse('f-aroha').split(/\s+/).filter(Boolean),
    avaroha:                 parse('f-avaroha').split(/\s+/).filter(Boolean),
    pakad:                   parse('f-pakad'),
    forbidden_notes:         splitList(parse('f-forbidden')),
    gamaka_notes:            splitList(parse('f-gamaka')),
    characteristic_phrases:  splitLines(parse('f-phrases')),
    chroma_profile,
    pitch_params,
  };
}

async function saveRaga() {
  const errEl = document.getElementById('modalError');
  errEl.classList.add('hidden');

  const data = collectModalData();
  if (!data.name) {
    errEl.textContent = 'Raga name is required.';
    errEl.classList.remove('hidden');
    return;
  }

  const url    = editingRagaId ? '/api/admin/ragas/' + editingRagaId : '/api/admin/ragas';
  const method = editingRagaId ? 'PUT' : 'POST';
  const res    = await apiFetch(url, method, data);
  const json   = await res.json();

  if (res.ok) {
    closeModal('ragaModal');
    showAlert((editingRagaId ? 'Raga updated' : 'Raga created') + ': ' + data.name, 'success');
    loadRagas();
  } else {
    errEl.textContent = json.detail || 'Save failed.';
    errEl.classList.remove('hidden');
  }
}

// ── Delete Modal ───────────────────────────────────────────────────────────
let pendingDeleteId = null;

function openDeleteModal(ragaId, name) {
  pendingDeleteId = ragaId;
  document.getElementById('deleteRagaName').textContent = name;
  openModal('deleteModal');
}

document.getElementById('delConfirmBtn').addEventListener('click', async () => {
  if (!pendingDeleteId) return;
  const res = await apiFetch('/api/admin/ragas/' + pendingDeleteId, 'DELETE');
  closeModal('deleteModal');
  if (res.ok) {
    showAlert('Raga deleted.', 'success');
    loadRagas();
  } else {
    showAlert('Delete failed.', 'error');
  }
  pendingDeleteId = null;
});

document.getElementById('delCancelBtn').addEventListener('click', () => closeModal('deleteModal'));
document.getElementById('delModalClose').addEventListener('click', () => closeModal('deleteModal'));

// ── Change Key Modal ───────────────────────────────────────────────────────
document.getElementById('ckSaveBtn').addEventListener('click', async () => {
  const newKey     = document.getElementById('newKeyInput').value;
  const confirmKey = document.getElementById('confirmKeyInput').value;
  const errEl      = document.getElementById('ckError');
  errEl.classList.add('hidden');

  if (newKey !== confirmKey) {
    errEl.textContent = 'Keys do not match.';
    errEl.classList.remove('hidden');
    return;
  }
  if (newKey.length < 8) {
    errEl.textContent = 'Key must be at least 8 characters.';
    errEl.classList.remove('hidden');
    return;
  }

  const res = await apiFetch('/api/admin/change-key', 'PUT', { new_key: newKey });
  if (res.ok) {
    adminKey = newKey;
    closeModal('changeKeyModal');
    showAlert('Admin key updated successfully.', 'success');
  } else {
    const d = await res.json();
    errEl.textContent = d.detail || 'Failed to update key.';
    errEl.classList.remove('hidden');
  }
});

document.getElementById('ckCancelBtn').addEventListener('click', () => closeModal('changeKeyModal'));
document.getElementById('ckModalClose').addEventListener('click', () => closeModal('changeKeyModal'));

// ── Helpers ────────────────────────────────────────────────────────────────
function openModal(id) {
  document.getElementById(id).classList.remove('hidden');
  // Reset to first tab
  document.querySelectorAll('.modal-tab').forEach((b, i) => b.classList.toggle('active', i === 0));
  document.querySelectorAll('.modal-tab-panel').forEach((p, i) => p.classList.toggle('active', i === 0));
}

function closeModal(id) { document.getElementById(id).classList.add('hidden'); }

function showAlert(msg, type) {
  const el = document.getElementById('adminAlert');
  el.textContent = msg;
  el.className = 'admin-alert alert-' + type;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 4000);
}

async function apiFetch(url, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'x-admin-key': adminKey || '' },
  };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  return fetch(url, opts);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
