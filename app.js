/* ═══════════════════════════════════════════════════════
   ResumeAid — app.js  (redesign)
   ═══════════════════════════════════════════════════════ */
'use strict';

const API_BASE = 'http://localhost:5000';

/* ── State ─────────────────────────────────────────────── */
const state = {
  lastResponse: null,
  selectedFile: null,
  country: localStorage.getItem('ra_country') || 'us',
  darkMode: localStorage.getItem('ra_dark') === 'true',
  analysisController: null,
};

/* ── DOM shortcuts ──────────────────────────────────────── */
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

/* ══════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════ */
function init() {
  // Dark mode: apply immediately, no flash
  applyDark(state.darkMode, false);
  $('dark-mode-toggle').checked = state.darkMode;
  $('country-display').textContent = state.country;

  // Splash → app
  const splashEl = $('splash');
  const appEl = $('app');
  setTimeout(() => {
    splashEl.classList.add('out');
    setTimeout(() => {
      splashEl.classList.add('hidden');
      appEl.classList.remove('hidden');
    }, 520);
  }, 2000);

  bindAll();
}

/* ══════════════════════════════════════════════════════════
   BIND EVENTS
   ══════════════════════════════════════════════════════════ */
function bindAll() {

  /* Sidebar nav */
  $$('.nav-item').forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.view)));

  /* Mobile top nav */
  $$('.mnav-btn').forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.view)));

  /* Drop zone */
  const dz = $('drop-zone');
  dz.addEventListener('click', () => $('file-input').click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); } });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
  $('file-input').addEventListener('change', () => { if ($('file-input').files[0]) onFile($('file-input').files[0]); });

  /* Remove file */
  $('file-pill-remove').addEventListener('click', e => { e.stopPropagation(); clearFile(); });

  /* Analyse */
  $('btn-analyse').addEventListener('click', runAnalysis);

  /* Resume check overlay */
  $('rc-close').addEventListener('click', () => closeOverlay('overlay-resume-check'));
  $('rc-backdrop').addEventListener('click', () => closeOverlay('overlay-resume-check'));
  $('btn-continue').addEventListener('click', () => closeOverlay('overlay-resume-check', () => switchView('matches')));

  /* Match detail modal */
  $('modal-match-close').addEventListener('click', () => closeModal('modal-match'));
  $('modal-match').addEventListener('click', e => { if (e.target === $('modal-match')) closeModal('modal-match'); });

  /* Sort matches */
  $('matches-sort').addEventListener('change', renderMatches);

  /* Settings */
  $('dark-mode-toggle').addEventListener('change', () => {
    state.darkMode = $('dark-mode-toggle').checked;
    localStorage.setItem('ra_dark', state.darkMode);
    applyDark(state.darkMode, true);
  });
  $('row-country').addEventListener('click', openCountryModal);
  $('row-country').addEventListener('keydown', e => { if (e.key === 'Enter') openCountryModal(); });
  $('row-about').addEventListener('click', () => openModal('modal-about'));
  $('row-about').addEventListener('keydown', e => { if (e.key === 'Enter') openModal('modal-about'); });
  $('row-privacy').addEventListener('click', () => toast('Privacy policy coming soon.'));
  $('row-privacy').addEventListener('keydown', e => { if (e.key === 'Enter') toast('Privacy policy coming soon.'); });

  /* Country modal */
  $('country-cancel').addEventListener('click', () => closeModal('modal-country'));
  $('modal-country-close').addEventListener('click', () => closeModal('modal-country'));
  $('modal-country').addEventListener('click', e => { if (e.target === $('modal-country')) closeModal('modal-country'); });
  $('country-save').addEventListener('click', saveCountry);
  $('country-input').addEventListener('keydown', e => { if (e.key === 'Enter') saveCountry(); });

  /* About modal */
  $('about-ok').addEventListener('click', () => closeModal('modal-about'));
  $('modal-about').addEventListener('click', e => { if (e.target === $('modal-about')) closeModal('modal-about'); });
}

/* ══════════════════════════════════════════════════════════
   VIEW SWITCHING
   ══════════════════════════════════════════════════════════ */
const VIEW_TITLES = { upload: 'Upload', matches: 'Matches', resume: 'Resume', settings: 'Settings' };

function switchView(viewId) {
  // Sidebar buttons
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === viewId));
  // Mobile nav
  $$('.mnav-btn').forEach(b => b.classList.toggle('active', b.dataset.view === viewId));
  // Sections
  $$('.view').forEach(v => v.classList.remove('active'));
  const el = $(`view-${viewId}`);
  if (el) el.classList.add('active');
  // Mobile title
  $('mobile-title').textContent = VIEW_TITLES[viewId] || '';
  // Populate on switch
  if (viewId === 'matches') renderMatches();
  if (viewId === 'resume') renderResume();
}

// Export so HTML inline handlers can call it
window.switchView = switchView;

/* ══════════════════════════════════════════════════════════
   FILE HANDLING
   ══════════════════════════════════════════════════════════ */
function onFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  const okTypes = ['application/pdf', 'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
  if (!okTypes.includes(file.type) && !['pdf', 'doc', 'docx'].includes(ext)) {
    toast('Please select a PDF, DOC or DOCX file.', true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) { toast('File is too large (max 10 MB).', true); return; }

  state.selectedFile = file;
  $('dz-primary').textContent = 'File selected';
  $('drop-zone').classList.add('has-file');
  $('file-pill-name').textContent = file.name;
  $('file-pill').classList.remove('hidden');
  $('btn-analyse').disabled = false;
}

function clearFile() {
  state.selectedFile = null;
  $('file-input').value = '';
  $('dz-primary').textContent = 'Drag & drop your file here';
  $('drop-zone').classList.remove('has-file');
  $('file-pill').classList.add('hidden');
  $('btn-analyse').disabled = true;
}

/* ══════════════════════════════════════════════════════════
   ANALYSIS
   ══════════════════════════════════════════════════════════ */
function runAnalysis() {
  if (!state.selectedFile) { toast('Please select a file first.', true); return; }

  setLoading(true);
  animateLoadingSteps();

  const fd = new FormData();
  fd.append('resume', state.selectedFile, state.selectedFile.name);
  fd.append('country', state.country);

  const ctrl = new AbortController();
  state.analysisController = ctrl;
  const tId = setTimeout(() => ctrl.abort(), 120_000);

  fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: fd, signal: ctrl.signal })
    .then(r => {
      clearTimeout(tId);
      if (!r.ok) throw new Error(r.status >= 500 ? 'Server error — please try again later.' : 'Unexpected error — please try again.');
      return r.json();
    })
    .then(data => {
      state.lastResponse = data;
      setLoading(false);
      updateMatchBadge(data.matches?.length ?? 0);
      if (data.health_check?.issues?.length > 0) {
        showResumeCheck(data);
      } else {
        switchView('matches');
      }
    })
    .catch(err => {
      clearTimeout(tId);
      setLoading(false);
      toast(err.name === 'AbortError' ? 'Request timed out.' : (err.message || 'Network error.'), true);
    });
}

function setLoading(on) {
  $('analyse-loading').classList.toggle('hidden', !on);
  $('btn-analyse').disabled = on || !state.selectedFile;
  $('drop-zone').style.pointerEvents = on ? 'none' : '';
  $('drop-zone').style.opacity = on ? '.5' : '';
  // Reset loading bar animation
  if (on) {
    const bar = $('loading-bar');
    bar.style.animation = 'none';
    void bar.offsetWidth;
    bar.style.animation = '';
  }
}

let stepTimer = null;
function animateLoadingSteps() {
  clearTimeout(stepTimer);
  const steps = $$('.loading-step');
  steps.forEach((s, i) => { s.classList.toggle('active', i === 0); s.classList.remove('done'); });
  let cur = 0;
  function next() {
    if (cur < steps.length - 1) {
      steps[cur].classList.remove('active'); steps[cur].classList.add('done');
      cur++;
      steps[cur].classList.add('active');
      stepTimer = setTimeout(next, 28000);
    }
  }
  stepTimer = setTimeout(next, 28000);
}

function updateMatchBadge(count) {
  const badge = $('nav-badge');
  if (count > 0) {
    badge.textContent = count;
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

/* ══════════════════════════════════════════════════════════
   RESUME CHECK OVERLAY
   ══════════════════════════════════════════════════════════ */
function showResumeCheck(data) {
  const hc = data.health_check;
  const issuesEl = $('rc-issues');
  issuesEl.innerHTML = '';

  (hc.issues || []).forEach((txt, i) => {
    const el = document.createElement('div');
    el.className = 'rc-issue';
    el.style.animationDelay = `${i * 0.06}s`;
    el.innerHTML = `
      <div class="rc-issue-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      </div>
      <p class="rc-issue-text">${esc(txt)}</p>`;
    issuesEl.appendChild(el);
  });

  const skillsWrap = $('rc-skills-section');
  const skillsEl = $('rc-skills');
  skillsEl.innerHTML = '';
  const skills = hc.suggested_skills || [];
  if (skills.length) {
    skillsWrap.classList.remove('hidden');
    skills.forEach((s, i) => {
      const c = document.createElement('span');
      c.className = 'chip'; c.style.animationDelay = `${i * 0.04}s`;
      c.textContent = s; skillsEl.appendChild(c);
    });
  } else {
    skillsWrap.classList.add('hidden');
  }

  openOverlay('overlay-resume-check');
}

/* ══════════════════════════════════════════════════════════
   MATCHES
   ══════════════════════════════════════════════════════════ */
function renderMatches() {
  const data = state.lastResponse;
  const grid = $('matches-grid');
  const empty = $('matches-empty');
  const sub = $('matches-subtitle');
  const sw = $('matches-sort-wrap');

  if (!data?.matches?.length) {
    grid.classList.add('hidden');
    empty.classList.remove('hidden');
    sub.textContent = 'Upload your resume to see live job matches.';
    sw.classList.add('hidden');
    return;
  }

  empty.classList.add('hidden');
  grid.classList.remove('hidden');
  sw.classList.remove('hidden');

  const matches = [...data.matches];
  const sort = $('matches-sort').value;
  if (sort === 'title') {
    matches.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
  } else {
    matches.sort((a, b) => scoreNum(b) - scoreNum(a));
  }

  sub.textContent = `${matches.length} live role${matches.length === 1 ? '' : 's'} matched to your resume`;

  grid.innerHTML = '';
  matches.forEach((m, i) => {
    const card = buildMatchCard(m, i);
    grid.appendChild(card);
  });
}

function scoreNum(m) {
  const raw = m.match_score ?? 0;
  return raw <= 1 ? raw * 100 : raw;
}

function buildMatchCard(match, idx) {
  const pct = Math.round(scoreNum(match));
  const high = pct >= 60;

  const card = document.createElement('div');
  card.className = 'match-card';
  card.style.animationDelay = `${idx * 0.045}s`;
  card.setAttribute('tabindex', '0');
  card.setAttribute('role', 'button');
  card.setAttribute('aria-label', `View details for ${match.title || 'job'}`);

  const company = esc(match.company || '');
  const location = esc(match.location || '');
  const subline = [company, location].filter(Boolean).join(' · ');

  card.innerHTML = `
    <div class="mc-header">
      <p class="mc-title">${esc(match.title || 'Unknown role')}</p>
      <span class="mc-badge ${high ? 'high' : 'low'}">${pct}%</span>
    </div>
    ${subline ? `<p class="mc-company">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
      ${subline}
    </p>` : ''}
    <div class="mc-footer">
      <span class="mc-link">
        View details
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
      </span>
      <svg class="mc-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
    </div>`;

  const open = () => showMatchDetail(match);
  card.addEventListener('click', open);
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  return card;
}

/* ══════════════════════════════════════════════════════════
   MATCH DETAIL MODAL
   ══════════════════════════════════════════════════════════ */
function showMatchDetail(match) {
  const pct = Math.round(scoreNum(match));

  $('modal-match-title').textContent = match.title || 'Match detail';
  $('modal-score').textContent = `${pct}%`;
  $('modal-company').textContent = match.company || '';
  $('modal-location').textContent = match.location || '';
  $('modal-comment').textContent = match.comment || 'Match based on your resume and the role requirements.';

  const listBtn = $('btn-view-listing');
  if (match.url) { listBtn.href = match.url; listBtn.style.display = ''; }
  else { listBtn.style.display = 'none'; }

  // Animate score ring  circumference = 2πr = 2π*50 ≈ 314
  const ringFill = $('score-ring-fill');
  const offset = 314 - (pct / 100) * 314;
  // Reset first
  ringFill.style.transition = 'none';
  ringFill.style.strokeDashoffset = '314';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ringFill.style.transition = 'stroke-dashoffset .8s cubic-bezier(.4,0,.2,1)';
      ringFill.style.strokeDashoffset = offset;
    });
  });

  // ── Semantic breakdown bars ─────────────────────────────
  const bd = match.breakdown;
  const dims = [
    { fillId: 'bd-skills-fill', pctId: 'bd-skills-pct', value: bd?.skills_score },
    { fillId: 'bd-exp-fill',    pctId: 'bd-exp-pct',    value: bd?.experience_score },
    { fillId: 'bd-title-fill',  pctId: 'bd-title-pct',  value: bd?.title_score },
  ];

  // Reset bars first (so re-opening animates cleanly)
  dims.forEach(({ fillId, pctId }) => {
    const fill = $(fillId);
    fill.style.transition = 'none';
    fill.style.width = '0%';
    $(pctId).textContent = '—';
  });

  // Stagger-animate bars after modal opens
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      dims.forEach(({ fillId, pctId, value }, i) => {
        setTimeout(() => {
          const fill = $(fillId);
          const v = value ?? 0;
          fill.style.transition = `width .7s cubic-bezier(.4,0,.2,1) ${i * 0.1}s`;
          fill.style.width = `${Math.min(100, v)}%`;
          $(pctId).textContent = bd ? `${Math.round(v)}%` : '—';
        }, i * 80);
      });
    });
  });

  // Matched skills chips
  const matchedWrap = $('modal-matched-skills-wrap');
  const matchedEl   = $('modal-matched-skills');
  matchedEl.innerHTML = '';
  const matched = bd?.matched_skills ?? [];
  if (matched.length) {
    matchedWrap.classList.remove('hidden');
    matched.forEach((s, i) => {
      const c = document.createElement('span');
      c.className = 'chip chip-match';
      c.style.animationDelay = `${i * 0.05}s`;
      c.textContent = s;
      matchedEl.appendChild(c);
    });
  } else {
    matchedWrap.classList.add('hidden');
  }

  openModal('modal-match');
}

/* ══════════════════════════════════════════════════════════
   RESUME VIEW
   ══════════════════════════════════════════════════════════ */
function renderResume() {
  const data = state.lastResponse;
  if (!data?.resume_summary) {
    $('resume-empty').classList.remove('hidden');
    $('resume-content').classList.add('hidden');
    return;
  }

  const s = data.resume_summary;
  $('resume-empty').classList.add('hidden');
  $('resume-content').classList.remove('hidden');

  const name = s.name || 'Unknown';
  $('profile-name').textContent = name;
  $('profile-email').textContent = s.email || '';
  $('profile-avatar').textContent = initials(name);

  const skillsEl = $('resume-skills');
  skillsEl.innerHTML = '';
  (s.skills || []).forEach((sk, i) => {
    const c = document.createElement('span');
    c.className = 'chip'; c.style.animationDelay = `${i * 0.04}s`;
    c.textContent = sk; skillsEl.appendChild(c);
  });

  $('resume-experience').textContent = s.experience || 'No experience information detected.';

  // ── Inline resume check ──────────────────────────────────
  const hc = data.health_check;
  const rcInline = $('resume-check-inline');
  if (hc?.issues?.length > 0) {
    rcInline.classList.remove('hidden');

    const issuesEl = $('rc-inline-issues');
    issuesEl.innerHTML = '';
    (hc.issues || []).forEach((txt, i) => {
      const el = document.createElement('div');
      el.className = 'rc-inline-issue';
      el.style.animationDelay = `${i * 0.06}s`;
      el.innerHTML = `
        <div class="rc-issue-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <p class="rc-issue-text">${esc(txt)}</p>`;
      issuesEl.appendChild(el);
    });

    const skillsSection = $('rc-inline-skills-section');
    const inlineSkillsEl = $('rc-inline-skills');
    inlineSkillsEl.innerHTML = '';
    const suggestedSkills = hc.suggested_skills || [];
    if (suggestedSkills.length) {
      skillsSection.classList.remove('hidden');
      suggestedSkills.forEach((sk, i) => {
        const c = document.createElement('span');
        c.className = 'chip'; c.style.animationDelay = `${i * 0.04}s`;
        c.textContent = sk; inlineSkillsEl.appendChild(c);
      });
    } else {
      skillsSection.classList.add('hidden');
    }
  } else {
    rcInline.classList.add('hidden');
  }
}

/* ══════════════════════════════════════════════════════════
   SETTINGS
   ══════════════════════════════════════════════════════════ */
function applyDark(on, animated) {
  if (!animated) document.documentElement.style.transition = 'none';
  document.body.classList.toggle('dark', on);
  if (!animated) requestAnimationFrame(() => document.documentElement.style.removeProperty('transition'));
}

function openCountryModal() {
  $('country-input').value = state.country;
  openModal('modal-country');
  setTimeout(() => $('country-input').focus(), 50);
}

function saveCountry() {
  const v = $('country-input').value.trim();
  if (v) {
    state.country = v;
    localStorage.setItem('ra_country', v);
    $('country-display').textContent = v;
    toast('Country saved.');
  }
  closeModal('modal-country');
}

/* ══════════════════════════════════════════════════════════
   OVERLAY & MODAL HELPERS
   ══════════════════════════════════════════════════════════ */
function openOverlay(id) {
  const el = $(id); el.classList.remove('hidden');
}
function closeOverlay(id, cb) {
  const el = $(id);
  const panel = el.querySelector('.overlay-panel');
  if (panel) {
    panel.style.transition = 'transform .28s cubic-bezier(.4,0,1,1)';
    panel.style.transform = 'translateX(100%)';
  }
  setTimeout(() => {
    el.classList.add('hidden');
    if (panel) { panel.style.transition = ''; panel.style.transform = ''; }
    if (cb) cb();
  }, 290);
}

function openModal(id) { $(id).classList.remove('hidden'); }
function closeModal(id) { $(id).classList.add('hidden'); }

/* ══════════════════════════════════════════════════════════
   TOASTS
   ══════════════════════════════════════════════════════════ */
function toast(msg, isError = false) {
  const t = document.createElement('div');
  t.className = `toast${isError ? ' error' : ''}`;
  t.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      ${isError
      ? '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
      : '<polyline points="20 6 9 17 4 12"/>'}
    </svg>
    <span>${esc(msg)}</span>`;
  $('toast-stack').appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 280); }, 3500);
}

/* ══════════════════════════════════════════════════════════
   UTILS
   ══════════════════════════════════════════════════════════ */
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function initials(name) {
  if (!name) return '?';
  const parts = name.trim().split(/\s+/);
  return parts.slice(0, 2).map(p => p[0]?.toUpperCase() || '').join('') || '?';
}

/* ── Boot ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);
