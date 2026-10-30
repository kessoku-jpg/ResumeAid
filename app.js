// Main UI logic and API communication for ResumeAid
'use strict';

// Backend URL: override with window.RESUMEAID_API_BASE (set before app.js loads)
// or localStorage 'ra_api'. Falls back to the local dev server.
const API_BASE = (window.RESUMEAID_API_BASE || localStorage.getItem('ra_api') || 'http://localhost:5000').replace(/\/+$/, '');

// Country codes understood by the backend (PH sources + Adzuna-supported markets)
const COUNTRIES = {
  ph: 'Philippines', us: 'United States', gb: 'United Kingdom', au: 'Australia',
  ca: 'Canada', sg: 'Singapore', nz: 'New Zealand', in: 'India', de: 'Germany',
  fr: 'France', nl: 'Netherlands', it: 'Italy', pl: 'Poland', at: 'Austria',
  za: 'South Africa', br: 'Brazil',
};

function loadCountry() {
  const saved = (localStorage.getItem('ra_country') || '').toLowerCase();
  return COUNTRIES[saved] ? saved : 'ph';
}

// Default match filters — "Best match" ordering with nothing filtered out
function defaultFilters() {
  return {
    q: '',
    sort: 'score',
    mode: 'any',          // any | remote | hybrid | onsite
    types: new Set(),     // employment_type values
    sources: new Set(),   // source values
    salaryMin: 0,         // approximate monthly PHP
    salaryOnly: false,
    minScore: 0,
    postedDays: 0,
  };
}

// Global application state
const state = {
  lastResponse: null,
  selectedFile: null,
  country: loadCountry(),
  darkMode: localStorage.getItem('ra_dark') === 'true',
  analysisController: null,
  filters: defaultFilters(),
  // Filter panel starts open on desktop, collapsed on phones
  filtersOpen: window.matchMedia('(min-width: 769px)').matches,
};

// DOM element selectors
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// Initialize app and apply user preferences
function init() {
  // Dark mode: apply immediately, no flash
  applyDark(state.darkMode, false);
  $('dark-mode-toggle').checked = state.darkMode;
  $('country-display').textContent = COUNTRIES[state.country];
  populateCountrySelect();

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
  checkBackend();
}

// Warn early if the API isn't reachable, instead of failing after upload
function checkBackend() {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 5000);
  fetch(`${API_BASE}/api/health`, { signal: ctrl.signal })
    .then(r => { if (!r.ok) throw new Error(); })
    .catch(() => setTimeout(() => toast('Can’t reach the ResumeAid server. Is the backend running?', true), 2600))
    .finally(() => clearTimeout(t));
}

function populateCountrySelect() {
  const sel = $('country-input');
  sel.innerHTML = '';
  Object.entries(COUNTRIES).forEach(([code, name]) => {
    const opt = document.createElement('option');
    opt.value = code; opt.textContent = name;
    sel.appendChild(opt);
  });
}

// Attach DOM event listeners
function bindAll() {

  // Desktop sidebar navigation
  $$('.nav-item').forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.view)));

  // Mobile top navigation
  $$('.mnav-btn').forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.view)));

  // File upload drag-and-drop zone
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

  // Clear selected file
  $('file-pill-remove').addEventListener('click', e => { e.stopPropagation(); clearFile(); });

  // Start resume analysis
  $('btn-analyse').addEventListener('click', () => runAnalysis(false));
  $('btn-refresh-listings').addEventListener('click', () => runAnalysis(true));

  // Resume health check overlay
  $('rc-close').addEventListener('click', () => closeOverlay('overlay-resume-check'));
  $('rc-backdrop').addEventListener('click', () => closeOverlay('overlay-resume-check'));
  $('btn-continue').addEventListener('click', () => closeOverlay('overlay-resume-check', () => switchView('matches')));

  // Job match detail modal
  $('modal-match-close').addEventListener('click', () => closeModal('modal-match'));
  $('modal-match').addEventListener('click', e => { if (e.target === $('modal-match')) closeModal('modal-match'); });

  // Match filters and sorting
  bindFilters();

  // Settings panel toggles
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

  // Country selection modal
  $('country-cancel').addEventListener('click', () => closeModal('modal-country'));
  $('modal-country-close').addEventListener('click', () => closeModal('modal-country'));
  $('modal-country').addEventListener('click', e => { if (e.target === $('modal-country')) closeModal('modal-country'); });
  $('country-save').addEventListener('click', saveCountry);
  $('country-input').addEventListener('keydown', e => { if (e.key === 'Enter') saveCountry(); });

  // About info modal
  $('about-ok').addEventListener('click', () => closeModal('modal-about'));
  $('modal-about').addEventListener('click', e => { if (e.target === $('modal-about')) closeModal('modal-about'); });

  // Escape closes whichever dialog is open
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    const openModalEl = [...$$('.modal-backdrop')].find(m => !m.classList.contains('hidden'));
    if (openModalEl) { closeModal(openModalEl.id); return; }
    if (!$('overlay-resume-check').classList.contains('hidden')) closeOverlay('overlay-resume-check');
  });
}

// Handle navigating between UI sections
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

// Validate and handle selected files
function onFile(file) {
  // Backend only supports PDF and DOCX (legacy .doc can't be parsed)
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['pdf', 'docx'].includes(ext)) {
    toast(ext === 'doc'
      ? 'Legacy .doc files aren’t supported — please save as PDF or DOCX.'
      : 'Please select a PDF or DOCX file.', true);
    return;
  }
  if (file.size === 0) { toast('That file is empty.', true); return; }
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

// API communication and loading states
function runAnalysis(forceRefresh = false) {
  if (!state.selectedFile) { toast('Please select a file first.', true); return; }
  if (state.analysisController) return; // already running

  setLoading(true);
  animateLoadingSteps();

  const fd = new FormData();
  fd.append('resume', state.selectedFile, state.selectedFile.name);
  fd.append('country', state.country);
  if (forceRefresh) {
    fd.append('refresh', '1');
  }

  const ctrl = new AbortController();
  state.analysisController = ctrl;
  const tId = setTimeout(() => ctrl.abort(), 180_000);

  fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: fd, signal: ctrl.signal })
    .then(async r => {
      // Surface the server's own error message when it provides one
      const body = await r.json().catch(() => null);
      if (!r.ok) {
        const fallback = r.status >= 500 ? 'Server error — please try again later.' : 'Unexpected error — please try again.';
        throw new Error(body?.error || fallback);
      }
      if (!body) throw new Error('Invalid response from server.');
      return body;
    })
    .then(data => {
      state.lastResponse = data;
      // New results → start from "Best match" with no filters applied
      state.filters = defaultFilters();
      syncFilterControls();
      updateMatchBadge(data.matches?.length ?? 0);
      if (data.search_info?.is_placeholder) {
        toast('No live listings found right now — showing sample roles instead.', true);
      }
      if (data.health_check?.issues?.length > 0) {
        showResumeCheck(data);
      } else {
        switchView('matches');
      }
    })
    .catch(err => {
      let msg = err.message || 'Network error.';
      if (err.name === 'AbortError') msg = 'Request timed out — the server took too long.';
      else if (err instanceof TypeError) msg = 'Can’t reach the server. Check that the backend is running.';
      toast(msg, true);
    })
    .finally(() => {
      clearTimeout(tId);
      state.analysisController = null;
      setLoading(false);
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
  } else {
    clearTimeout(stepTimer);
  }
}

let stepTimer = null;
function animateLoadingSteps() {
  clearTimeout(stepTimer);
  const steps = $$('.loading-step');
  steps.forEach((s, i) => { s.classList.toggle('active', i === 0); s.classList.remove('done'); });
  let cur = 0;
  // Parsing is quick; scraping is the slow part, so the step timings are uneven
  const delays = [6000, 25000];
  function next() {
    if (cur < steps.length - 1) {
      steps[cur].classList.remove('active'); steps[cur].classList.add('done');
      cur++;
      steps[cur].classList.add('active');
      if (cur < delays.length) stepTimer = setTimeout(next, delays[cur]);
    }
  }
  stepTimer = setTimeout(next, delays[0]);
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

// Display resume issues and skill suggestions
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

// Render, filter and sort job match cards
function renderMatches() {
  const data = state.lastResponse;
  const grid = $('matches-grid');
  const empty = $('matches-empty');
  const sub = $('matches-subtitle');
  const sw = $('matches-sort-wrap');
  const filtersEl = $('match-filters');
  const filterEmpty = $('matches-filter-empty');

  if (!data?.matches?.length) {
    grid.classList.add('hidden');
    empty.classList.remove('hidden');
    filtersEl.classList.add('hidden');
    filterEmpty.classList.add('hidden');
    $('matches-notice').classList.add('hidden');
    sub.textContent = data
      ? 'No matching roles found. Try a different country in Settings.'
      : 'Upload your resume to see live job matches.';
    sw.classList.add('hidden');
    return;
  }

  empty.classList.add('hidden');
  sw.classList.remove('hidden');
  filtersEl.classList.remove('hidden');

  const all = data.matches;
  renderFilterOptions(all);
  const visible = sortMatches(all.filter(passesFilters), state.filters.sort);

  const info = data.search_info || {};
  const placeholder = !!info.is_placeholder;
  const noun = placeholder ? 'sample role' : 'live role';
  const query = info.query ? ` for “${info.query}”` : '';
  sub.textContent = `${all.length} ${noun}${all.length === 1 ? '' : 's'} matched${query}`;

  const active = activeFilterCount();
  $('mf-result-count').textContent = active || state.filters.q
    ? `Showing ${visible.length} of ${all.length}`
    : `Showing all ${all.length}`;
  $('mf-count').textContent = active;
  $('mf-count').classList.toggle('hidden', active === 0);
  $('mf-reset').disabled = active === 0 && !state.filters.q;

  const isCached = !!info.from_cache;
  const cacheBadge = $('cache-badge');
  const refreshBtn = $('btn-refresh-listings');
  if (cacheBadge) cacheBadge.classList.toggle('hidden', !isCached || placeholder);
  if (refreshBtn) refreshBtn.classList.toggle('hidden', placeholder);

  const notice = $('matches-notice');
  notice.classList.toggle('hidden', !placeholder);
  if (placeholder) {
    notice.textContent = 'We couldn’t fetch live listings right now, so these are example roles to show how matching works. Try again later.';
  }

  grid.classList.toggle('hidden', visible.length === 0);
  filterEmpty.classList.toggle('hidden', visible.length > 0);
  grid.innerHTML = '';
  visible.forEach((m, i) => grid.appendChild(buildMatchCard(m, i)));
}

// ── Match filters ────────────────────────────────────────────
const MODE_LABELS = { remote: 'Remote', hybrid: 'Hybrid', onsite: 'On-site' };
const TYPE_LABELS = {
  full_time: 'Full-time', part_time: 'Part-time', contract: 'Contract',
  freelance: 'Freelance', internship: 'Internship',
};
const SOURCE_LABELS = {
  philjobnet: 'PhilJobNet', onlinejobs: 'OnlineJobs.ph', adzuna: 'Adzuna', sample: 'Sample',
};

function bindFilters() {
  const f = () => state.filters;

  $('mf-search').addEventListener('input', e => { f().q = e.target.value.trim().toLowerCase(); renderMatches(); });
  $('matches-sort').addEventListener('change', e => { f().sort = e.target.value; renderMatches(); });

  $$('#mf-mode .seg-btn').forEach(btn => btn.addEventListener('click', () => {
    f().mode = btn.dataset.value;
    syncFilterControls();
    renderMatches();
  }));

  // Chip groups are re-rendered with the results, so delegate clicks
  const chipToggle = (containerId, key) => $(containerId).addEventListener('click', e => {
    const chip = e.target.closest('.mf-chip');
    if (!chip || chip.disabled) return;
    const set = f()[key];
    set.has(chip.dataset.value) ? set.delete(chip.dataset.value) : set.add(chip.dataset.value);
    renderMatches();
  });
  chipToggle('mf-type', 'types');
  chipToggle('mf-source', 'sources');

  $('mf-salary-min').addEventListener('change', e => { f().salaryMin = Number(e.target.value) || 0; renderMatches(); });
  $('mf-salary-only').addEventListener('change', e => { f().salaryOnly = e.target.checked; renderMatches(); });
  $('mf-score').addEventListener('input', e => {
    f().minScore = Number(e.target.value) || 0;
    $('mf-score-val').textContent = `${f().minScore}%`;
    renderMatches();
  });
  $('mf-posted').addEventListener('change', e => { f().postedDays = Number(e.target.value) || 0; renderMatches(); });

  $('mf-toggle').addEventListener('click', () => {
    state.filtersOpen = !state.filtersOpen;
    syncFilterControls();
  });

  const reset = () => {
    const sort = state.filters.sort; // clearing filters keeps the chosen order
    state.filters = defaultFilters();
    state.filters.sort = sort;
    syncFilterControls();
    renderMatches();
  };
  $('mf-reset').addEventListener('click', reset);
  $('mf-reset-empty').addEventListener('click', reset);

  syncFilterControls();
}

// Push state.filters into the form controls
function syncFilterControls() {
  const f = state.filters;
  $('mf-search').value = f.q;
  $('matches-sort').value = f.sort;
  $$('#mf-mode .seg-btn').forEach(b => b.setAttribute('aria-checked', String(b.dataset.value === f.mode)));
  $('mf-salary-min').value = String(f.salaryMin);
  $('mf-salary-only').checked = f.salaryOnly;
  $('mf-score').value = String(f.minScore);
  $('mf-score-val').textContent = `${f.minScore}%`;
  $('mf-posted').value = String(f.postedDays);
  $('mf-panel').classList.toggle('hidden', !state.filtersOpen);
  $('mf-toggle').setAttribute('aria-expanded', String(state.filtersOpen));
}

// Only offer job-type / board options that exist in these results, with counts
function renderFilterOptions(matches) {
  const count = key => matches.reduce((acc, m) => {
    if (m[key]) acc[m[key]] = (acc[m[key]] || 0) + 1;
    return acc;
  }, {});

  const renderChips = (containerId, labels, counts, selected, noneText) => {
    const el = $(containerId);
    const keys = Object.keys(labels).filter(k => counts[k]);
    // Keep selected chips visible even if they'd otherwise disappear
    selected.forEach(k => { if (!keys.includes(k)) keys.push(k); });
    if (!keys.length) { el.innerHTML = `<span class="mf-none">${esc(noneText)}</span>`; return; }
    el.innerHTML = keys.map(k => `
      <button type="button" class="mf-chip" data-value="${esc(k)}" aria-pressed="${selected.has(k)}">
        ${esc(labels[k] || k)}<span class="mf-chip-n">${counts[k] || 0}</span>
      </button>`).join('');
  };

  renderChips('mf-type', TYPE_LABELS, count('employment_type'), state.filters.types, 'Not listed for these roles');
  renderChips('mf-source', SOURCE_LABELS, count('source'), state.filters.sources, 'Unknown');

  // Show how many roles fall under each work setup
  const modes = count('work_mode');
  $$('#mf-mode .seg-btn').forEach(b => {
    const v = b.dataset.value;
    const n = v === 'any' ? matches.length : (modes[v] || 0);
    b.textContent = v === 'any' ? 'Any' : MODE_LABELS[v];
    b.title = `${n} role${n === 1 ? '' : 's'}`;
    b.classList.toggle('is-empty', v !== 'any' && n === 0);
  });
}

function activeFilterCount() {
  const f = state.filters;
  return (f.mode !== 'any') + f.types.size + f.sources.size + (f.salaryMin > 0)
    + f.salaryOnly + (f.minScore > 0) + (f.postedDays > 0);
}

function passesFilters(m) {
  const f = state.filters;
  if (f.q) {
    const hay = `${m.title || ''} ${m.company || ''} ${m.location || ''}`.toLowerCase();
    if (!hay.includes(f.q)) return false;
  }
  if (f.mode !== 'any' && m.work_mode !== f.mode) return false;
  if (f.types.size && !f.types.has(m.employment_type)) return false;
  if (f.sources.size && !f.sources.has(m.source)) return false;
  if ((f.salaryOnly || f.salaryMin > 0) && !m.salary) return false;
  if (f.salaryMin > 0 && salaryHigh(m) < f.salaryMin) return false;
  if (f.minScore > 0 && scoreNum(m) < f.minScore) return false;
  if (f.postedDays > 0) {
    const age = daysSince(m.posted_at);
    if (age === null || age > f.postedDays) return false;
  }
  return true;
}

// Approximate monthly PHP figures (computed by the backend) for comparison
function salaryHigh(m) { return m.salary?.monthly_php_max ?? m.salary?.monthly_php_min ?? -1; }
function salaryLow(m) { return m.salary?.monthly_php_min ?? m.salary?.monthly_php_max ?? Infinity; }

function sortMatches(list, sort) {
  const byScore = (a, b) => scoreNum(b) - scoreNum(a);
  // Items missing the sort field always go last, then fall back to match score
  const missingLast = (hasA, hasB) => (hasA === hasB ? 0 : hasA ? -1 : 1);
  const text = v => (v && v !== 'Unknown' ? v : '');
  const cmp = {
    score: byScore,
    salary_desc: (a, b) => missingLast(!!a.salary, !!b.salary) || salaryHigh(b) - salaryHigh(a) || byScore(a, b),
    salary_asc: (a, b) => missingLast(!!a.salary, !!b.salary) || salaryLow(a) - salaryLow(b) || byScore(a, b),
    newest: (a, b) => missingLast(!!a.posted_at, !!b.posted_at)
      || (b.posted_at || '').localeCompare(a.posted_at || '') || byScore(a, b),
    title: (a, b) => (a.title || '').localeCompare(b.title || ''),
    company: (a, b) => missingLast(!!text(a.company), !!text(b.company))
      || text(a.company).localeCompare(text(b.company)) || byScore(a, b),
  }[sort] || byScore;
  return [...list].sort(cmp);
}

// Days between a YYYY-MM-DD date and today (local time), or null
function daysSince(isoDate) {
  if (!isoDate) return null;
  const [y, mo, d] = isoDate.split('-').map(Number);
  if (!y || !mo || !d) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  return Math.max(0, Math.round((today - new Date(y, mo - 1, d)) / 86_400_000));
}

function postedLabel(isoDate) {
  const n = daysSince(isoDate);
  if (n === null) return '';
  if (n === 0) return 'Today';
  if (n === 1) return 'Yesterday';
  if (n < 30) return `${n} days ago`;
  return new Date(isoDate).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

const CURRENCY_SYMBOLS = {
  PHP: '₱', USD: '$', GBP: '£', EUR: '€', AUD: 'A$', CAD: 'C$', SGD: 'S$', NZD: 'NZ$', INR: '₹',
};
const PERIOD_LABELS = { hour: '/hr', day: '/day', week: '/wk', month: '/mo', year: '/yr' };

function fmtAmount(n) {
  if (n >= 100_000) return `${Math.round(n / 1000).toLocaleString()}k`;
  return n.toLocaleString(undefined, { maximumFractionDigits: n < 100 ? 2 : 0 });
}

// "₱18,000/mo", "$5–7/hr", "Est. £40k–50k/yr"
function formatSalary(s) {
  if (!s) return '';
  const sym = CURRENCY_SYMBOLS[s.currency] ?? `${s.currency} `;
  const range = s.min === s.max ? fmtAmount(s.min) : `${fmtAmount(s.min)}–${fmtAmount(s.max)}`;
  return `${s.is_estimate ? 'Est. ' : ''}${sym}${range}${PERIOD_LABELS[s.period] || ''}`;
}

// "≈ ₱25,000/mo" for anything not already a monthly peso figure
function formatSalaryPhp(s) {
  if (!s || (s.currency === 'PHP' && s.period === 'month')) return '';
  const lo = s.monthly_php_min, hi = s.monthly_php_max;
  if (lo == null && hi == null) return '';
  const range = lo === hi || hi == null || lo == null
    ? `₱${(lo ?? hi).toLocaleString()}`
    : `₱${lo.toLocaleString()}–${hi.toLocaleString()}`;
  return `≈ ${range}/mo`;
}

// Backend always returns match_score as a 0–100 percentage
function scoreNum(m) {
  const n = Number(m.match_score);
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
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

  // Only show attributes we actually know — never "Unknown" tags
  const tags = [];
  if (match.salary) {
    const php = formatSalaryPhp(match.salary);
    tags.push(`<span class="mc-tag is-salary"${php ? ` title="${esc(php)}"` : ''}>${esc(formatSalary(match.salary))}</span>`);
  }
  if (match.work_mode) tags.push(`<span class="mc-tag${match.work_mode === 'remote' ? ' is-remote' : ''}">${esc(MODE_LABELS[match.work_mode] || match.work_mode)}</span>`);
  if (match.employment_type) tags.push(`<span class="mc-tag">${esc(TYPE_LABELS[match.employment_type] || match.employment_type)}</span>`);
  const posted = postedLabel(match.posted_at);
  if (posted) tags.push(`<span class="mc-tag is-quiet">${esc(posted)}</span>`);
  const source = SOURCE_LABELS[match.source] || '';

  card.innerHTML = `
    <div class="mc-header">
      <p class="mc-title">${esc(match.title || 'Unknown role')}</p>
      <span class="mc-badge ${high ? 'high' : 'low'}">${pct}%</span>
    </div>
    ${subline ? `<p class="mc-company">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
      ${subline}
    </p>` : ''}
    ${tags.length ? `<div class="mc-tags">${tags.join('')}</div>` : ''}
    <div class="mc-footer">
      <span class="mc-link">
        View details
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
      </span>
      ${source ? `<span class="mc-source">via ${esc(source)}</span>` : ''}
      <svg class="mc-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
    </div>`;

  const open = () => showMatchDetail(match);
  card.addEventListener('click', open);
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  return card;
}

// Display detailed job match breakdown and skills
function showMatchDetail(match) {
  const pct = Math.round(scoreNum(match));

  $('modal-match-title').textContent = match.title || 'Match detail';
  $('modal-score').textContent = `${pct}%`;
  $('modal-company').textContent = match.company || '';
  $('modal-location').textContent = match.location || '';
  $('modal-comment').textContent = match.comment || 'Match based on your resume and the role requirements.';

  // Job details list — rows only for attributes the listing provides
  const facts = [];
  if (match.salary) {
    const php = formatSalaryPhp(match.salary);
    const note = match.salary.is_estimate ? 'Estimated by Adzuna' : '';
    facts.push(['Salary', formatSalary(match.salary), [php, note].filter(Boolean).join(' · ')]);
  }
  if (match.work_mode) facts.push(['Work setup', MODE_LABELS[match.work_mode] || match.work_mode]);
  if (match.employment_type) facts.push(['Job type', TYPE_LABELS[match.employment_type] || match.employment_type]);
  if (match.posted_at) facts.push(['Posted', postedLabel(match.posted_at)]);
  if (SOURCE_LABELS[match.source]) facts.push(['Job board', SOURCE_LABELS[match.source]]);
  $('modal-facts').innerHTML = facts.map(([k, v, sub]) =>
    `<dt>${esc(k)}</dt><dd>${esc(v)}${sub ? `<span class="fact-sub">${esc(sub)}</span>` : ''}</dd>`).join('');
  $('modal-facts-wrap').classList.toggle('hidden', facts.length === 0);

  const listBtn = $('btn-view-listing');
  if (match.url && /^https?:\/\//i.test(match.url)) {
    listBtn.href = match.url;
    listBtn.target = '_blank';
    listBtn.rel = 'noopener noreferrer';
    listBtn.style.display = '';
  } else { listBtn.style.display = 'none'; }

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
    { fillId: 'bd-exp-fill', pctId: 'bd-exp-pct', value: bd?.experience_score },
    { fillId: 'bd-title-fill', pctId: 'bd-title-pct', value: bd?.title_score },
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
  const matchedEl = $('modal-matched-skills');
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

// Render parsed resume summary inline
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
  $('resume-education').textContent = s.education || 'No education information detected.';

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

// User preferences logic
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
  const v = $('country-input').value;
  if (COUNTRIES[v]) {
    state.country = v;
    localStorage.setItem('ra_country', v);
    $('country-display').textContent = COUNTRIES[v];
    toast('Country saved.');
  }
  closeModal('modal-country');
}

// Utilities for opening/closing modals and overlays
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

// Temporary success/error notifications
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

// General utility and formatting functions
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

// Bootstrap app on load
document.addEventListener('DOMContentLoaded', init);
