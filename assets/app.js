(function () {
  'use strict';

  const PAGES = [
    { key: 'plan', file: '00_Preparation_Plan.md', title: 'Preparation Plan', icon: '🗺️',
      desc: 'Study schedule, what interviewers look for, and how to present your projects.' },
    { key: 'concepts', file: '01_DS_Concepts_Notes.md', title: 'Concept Notes', icon: '📘',
      desc: 'Round 1 theory: statistics, ML, deep learning, NLP, SQL, Python, MLOps.' },
    { key: 'questions', file: '02_DS_Interview_Questions.md', title: 'Interview Questions', icon: '❓',
      desc: '200 questions with hidden answers. Test yourself and track what to revise.' },
    { key: 'cases', file: '03_DS_Case_Study_Questions.md', title: 'Case Studies', icon: '🧩',
      desc: 'Round 2 frameworks and cases with worked solutions to try under a timer.' },
  ];
  const PAGE_BY_KEY = Object.fromEntries(PAGES.map(p => [p.key, p]));

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const main = $('#main');

  /* ---------- Persistent state (per browser) ---------- */
  const STORE_KEY = 'dsprep-state-v1';
  let state = {};
  try { state = JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch (e) { state = {}; }
  function setMark(key, value) {
    if (value == null) delete state[key]; else state[key] = value;
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) { /* storage unavailable */ }
    refreshMarks();
  }

  /* ---------- Helpers ---------- */
  function slug(text) {
    return text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
  }
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function stripEmoji(s) {
    return s.replace(/[🟢🟡🔴]/gu, '').replace(/\s+/g, ' ').trim();
  }
  function flash(el) {
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }

  /* ---------- Render pages ---------- */
  const index = [];          // search index
  let currentPage = 'home';

  function renderAll() {
    const content = window.PREP_CONTENT;
    if (!content || !window.marked) {
      main.innerHTML = '<p class="loading">Could not load the content. Check your internet connection (the page loads a Markdown renderer from a CDN) and make sure <code>assets/content.js</code> exists — run <code>python3 build.py</code>.</p>';
      return false;
    }
    main.innerHTML = '';
    main.appendChild(renderHome());
    PAGES.forEach(p => main.appendChild(renderPage(p, content[p.file] || '')));
    const footer = document.createElement('footer');
    footer.className = 'footer page';
    footer.innerHTML = 'Your progress is saved in this browser only. Content lives in the Markdown files; edit them and run <code>python3 build.py</code> to update the site.';
    main.appendChild(footer);
    return true;
  }

  function renderHome() {
    const el = document.createElement('section');
    el.className = 'page home';
    el.id = 'page-home';
    el.innerHTML = `
      <div class="hero">
        <p class="eyebrow">Data Scientist Interview Prep</p>
        <h1>Learn the concepts. Crack the case.</h1>
        <p class="lead">Everything for a two-round data science interview: concept notes, 200 questions with answers to test yourself on, and case studies with worked solutions. Your progress is saved as you go.</p>
        <div class="hero-actions">
          <a class="btn primary" href="#/plan">Start with the plan</a>
          <button class="btn" data-action="random-q">🎲 Random question</button>
          <button class="btn" data-action="random-case">🎲 Random case</button>
        </div>
      </div>
      <div class="rounds">
        <div class="round"><h3>Round 1 · Concepts</h3><p>Statistics, probability, A/B testing, ML algorithms, metrics, deep learning, SQL and Python. Use the <a href="#/concepts">notes</a> then drill the <a href="#/questions">questions</a>.</p></div>
        <div class="round"><h3>Round 2 · Case studies</h3><p>Clarify, structure, pick metrics, design the model or experiment, recommend. Learn the <a href="#/cases/part-a--frameworks">frameworks</a> and practise the <a href="#/cases">cases</a>.</p></div>
      </div>
      <div class="cards">
        ${PAGES.map(p => `
          <a class="card" href="#/${p.key}">
            <span class="card-icon">${p.icon}</span>
            <h3>${p.title}</h3>
            <p>${p.desc}</p>
            <div class="prog" data-progress="${p.key}"></div>
          </a>`).join('')}
      </div>
      <div class="tips">
        <h2>How to study here</h2>
        <ol>
          <li><strong>Concept notes:</strong> read a section, explain it out loud in your own words, then press <em>Mark done</em>.</li>
          <li><strong>Questions:</strong> answer out loud <em>before</em> opening the answer. Mark it <em>I know this</em> or <em>Revise</em>, then filter by <em>Revise</em> before your interview.</li>
          <li><strong>Case studies:</strong> start the 25-minute timer, solve on paper, then reveal the solution and compare.</li>
          <li>Press <kbd>/</kbd> anywhere to search every page.</li>
        </ol>
      </div>`;
    return el;
  }

  function renderPage(p, md) {
    const sec = document.createElement('article');
    sec.className = 'page page-' + p.key;
    sec.id = 'page-' + p.key;
    sec.hidden = true;
    sec.innerHTML = window.marked.parse(md, { gfm: true });

    removeInlineToc(sec);
    assignHeadingIds(sec, p.key);

    $$('table', sec).forEach(t => {
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      t.before(wrap);
      wrap.appendChild(t);
    });
    $$('a[href^="http"]', sec).forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });
    if (window.hljs) $$('pre code[class*="language-"]', sec).forEach(c => window.hljs.highlightElement(c));

    if (p.key === 'questions') buildQuestions(sec);
    if (p.key === 'cases') buildCases(sec);
    if (p.key === 'concepts') buildSectionToggles(sec, p.key);

    buildIndex(p, sec);
    return sec;
  }

  // The Markdown files carry their own contents list; the sidebar replaces it on the site.
  function removeInlineToc(sec) {
    $$('h2', sec).forEach(h => {
      if (!/^(table of contents|contents|sections)$/i.test(h.textContent.trim())) return;
      let n = h.nextElementSibling;
      while (n && !/^(H1|H2|HR)$/.test(n.tagName)) { const next = n.nextElementSibling; n.remove(); n = next; }
      if (n && n.tagName === 'HR') n.remove();
      h.remove();
    });
  }

  function assignHeadingIds(sec, key) {
    const seen = {};
    $$('h1, h2, h3, h4', sec).forEach(h => {
      const title = h.textContent.trim();
      let id = slug(title);
      if (seen[id] != null) { seen[id] += 1; id += '-' + seen[id]; } else seen[id] = 0;
      h.id = key + '--' + id;
      h.dataset.anchor = id;
      h.dataset.title = title;
    });
  }

  /* Questions: "**Q12. ...**" paragraphs become flashcards with hidden answers. */
  function buildQuestions(sec) {
    const isQ = el => el.tagName === 'P' && el.firstChild && el.firstChild.nodeName === 'STRONG' &&
      /^Q\d+\./.test(el.firstChild.textContent);
    const qParas = $$('p', sec).filter(isQ);
    const qSet = new Set(qParas);

    qParas.forEach(p => {
      const strong = p.firstChild;
      const raw = strong.textContent.trim();
      const num = raw.match(/^Q(\d+)/)[1];
      const diff = raw.includes('🟢') ? 'easy' : raw.includes('🟡') ? 'medium' : raw.includes('🔴') ? 'hard' : 'none';
      const qtext = stripEmoji(raw.replace(/^Q\d+\.\s*/, ''));

      const card = document.createElement('details');
      card.className = 'qcard';
      card.id = 'questions--q' + num;
      card.dataset.anchor = 'q' + num;
      card.dataset.qid = 'q' + num;
      card.dataset.diff = diff;
      card.innerHTML = `
        <summary>
          <span class="qnum">Q${num}</span>
          <span class="qtext">${escapeHtml(qtext)}</span>
          <span class="qmeta">
            ${diff !== 'none' ? `<span class="diff ${diff}">${diff}</span>` : ''}
            <svg class="chev" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </span>
        </summary>`;
      const body = document.createElement('div');
      body.className = 'qanswer';

      p.before(card);
      const siblings = [];
      let n = p.nextElementSibling;
      while (n && !qSet.has(n) && !/^(H1|H2|H3|H4|HR)$/.test(n.tagName)) { siblings.push(n); n = n.nextElementSibling; }

      strong.remove();
      if (p.textContent.trim()) {
        while (p.firstChild && p.firstChild.nodeType === 3 && !p.firstChild.textContent.trim()) p.firstChild.remove();
        body.appendChild(p);
      } else {
        p.remove();
      }
      siblings.forEach(s => body.appendChild(s));

      const actions = document.createElement('div');
      actions.className = 'qactions';
      actions.innerHTML = `
        <button class="mark-btn known" data-mark="known">✓ I know this</button>
        <button class="mark-btn revise" data-mark="revise">↻ Revise later</button>`;
      body.appendChild(actions);
      card.appendChild(body);
    });

    const h1 = $('h1', sec);
    const toolbar = document.createElement('div');
    toolbar.className = 'toolbar';
    toolbar.innerHTML = `
      <div class="toolbar-row">
        <span class="toolbar-label">Show</span>
        <button class="chip on" data-filter-status="all">All</button>
        <button class="chip" data-filter-status="unseen">Not marked</button>
        <button class="chip" data-filter-status="revise">Revise</button>
        <button class="chip" data-filter-status="known">Known</button>
      </div>
      <div class="toolbar-row">
        <span class="toolbar-label">Level</span>
        <button class="chip on" data-filter-diff="all">Any</button>
        <button class="chip" data-filter-diff="easy">Easy</button>
        <button class="chip" data-filter-diff="medium">Medium</button>
        <button class="chip" data-filter-diff="hard">Hard</button>
      </div>
      <div class="toolbar-row">
        <button class="btn small" data-action="random-q">🎲 Random question</button>
        <button class="btn small ghost" data-action="toggle-answers">Open all answers</button>
        <span class="toolbar-stats" id="qStats"></span>
      </div>`;
    (h1 && h1.nextElementSibling ? h1.nextElementSibling : h1).after(toolbar);
  }

  /* Case studies: hide each worked solution until the reader asks for it. */
  function buildCases(sec) {
    $$('h3', sec).filter(h => /^Case\s/.test(h.textContent.trim())).forEach(h => {
      const card = document.createElement('section');
      card.className = 'case-card';
      card.dataset.caseId = h.dataset.anchor;
      h.before(card);
      card.appendChild(h);

      const solution = document.createElement('div');
      solution.className = 'case-solution';
      solution.hidden = true;
      let n = card.nextElementSibling;
      while (n && !/^(H1|H2|H3|HR)$/.test(n.tagName)) { const next = n.nextElementSibling; solution.appendChild(n); n = next; }
      if (n && n.tagName === 'HR') n.remove();

      const bar = document.createElement('div');
      bar.className = 'case-bar';
      bar.innerHTML = `
        <button class="btn small primary" data-action="toggle-solution" aria-expanded="false">Show solution</button>
        <button class="done-toggle" data-done="case:${h.dataset.anchor}">Mark practised</button>
        <span class="case-hint">Try it yourself first: start the timer and solve it on paper.</span>`;
      card.appendChild(bar);
      card.appendChild(solution);
    });
  }

  function buildSectionToggles(sec, key) {
    $$('h2', sec).forEach(h => {
      const btn = document.createElement('button');
      btn.className = 'done-toggle';
      btn.dataset.done = `sec:${key}:${h.dataset.anchor}`;
      btn.textContent = 'Mark done';
      h.appendChild(btn);
    });
  }

  /* ---------- Search ---------- */
  function buildIndex(p, sec) {
    $$('.qcard', sec).forEach(c => {
      const title = 'Q' + c.dataset.qid.slice(1) + '. ' + $('.qtext', c).textContent;
      addEntry(p, c.dataset.anchor, title, $('.qanswer', c).textContent);
    });
    $$('h2, h3', sec).forEach(h => {
      let text = '';
      const card = h.parentElement.classList.contains('case-card') ? h.parentElement : null;
      if (card) {
        text = $('.case-solution', card).textContent;
      } else {
        let n = h.nextElementSibling;
        while (n && !/^(H1|H2|H3)$/.test(n.tagName) && !n.classList.contains('case-card')) {
          if (!n.classList.contains('qcard') && !n.classList.contains('toolbar')) text += ' ' + n.textContent;
          n = n.nextElementSibling;
        }
      }
      addEntry(p, h.dataset.anchor, h.dataset.title, text);
    });
  }
  function addEntry(p, anchor, title, text) {
    const clean = text.replace(/\s+/g, ' ').trim();
    index.push({ page: p.key, anchor, title, titleL: title.toLowerCase(), text: clean, textL: clean.toLowerCase() });
  }

  const searchInput = $('#search');
  const searchResults = $('#searchResults');
  let activeResult = -1;

  function highlight(s, terms) {
    let out = escapeHtml(s);
    terms.forEach(t => {
      if (!t) return;
      const re = new RegExp('(' + escapeHtml(t).replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
      out = out.replace(re, '<mark>$1</mark>');
    });
    return out;
  }

  function runSearch() {
    const q = searchInput.value.trim().toLowerCase();
    activeResult = -1;
    if (q.length < 2) { searchResults.hidden = true; return; }
    const terms = q.split(/\s+/).filter(Boolean);
    const hits = index
      .filter(e => terms.every(t => e.titleL.includes(t) || e.textL.includes(t)))
      .map(e => ({ e, score: (e.titleL.includes(q) ? 10 : 0) + terms.filter(t => e.titleL.includes(t)).length * 2 + (e.textL.includes(q) ? 1 : 0) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 30);

    if (!hits.length) {
      searchResults.innerHTML = '<div class="sr-empty">No results. Try fewer or different words.</div>';
    } else {
      searchResults.innerHTML = hits.map(({ e }) => {
        const pos = e.textL.indexOf(terms[0]);
        let snippet = '';
        if (pos >= 0) {
          const start = Math.max(0, pos - 50);
          snippet = (start > 0 ? '…' : '') + e.text.slice(start, pos + 110) + '…';
        } else {
          snippet = e.text.slice(0, 120) + (e.text.length > 120 ? '…' : '');
        }
        return `<a href="#/${e.page}/${encodeURIComponent(e.anchor)}" data-open="1">
          <div class="sr-title"><span class="sr-page">${PAGE_BY_KEY[e.page].title}</span><span>${highlight(e.title, terms)}</span></div>
          <div class="sr-snippet">${highlight(snippet, terms)}</div></a>`;
      }).join('');
    }
    searchResults.hidden = false;
  }

  let searchTimer;
  searchInput.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(runSearch, 120); });
  searchInput.addEventListener('focus', () => { if (searchInput.value.trim().length >= 2) runSearch(); });
  searchInput.addEventListener('keydown', e => {
    const links = $$('a', searchResults);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!links.length) return;
      e.preventDefault();
      activeResult = (activeResult + (e.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
      links.forEach((l, i) => l.classList.toggle('active', i === activeResult));
      links[activeResult].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter') {
      const target = links[activeResult >= 0 ? activeResult : 0];
      if (target) { e.preventDefault(); target.click(); }
    } else if (e.key === 'Escape') {
      searchInput.value = '';
      searchResults.hidden = true;
      searchInput.blur();
    }
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.search')) searchResults.hidden = true;
  });
  searchResults.addEventListener('click', e => {
    if (e.target.closest('a')) { openOnNavigate = true; searchResults.hidden = true; searchInput.blur(); }
  });

  /* ---------- Navigation ---------- */
  let openOnNavigate = false;

  function buildNav() {
    const items = [{ key: 'home', title: 'Home', icon: '🏠' }].concat(PAGES);
    $('#nav').innerHTML = items.map(p => `
      <a class="nav-item" href="#/${p.key === 'home' ? '' : p.key}" data-nav="${p.key}">
        <div class="nav-row"><span>${p.icon}</span><span>${p.title}</span></div>
        <div data-nav-progress="${p.key}"></div>
      </a>`).join('');
  }

  function buildToc(pageKey) {
    const toc = $('#toc');
    if (pageKey === 'home') { toc.innerHTML = ''; return; }
    const sec = $('#page-' + pageKey);
    const heads = $$('h2', sec);
    toc.innerHTML = `<div class="toc"><div class="toc-title">On this page</div>${heads.map(h => `
      <a href="#/${pageKey}/${encodeURIComponent(h.dataset.anchor)}" data-toc="${h.id}">
        <span>${escapeHtml(h.dataset.title)}</span><span class="toc-check" data-toc-check="sec:${pageKey}:${h.dataset.anchor}"></span>
      </a>`).join('')}</div>`;
    refreshMarks();
  }

  function route() {
    const hash = location.hash.replace(/^#\/?/, '');
    const parts = hash.split('/');
    const pageKey = PAGE_BY_KEY[parts[0]] ? parts[0] : 'home';
    const anchor = decodeURIComponent(parts.slice(1).join('/'));

    if (pageKey !== currentPage) {
      $$('main > .page:not(.footer)').forEach(s => { s.hidden = s.id !== 'page-' + pageKey; });
      currentPage = pageKey;
      $$('[data-nav]').forEach(a => a.classList.toggle('active', a.dataset.nav === pageKey));
      buildToc(pageKey);
      $('#timer').hidden = pageKey !== 'cases';
      document.title = (pageKey === 'home' ? '' : PAGE_BY_KEY[pageKey].title + ' · ') + 'DS Interview Prep';
    }

    closeDrawer();
    const target = anchor ? document.getElementById(pageKey + '--' + anchor) : null;
    if (target) {
      revealTarget(target);
      requestAnimationFrame(() => { target.scrollIntoView({ block: 'start' }); flash(target); });
    } else {
      window.scrollTo(0, 0);
    }
    openOnNavigate = false;
    onScroll();
  }

  function revealTarget(el) {
    if (el.classList.contains('qcard') && openOnNavigate) el.open = true;
    const sol = el.closest('.case-solution');
    if (sol) setSolution(sol.parentElement, true);
    if (el.parentElement.classList.contains('case-card') && openOnNavigate) setSolution(el.parentElement, true);
  }

  function go(hash) {
    if (location.hash === hash) route(); else location.hash = hash;
  }

  window.addEventListener('hashchange', route);

  // Links inside the Markdown point to "#slug"; resolve them against the current page.
  document.addEventListener('click', e => {
    const a = e.target.closest('main a[href^="#"]');
    if (!a) return;
    const href = a.getAttribute('href');
    if (href.startsWith('#/')) return;
    e.preventDefault();
    go(`#/${currentPage}/${href.slice(1)}`);
  });

  /* ---------- Drawer ---------- */
  function closeDrawer() { document.body.classList.remove('drawer-open'); }
  $('#menuBtn').addEventListener('click', () => document.body.classList.toggle('drawer-open'));
  $('#overlay').addEventListener('click', closeDrawer);

  /* ---------- Progress & marks ---------- */
  function counts() {
    const qs = $$('.qcard');
    const known = qs.filter(c => state['q:' + c.dataset.qid] === 'known').length;
    const revise = qs.filter(c => state['q:' + c.dataset.qid] === 'revise').length;
    const secs = $$('#page-concepts h2');
    const secDone = secs.filter(h => state[`sec:concepts:${h.dataset.anchor}`]).length;
    const cases = $$('.case-card');
    const casesDone = cases.filter(c => state['case:' + c.dataset.caseId]).length;
    return {
      concepts: { done: secDone, total: secs.length, label: `${secDone} / ${secs.length} sections done` },
      questions: { done: known, total: qs.length, revise, label: `${known} / ${qs.length} known${revise ? ` · ${revise} to revise` : ''}` },
      cases: { done: casesDone, total: cases.length, label: `${casesDone} / ${cases.length} cases practised` },
    };
  }

  function progressHtml(c, compact) {
    const pct = c.total ? Math.round(100 * c.done / c.total) : 0;
    if (compact) return `<div class="nav-bar"><span style="width:${pct}%"></span></div><div class="nav-sub">${c.label}</div>`;
    return `<div class="prog-bar"><span style="width:${pct}%"></span></div><small>${c.label}</small>`;
  }

  function refreshMarks() {
    $$('.qcard').forEach(c => {
      const st = state['q:' + c.dataset.qid];
      if (st) c.dataset.status = st; else delete c.dataset.status;
      $$('.mark-btn', c).forEach(b => b.classList.toggle('on', b.dataset.mark === st));
    });
    $$('[data-done]').forEach(b => {
      const on = !!state[b.dataset.done];
      b.classList.toggle('on', on);
      const isCase = b.dataset.done.startsWith('case:');
      b.textContent = on ? (isCase ? '✓ Practised' : '✓ Done') : (isCase ? 'Mark practised' : 'Mark done');
      if (isCase) b.closest('.case-card').classList.toggle('done', on);
    });
    $$('[data-toc-check]').forEach(s => { s.textContent = state[s.dataset.tocCheck] ? '✓' : ''; });

    const c = counts();
    $$('[data-progress]').forEach(el => {
      const k = el.dataset.progress;
      el.innerHTML = c[k] ? progressHtml(c[k], false) : '<small>Read this first</small>';
    });
    $$('[data-nav-progress]').forEach(el => {
      const k = el.dataset.navProgress;
      el.innerHTML = c[k] ? progressHtml(c[k], true) : '';
    });
    applyFilters();
  }

  /* ---------- Question filters ---------- */
  let fStatus = 'all';
  let fDiff = 'all';
  function applyFilters() {
    const cards = $$('.qcard');
    let shown = 0;
    cards.forEach(c => {
      const st = state['q:' + c.dataset.qid];
      const okS = fStatus === 'all' || (fStatus === 'unseen' ? !st : st === fStatus);
      const okD = fDiff === 'all' || c.dataset.diff === fDiff;
      c.hidden = !(okS && okD);
      if (!c.hidden) shown++;
    });
    // Hide section headings whose questions are all filtered out.
    $$('#page-questions h2').forEach(h => {
      let n = h.nextElementSibling, any = false, hasCards = false;
      while (n && n.tagName !== 'H2') {
        if (n.classList.contains('qcard')) { hasCards = true; if (!n.hidden) any = true; }
        n = n.nextElementSibling;
      }
      h.hidden = hasCards && !any;
    });
    const stats = $('#qStats');
    if (stats) stats.innerHTML = `Showing <b>${shown}</b> of ${cards.length}`;
  }

  /* ---------- Case solutions ---------- */
  function setSolution(card, open) {
    const sol = $('.case-solution', card);
    const btn = $('[data-action="toggle-solution"]', card);
    sol.hidden = !open;
    btn.textContent = open ? 'Hide solution' : 'Show solution';
    btn.setAttribute('aria-expanded', String(open));
  }

  /* ---------- Random picks ---------- */
  function randomFrom(list) { return list[Math.floor(Math.random() * list.length)]; }
  function randomQuestion() {
    let cards = $$('.qcard').filter(c => !c.hidden);
    if (!cards.length) cards = $$('.qcard');
    const card = randomFrom(cards);
    card.open = false;
    go(`#/questions/${card.dataset.anchor}`);
  }
  function randomCase() {
    const pending = $$('.case-card').filter(c => !state['case:' + c.dataset.caseId]);
    const card = randomFrom(pending.length ? pending : $$('.case-card'));
    setSolution(card, false);
    go(`#/cases/${card.dataset.caseId}`);
  }

  /* ---------- Delegated actions ---------- */
  document.addEventListener('click', e => {
    const t = e.target.closest('button');
    if (!t) return;

    if (t.dataset.mark) {
      const card = t.closest('.qcard');
      const key = 'q:' + card.dataset.qid;
      setMark(key, state[key] === t.dataset.mark ? null : t.dataset.mark);
      return;
    }
    if (t.dataset.done) {
      e.preventDefault();
      setMark(t.dataset.done, state[t.dataset.done] ? null : 1);
      return;
    }
    if (t.dataset.filterStatus) {
      fStatus = t.dataset.filterStatus;
      $$('[data-filter-status]').forEach(b => b.classList.toggle('on', b === t));
      applyFilters();
      return;
    }
    if (t.dataset.filterDiff) {
      fDiff = t.dataset.filterDiff;
      $$('[data-filter-diff]').forEach(b => b.classList.toggle('on', b === t));
      applyFilters();
      return;
    }
    switch (t.dataset.action) {
      case 'random-q': randomQuestion(); break;
      case 'random-case': randomCase(); break;
      case 'toggle-solution': {
        const card = t.closest('.case-card');
        setSolution(card, $('.case-solution', card).hidden);
        break;
      }
      case 'toggle-answers': {
        const cards = $$('.qcard');
        const open = !cards.every(c => c.open);
        cards.forEach(c => { c.open = open; });
        t.textContent = open ? 'Close all answers' : 'Open all answers';
        break;
      }
    }
  });

  /* ---------- Scroll: reading bar, back-to-top, TOC highlight ---------- */
  const readBar = $('#readBar');
  const toTop = $('#toTop');
  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      readBar.style.width = (max > 0 ? (100 * window.scrollY / max) : 0) + '%';
      toTop.hidden = window.scrollY < 600;

      if (currentPage === 'home') return;
      const offset = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--header-h'), 10) + 40;
      let current = null;
      $$('#page-' + currentPage + ' h2').forEach(h => { if (h.getBoundingClientRect().top < offset) current = h.id; });
      $$('[data-toc]').forEach(a => a.classList.toggle('active', a.dataset.toc === current));
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

  /* ---------- Theme ---------- */
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  function effectiveTheme() { return document.documentElement.dataset.theme || (media.matches ? 'dark' : 'light'); }
  function syncCodeTheme() {
    const dark = effectiveTheme() === 'dark';
    $('#hl-dark').disabled = !dark;
    $('#hl-light').disabled = dark;
  }
  $('#themeBtn').addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('dsprep-theme', next); } catch (e) { /* ignore */ }
    syncCodeTheme();
  });
  media.addEventListener('change', syncCodeTheme);
  syncCodeTheme();

  /* ---------- Case timer ---------- */
  const TIMER_SECONDS = 25 * 60;
  let remaining = TIMER_SECONDS;
  let timerId = null;
  const timerEl = $('#timer');
  const timerTime = $('#timerTime');
  const timerStart = $('#timerStart');
  function renderTimer() {
    const m = Math.floor(remaining / 60), s = remaining % 60;
    timerTime.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    timerEl.classList.toggle('done', remaining === 0);
  }
  function stopTimer() { clearInterval(timerId); timerId = null; timerStart.textContent = 'Start'; }
  timerStart.addEventListener('click', () => {
    if (timerId) { stopTimer(); return; }
    if (remaining === 0) remaining = TIMER_SECONDS;
    timerStart.textContent = 'Pause';
    timerId = setInterval(() => {
      remaining = Math.max(0, remaining - 1);
      renderTimer();
      if (remaining === 0) { stopTimer(); timerTime.textContent = "Time's up"; }
    }, 1000);
  });
  $('#timerReset').addEventListener('click', () => { stopTimer(); remaining = TIMER_SECONDS; renderTimer(); });

  /* ---------- Keyboard & print ---------- */
  document.addEventListener('keydown', e => {
    if (e.key === '/' && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      searchInput.focus();
    } else if (e.key === 'Escape') {
      closeDrawer();
    }
  });
  window.addEventListener('beforeprint', () => {
    $$('.qcard').forEach(c => { c.dataset.wasOpen = c.open ? '1' : ''; c.open = true; });
  });
  window.addEventListener('afterprint', () => {
    $$('.qcard').forEach(c => { c.open = c.dataset.wasOpen === '1'; });
  });

  /* ---------- Boot ---------- */
  buildNav();
  if (renderAll()) {
    currentPage = null;
    refreshMarks();
    route();
  }
})();
