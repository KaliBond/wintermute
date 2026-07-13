// Neural Nations — single source of truth for navbar + footer.
// Edit THIS object; every migrated page updates. No build step.
const SITE = {
  brand: { label: 'Neural Nations', badge: 'NN', href: 'index.html' },
  primary: [
    { label: 'Home',       href: 'index.html' },
    { label: 'Start Here', href: 'start-here.html' },
    { label: 'Model',      href: 'model.html' },
    { label: 'Datasets',   href: 'datasets.html' },
    { label: 'Explore',    href: 'explore.html' },
  ],
  insights: {
    label: 'Insights',
    items: [
      { label: 'Results at a Glance',   href: 'results.html' },
      { label: 'Validation & Limits',   href: 'validation.html' },
      { label: 'Advanced Analysis',     href: 'cams-advanced-analysis.html' },
      { label: 'Escalation Archetypes', href: 'escalation-archetypes.html' },
      { label: 'Failure Modes Map',     href: 'cams-failure-modes.html' },
      { label: 'Failure Modes Story',   href: 'failure-modes-story.html' },
      { label: 'Attractors',            href: 'attractors.html' },
      { label: 'Predictions 2027–2028', href: 'predictions.html' },
      { divider: true },
      { label: 'Aha! Maps · Bellinger', href: 'aha-maps.html' },
    ],
  },
  trailing: [
    { label: 'Research',       href: 'research/' },
    { label: 'Research Diary', href: 'research-diary.html' },
    { label: 'Mindscapes',     href: 'mindscapes.html' },
    { label: 'FAQ',            href: 'faq.html' },
    { label: 'Contact',        href: 'contact.html' },
  ],
  frameworkLink: { label: 'Framework ↗', href: 'framework/index.html' },
  ctas: {
    github:     'https://github.com/KaliBond/wintermute',
    newsletter: 'newsletter.html',
  },
  footerLinks: [
    { label: 'Home',            href: 'index.html' },
    { label: 'Explore',         href: 'explore.html' },
    { label: 'Research Diary',  href: 'research-diary.html' },
  ],
  footerNote: '© 2026 Kari McKern — Neural Nations CAMS Research. Built as open science — contributions & forks welcome at <a href="https://github.com/KaliBond/wintermute" target="_blank">KaliBond/wintermute</a>.',
};

function currentPage() {
  const f = location.pathname.split('/').filter(Boolean).pop() || 'index.html';
  return f.includes('.') ? f : 'index.html';
}

function escAttr(s) {
  return String(s).replace(/"/g, '&quot;');
}

function linkHTML(item, cls) {
  const classAttr = cls ? ` class="${cls}"` : '';
  return `<a href="${escAttr(item.href)}"${classAttr}>${item.label}</a>`;
}

function renderChrome() {
  const here = currentPage();
  const isActive = (href) => href === here;
  const inDropdown = SITE.insights.items.some(i => i.href === here);

  const primaryHTML = SITE.primary.map(i =>
    `<a href="${escAttr(i.href)}" class="nav-link${isActive(i.href) ? ' active' : ''}">${i.label}</a>`
  ).join('\n            ');

  const dropdownItemsHTML = SITE.insights.items.map(i =>
    i.divider ? '<hr class="menu-divider">' : `<a href="${escAttr(i.href)}">${i.label}</a>`
  ).join('\n                    ');

  const trailingHTML = SITE.trailing.map(i =>
    `<a href="${escAttr(i.href)}" class="nav-link${isActive(i.href) ? ' active' : ''}">${i.label}</a>`
  ).join('\n            ');

  const fw = SITE.frameworkLink;
  const frameworkHTML = `<a href="${escAttr(fw.href)}" class="nav-link${isActive(fw.href) ? ' active' : ''}">${fw.label}</a>`;

  const mobilePrimaryHTML = SITE.primary.map(i => linkHTML(i)).join('\n        ');
  const mobileTrailingHTML = SITE.trailing.map(i => linkHTML(i)).join('\n        ');
  const mobileInsightsHTML = SITE.insights.items.map(i =>
    i.divider ? '<hr class="menu-divider">' : linkHTML(i)
  ).join('\n        ');

  const navHTML = `
    <div class="nav-container">
        <a href="${escAttr(SITE.brand.href)}" class="nav-logo">
            <span class="nav-logo-badge">${SITE.brand.badge}</span>
            <span class="nav-logo-text">${SITE.brand.label}</span>
        </a>
        <div class="nav-menu">
            ${primaryHTML}
            <div class="nav-dropdown${inDropdown ? ' open' : ''}">
                <button class="nav-link nav-dropdown-toggle${inDropdown ? ' active' : ''}">${SITE.insights.label} &#9662;</button>
                <div class="nav-dropdown-menu">
                    ${dropdownItemsHTML}
                </div>
            </div>
            ${trailingHTML}
            ${frameworkHTML}
        </div>
        <div class="nav-actions">
            <a href="${escAttr(SITE.ctas.github)}" class="nav-btn-github" target="_blank">
                <svg fill="currentColor" viewBox="0 0 24 24" style="width:15px;height:15px;flex-shrink:0"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.21 11.39.6.11.79-.26.79-.58v-2.05c-3.34.73-4.04-1.41-4.04-1.41-.55-1.39-1.33-1.76-1.33-1.76-1.09-.74.08-.73.08-.73 1.2.08 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.49 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23.96-.27 1.98-.4 3-.4s2.04.13 3 .4c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.19.69.8.57C20.57 21.8 24 17.3 24 12 24 5.37 18.63 0 12 0z"/></svg> GitHub
            </a>
            <a href="${escAttr(SITE.ctas.newsletter)}" class="nav-btn-newsletter">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" style="width:15px;height:15px;flex-shrink:0"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg> <span class="newsletter-text">Newsletter</span>
            </a>
            <button class="nav-hamburger" id="nav-hamburger" aria-label="Open menu">
                <span></span><span></span><span></span>
            </button>
        </div>
    </div>
    <div class="nav-mobile" id="nav-mobile">
        ${mobilePrimaryHTML}
        ${mobileTrailingHTML}
        ${linkHTML(fw)}
        <hr class="mobile-divider">
        <div class="mobile-section-title">${SITE.insights.label}</div>
        ${mobileInsightsHTML}
    </div>`;

  const footerLinksHTML = SITE.footerLinks.map(i =>
    `<a href="${escAttr(i.href)}" style="color:#1f77b4;text-decoration:none;font-size:0.9em;font-weight:600;">${i.label}</a>`
  ).join('\n        ');

  const footerHTML = `
    <nav style="margin-bottom:10px;display:flex;gap:20px;justify-content:center;flex-wrap:wrap;">
        ${footerLinksHTML}
    </nav>
    <p>${SITE.footerNote}</p>`;

  const navMount = document.getElementById('site-nav');
  const footMount = document.getElementById('site-footer');
  if (navMount) navMount.innerHTML = navHTML;
  if (footMount) footMount.innerHTML = footerHTML;
}
