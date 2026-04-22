/**
 * custom-select.js — replaces native <select> elements with div-based dropdowns.
 * Works in embedded/WebView browsers (Tesla, HbbTV, older Chromium) where native
 * <select> popups fail to open. The original <select> stays in the DOM (hidden)
 * so all existing .addEventListener('change', ...) handlers keep working.
 *
 * Usage:
 *   customSelect('mySelectId');           // replace one select
 *   customSelect('mySelectId', label);    // optional placeholder label
 *
 * Re-call after dynamically repopulating options:
 *   customSelect('mySelectId');
 */
(function () {
  'use strict';

  function customSelect(selectId) {
    const orig = document.getElementById(selectId);
    if (!orig) return;

    // Remove any previous custom wrapper so we can rebuild after option changes
    const prevWrapper = document.getElementById(selectId + '-cs-wrapper');
    if (prevWrapper) prevWrapper.remove();

    // Hide the native select (keep it for value/event plumbing)
    orig.style.display = 'none';

    // ── Build wrapper ──────────────────────────────────────────────────────
    const wrapper = document.createElement('div');
    wrapper.id = selectId + '-cs-wrapper';
    wrapper.className = 'cs-wrapper';
    wrapper.setAttribute('role', 'combobox');
    wrapper.setAttribute('aria-haspopup', 'listbox');
    wrapper.setAttribute('aria-expanded', 'false');
    wrapper.setAttribute('tabindex', '0');

    // Selected-value button
    const trigger = document.createElement('div');
    trigger.className = 'cs-trigger';

    const label = document.createElement('span');
    label.className = 'cs-label';

    const arrow = document.createElement('span');
    arrow.className = 'cs-arrow';
    arrow.innerHTML = '&#9662;';

    trigger.appendChild(label);
    trigger.appendChild(arrow);

    // Options list
    const list = document.createElement('ul');
    list.className = 'cs-list';
    list.setAttribute('role', 'listbox');

    wrapper.appendChild(trigger);
    wrapper.appendChild(list);

    // ── Populate options ───────────────────────────────────────────────────
    function buildOptions() {
      list.innerHTML = '';
      Array.from(orig.options).forEach((opt, i) => {
        const li = document.createElement('li');
        li.className = 'cs-option' + (opt.selected ? ' cs-selected' : '');
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', opt.selected ? 'true' : 'false');
        li.dataset.value = opt.value;
        li.textContent = opt.textContent;
        li.addEventListener('mousedown', function (e) {
          e.preventDefault();
        });
        li.addEventListener('click', function () {
          orig.value = opt.value;
          orig.dispatchEvent(new Event('change', { bubbles: true }));
          syncLabel();
          close();
        });
        list.appendChild(li);
      });
    }

    function syncLabel() {
      const sel = orig.options[orig.selectedIndex];
      label.textContent = sel ? sel.textContent : '';
      list.querySelectorAll('.cs-option').forEach(li => {
        const active = li.dataset.value === orig.value;
        li.classList.toggle('cs-selected', active);
        li.setAttribute('aria-selected', active ? 'true' : 'false');
      });
    }

    function open() {
      wrapper.classList.add('cs-open');
      wrapper.setAttribute('aria-expanded', 'true');
      // Scroll selected item into view
      const sel = list.querySelector('.cs-selected');
      if (sel) sel.scrollIntoView({ block: 'nearest' });
    }

    function close() {
      wrapper.classList.remove('cs-open');
      wrapper.setAttribute('aria-expanded', 'false');
    }

    function toggle() {
      wrapper.classList.contains('cs-open') ? close() : open();
    }

    trigger.addEventListener('click', function (e) {
      e.stopPropagation();
      toggle();
    });

    wrapper.addEventListener('keydown', function (e) {
      const opts = list.querySelectorAll('.cs-option');
      const cur = list.querySelector('.cs-selected');
      let idx = Array.from(opts).indexOf(cur);

      if (e.key === 'Enter' || e.key === ' ') { toggle(); e.preventDefault(); }
      else if (e.key === 'Escape') { close(); }
      else if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (!wrapper.classList.contains('cs-open')) open();
        const next = opts[Math.min(idx + 1, opts.length - 1)];
        if (next) next.click();
      }
      else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const prev = opts[Math.max(idx - 1, 0)];
        if (prev) prev.click();
      }
    });

    // Close when clicking outside
    document.addEventListener('click', function (e) {
      if (!wrapper.contains(e.target)) close();
    });

    buildOptions();
    syncLabel();

    orig.parentNode.insertBefore(wrapper, orig);

    // Watch for external option changes (dynamic population)
    const observer = new MutationObserver(function () {
      buildOptions();
      syncLabel();
    });
    observer.observe(orig, { childList: true, subtree: true });
  }

  // ── Styles ─────────────────────────────────────────────────────────────
  const style = document.createElement('style');
  style.textContent = `
    .cs-wrapper {
      position: relative;
      display: inline-block;
      width: 100%;
      font-family: inherit;
      font-size: 0.875rem;
      cursor: pointer;
      user-select: none;
      -webkit-user-select: none;
    }
    .cs-trigger {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 0.75rem;
      padding: 0.625rem 1rem;
      color: #2d3748;
      font-weight: 500;
      box-sizing: border-box;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .cs-wrapper:focus .cs-trigger,
    .cs-wrapper.cs-open .cs-trigger {
      border-color: #2d3748;
      box-shadow: 0 0 0 2px rgba(45,55,72,0.18);
      outline: none;
    }
    .cs-arrow {
      color: #6b7280;
      font-size: 0.75rem;
      margin-left: 8px;
      flex-shrink: 0;
      transition: transform 0.15s;
    }
    .cs-wrapper.cs-open .cs-arrow {
      transform: rotate(180deg);
    }
    .cs-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }
    .cs-list {
      display: none;
      position: absolute;
      top: calc(100% + 4px);
      left: 0;
      right: 0;
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 0.75rem;
      padding: 4px;
      max-height: 260px;
      overflow-y: auto;
      z-index: 9999;
      box-shadow: 0 8px 24px rgba(0,0,0,0.14);
      list-style: none;
      margin: 0;
    }
    .cs-wrapper.cs-open .cs-list {
      display: block;
    }
    .cs-option {
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      color: #374151;
      font-size: 0.875rem;
      transition: background 0.1s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cs-option:hover {
      background: #f3f4f6;
    }
    .cs-option.cs-selected {
      background: #ede9fe;
      color: #5b21b6;
      font-weight: 600;
    }
  `;
  document.head.appendChild(style);

  window.customSelect = customSelect;
})();
