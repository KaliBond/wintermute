/* ============================================================
   Epiphenomenon — Chart.js defaults bound to CAMS DS · ACADEMIC.
   Cream paper · warm-teal ink · single-accent · journal palette.
   ============================================================ */
(function () {
  if (typeof Chart === 'undefined') return;

  const css = (name, fallback) => {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  };

  // Token palette — paper / academic mode.
  const NN = window.NN_CHART = {
    text:     css('--ink-900', '#1B2A33'),
    textMute: css('--ink-500', '#5A6E78'),
    grid:     css('--paper-3', '#E2DAC3'),
    surface:  css('--paper-0', '#FBF7EC'),

    // semantic series — restrained journal palette
    sino:     css('--accent-teal',      '#2F6F7D'),  /* coordination / sino */
    russo:    css('--mode-abstraction', '#8A5BB8'),  /* counter-signal / russo */
    shield:   css('--accent-ochre',     '#D98B2B'),  /* SHI / shield series */
    shieldHi: css('--accent-ochre-hi',  '#EFA94A'),

    // thresholds
    alarm:    '#C8961A',
    critical: css('--state-collapse', '#8B2E22'),
    zero:     'rgba(27, 42, 51, 0.18)',

    // layers (paper-friendly equivalents)
    mythic:    '#6B3F7A',
    interface: '#2F6F7D',
    material:  '#2E8B6B'
  };

  // Hex → rgba(...,a)
  const rgba = (hex, a) => {
    if (!hex || hex[0] !== '#') return hex;
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
  };
  NN.rgba = rgba;

  // Global Chart.js defaults
  Chart.defaults.color = NN.textMute;
  Chart.defaults.borderColor = NN.grid;
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  Chart.defaults.font.size = 11;

  NN.scales = (overrides = {}) => ({
    x: {
      grid: { color: NN.grid, drawTicks: false, lineWidth: 0.5 },
      ticks: { maxTicksLimit: 9, font: { size: 10, family: 'JetBrains Mono, ui-monospace, monospace' }, color: NN.textMute, padding: 6 },
      border: { color: NN.text, width: 1 }
    },
    y: {
      grid: { color: NN.grid, drawTicks: false, lineWidth: 0.5 },
      ticks: { font: { size: 10, family: 'JetBrains Mono, ui-monospace, monospace' }, color: NN.textMute, padding: 8 },
      border: { color: NN.text, width: 1 }
    },
    ...overrides
  });

  NN.baseOpts = (overrides = {}) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index', intersect: false,
        backgroundColor: NN.surface,
        titleColor: NN.text,
        bodyColor: NN.text,
        borderColor: NN.text, borderWidth: 1,
        padding: 10, cornerRadius: 0,
        titleFont: { family: 'JetBrains Mono, monospace', size: 11, weight: 500 },
        bodyFont:  { family: 'Inter, sans-serif', size: 12 }
      }
    },
    scales: NN.scales(),
    ...overrides
  });

  NN.legend = (overrides = {}) => ({
    display: true,
    labels: {
      font: { size: 11, family: 'Inter, sans-serif' },
      color: NN.textMute,
      boxWidth: 14, boxHeight: 4,
      usePointStyle: false
    },
    ...overrides
  });

  NN.line = (data, color, opts = {}) => ({
    data, borderColor: color,
    backgroundColor: opts.fill ? rgba(color, 0.10) : 'transparent',
    fill: !!opts.fill,
    tension: opts.tension ?? 0.32,
    borderWidth: opts.width ?? 1.6,
    pointRadius: opts.pointRadius ?? 0,
    pointHoverRadius: 4,
    pointBackgroundColor: color,
    label: opts.label,
    spanGaps: true
  });

  NN.bar = (data, color, opts = {}) => ({
    data,
    backgroundColor: rgba(color, opts.alpha ?? 0.55),
    borderColor: color,
    borderWidth: 1,
    borderRadius: 0,
    label: opts.label
  });

  NN.threshLine = (labels, value, color, label) => ({
    data: labels.map(() => value),
    borderColor: color,
    borderDash: [4, 4],
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
    label
  });
})();
