/* ADVERSARY-02 entity trajectories: interactive chart used on explore.html and adversary-02-results.html.
   Data: analysis/adversary_02/part2/entity_personality_entity_year.csv (2021-2024 = Part 2 primary pool, 7 runs;
   2010-2020 = Part 1 pooled baseline, a different scoring method, so 2020 -> 2021 is a method seam).
   Each point is [year, energy (K - S), focus (A x C)], rounded to 2 dp. Usage: <div class="a02-chart"></div> */
(function () {
  var DATA = {"BYD":[[2010,1.05,29.62],[2011,0.36,28.66],[2012,0.15,27.42],[2013,0.93,29.74],[2014,1.3,32.49],[2015,1.96,35.55],[2016,2.03,36.41],[2017,1.63,35.91],[2018,1.74,36.59],[2019,1.01,35.93],[2020,2.38,43.12],[2021,2.98,46.83],[2022,3.88,52.8],[2023,4.04,55.95],[2024,3.88,55.59]],"CATL":[[2011,-0.82,25.85],[2012,0.0,28.03],[2013,0.61,29.19],[2014,0.93,30.28],[2015,1.29,32.53],[2016,1.77,35.6],[2017,2.21,39.09],[2018,2.63,42.41],[2019,2.6,44.61],[2020,2.88,47.1],[2021,4.01,53.99],[2022,3.98,54.82],[2023,3.97,56.3],[2024,4.07,56.91]],"China":[[2010,2.02,33.51],[2011,1.73,33.4],[2012,1.46,32.32],[2013,1.72,36.57],[2014,1.76,37.95],[2015,1.3,38.25],[2016,1.79,40.23],[2017,2.37,43.81],[2018,2.01,43.31],[2019,1.78,43.14],[2020,1.48,43.78],[2021,2.33,41.81],[2022,0.44,36.22],[2023,0.86,35.81],[2024,1.28,37.48]],"Evergrande":[[2010,1.37,27.85],[2011,1.25,28.36],[2012,0.99,28.2],[2013,1.49,29.97],[2014,1.19,29.66],[2015,0.97,29.92],[2016,0.87,29.84],[2017,1.0,31.05],[2018,0.73,30.65],[2019,-0.32,26.51],[2020,-1.7,22.66],[2021,-5.93,8.72],[2022,-6.5,6.6],[2023,-7.25,4.28],[2024,-7.78,2.49]],"Huawei":[[2010,2.47,38.43],[2011,2.41,38.54],[2012,2.43,38.89],[2013,2.8,41.75],[2014,2.97,43.56],[2015,3.1,45.92],[2016,3.05,46.41],[2017,3.19,47.84],[2018,2.63,47.88],[2019,1.58,48.66],[2020,0.67,46.18],[2021,1.4,47.32],[2022,1.53,47.04],[2023,3.37,56.33],[2024,3.75,57.44]],"Longfor":[[2010,2.29,34.61],[2011,2.23,34.74],[2012,2.08,34.76],[2013,2.57,36.0],[2014,2.46,36.46],[2015,2.56,37.54],[2016,2.59,39.34],[2017,2.83,40.63],[2018,2.87,41.61],[2019,3.08,42.92],[2020,2.89,43.25],[2021,3.53,44.24],[2022,1.64,37.99],[2023,1.02,34.78],[2024,0.35,32.77]],"SMIC":[[2010,0.37,26.16],[2011,-0.32,24.88],[2012,0.62,27.68],[2013,1.2,29.97],[2014,1.34,31.58],[2015,1.56,33.2],[2016,1.65,33.52],[2017,1.29,33.66],[2018,1.28,34.79],[2019,1.41,36.39],[2020,0.7,36.24],[2021,1.82,40.45],[2022,1.55,41.19],[2023,1.23,41.09],[2024,1.94,43.62]],"Vanke":[[2010,3.04,40.79],[2011,2.92,40.72],[2012,3.02,40.74],[2013,3.07,41.24],[2014,2.83,41.8],[2015,1.58,37.43],[2016,1.02,35.1],[2017,2.29,39.97],[2018,2.66,41.82],[2019,2.72,42.28],[2020,2.54,42.48],[2021,2.38,38.61],[2022,1.46,35.49],[2023,-0.87,27.77],[2024,-3.02,19.37]]};
  var COLORS = {Evergrande:'#C84C3E', Huawei:'#6b3fa0', CATL:'#2F6F7D', BYD:'#1f77b4', SMIC:'#d97706', China:'#2ca02c', Longfor:'#8c564b', Vanke:'#555555'};
  var ORDER = ['China','Evergrande','Vanke','Longfor','SMIC','Huawei','BYD','CATL'];
  var X0 = 0, Y0 = 30.25, NS = 'http://www.w3.org/2000/svg';
  var css = '.a02-wrap{font-family:-apple-system,"Segoe UI",sans-serif;font-size:14px;color:#1B2A33}' +
    '.a02-ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:10px}' +
    '.a02-ctl label{cursor:pointer}.a02-ctl select{padding:3px 6px;border:1px solid #cfc8b0;border-radius:4px;background:#fff}' +
    '.a02-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}' +
    '.a02-chip{border:2px solid;border-radius:14px;padding:2px 10px;font-size:12.5px;font-weight:600;background:#fff;cursor:pointer;line-height:1.5}' +
    '.a02-chip[aria-pressed=false]{opacity:.35}' +
    '.a02-svg{width:100%;height:auto;background:#fff;border:1px solid #e2dac3;border-radius:6px}' +
    '.a02-note{font-size:12.5px;color:#5d6577;margin:8px 0 0;line-height:1.5}' +
    '.a02-wrap details{margin-top:8px;font-size:13px}.a02-wrap table{border-collapse:collapse;width:100%;margin-top:6px}' +
    '.a02-wrap th,.a02-wrap td{padding:4px 8px;border-bottom:1px solid #e2dac3;text-align:right}.a02-wrap th:first-child,.a02-wrap td:first-child{text-align:left}';
  function el(n, a, t) { var e = document.createElementNS(NS, n); for (var k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; return e; }
  function h(n, a, t) { var e = document.createElement(n); for (var k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; return e; }
  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }

  function init(root) {
    if (!document.getElementById('a02-css')) { var s = h('style', {id: 'a02-css'}); s.textContent = css; document.head.appendChild(s); }
    var state = {view: 'map', part1: false, on: {}};
    ORDER.forEach(function (n) { state.on[n] = true; });
    root.innerHTML = '';
    var wrap = h('div', {'class': 'a02-wrap'});
    var ctl = h('div', {'class': 'a02-ctl'});
    var sel = h('select', {'aria-label': 'Chart view'});
    [['map', 'Energy × focus map'], ['energy', 'Energy over time (K − S)'], ['focus', 'Focus over time (A × C)']].forEach(function (o) { sel.appendChild(h('option', {value: o[0]}, o[1])); });
    var lab = h('label'); var cb = h('input', {type: 'checkbox'}); lab.appendChild(cb); lab.appendChild(document.createTextNode(' Also show the 2010–2020 Part 1 baseline'));
    ctl.appendChild(sel); ctl.appendChild(lab);
    var chips = h('div', {'class': 'a02-chips', role: 'group', 'aria-label': 'Entities'});
    var holder = h('div'); var note = h('p', {'class': 'a02-note'}); var tbl = h('details');
    wrap.appendChild(ctl); wrap.appendChild(chips); wrap.appendChild(holder); wrap.appendChild(note); wrap.appendChild(tbl); root.appendChild(wrap);
    ORDER.forEach(function (n) {
      var b = h('button', {type: 'button', 'class': 'a02-chip', 'aria-pressed': 'true'}, n);
      b.style.borderColor = COLORS[n]; b.style.color = COLORS[n];
      b.onclick = function () { state.on[n] = !state.on[n]; b.setAttribute('aria-pressed', state.on[n]); draw(); };
      chips.appendChild(b);
    });
    sel.onchange = function () { state.view = sel.value; draw(); };
    cb.onchange = function () { state.part1 = cb.checked; draw(); };

    function draw() {
      holder.innerHTML = '';
      var W = 640, H = 380, m = {l: 52, r: 24, t: 14, b: 44};
      var yMin = state.part1 ? 2010 : 2021, pts = {}, names = ORDER.filter(function (n) { return state.on[n]; });
      names.forEach(function (n) { pts[n] = DATA[n].filter(function (p) { return p[0] >= yMin; }); });
      var xs = [], ys = [], xv, yv, xl, yl, map = state.view === 'map';
      names.forEach(function (n) { pts[n].forEach(function (p) {
        if (map) { xs.push(p[1]); ys.push(p[2]); } else { xs.push(p[0]); ys.push(state.view === 'energy' ? p[1] : p[2]); } }); });
      if (!xs.length) { holder.appendChild(h('p', {'class': 'a02-note'}, 'Select at least one entity.')); note.textContent = ''; return; }
      var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs), ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys);
      if (map) { xlo = Math.min(xlo, X0); xhi = Math.max(xhi, X0); ylo = Math.min(ylo, Y0); yhi = Math.max(yhi, Y0); }
      else if (state.view === 'energy') { ylo = Math.min(ylo, 0); yhi = Math.max(yhi, 0); } else { ylo = Math.min(ylo, Y0); yhi = Math.max(yhi, Y0); }
      var px = (xhi - xlo) * 0.06 || 1, py = (yhi - ylo) * 0.08 || 1; xlo -= map ? px : 0.3; xhi += map ? px : 0.3; ylo -= py; yhi += py;
      xv = function (v) { return m.l + (v - xlo) / (xhi - xlo) * (W - m.l - m.r); };
      yv = function (v) { return H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b); };
      var svg = el('svg', {'class': 'a02-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img',
        'aria-label': map ? 'Scatter of energy against focus for each entity, 2021 to 2024' : 'Line chart of ' + state.view + ' by year for each entity'});
      // axes box + ticks
      svg.appendChild(el('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: '#fff', stroke: '#cfc8b0'}));
      function ticks(lo, hi, n) { var step = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step))), r = step / mag, s = r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10; step = s * mag; var out = [], t = Math.ceil(lo / step) * step; for (; t <= hi + 1e-9; t += step) out.push(+t.toFixed(6)); return out; }
      ticks(ylo, yhi, 6).forEach(function (t) { var y = yv(t); svg.appendChild(el('line', {x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: '#eee'})); svg.appendChild(el('text', {x: m.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#5d6577'}, String(t))); });
      if (map) ticks(xlo, xhi, 7).forEach(function (t) { var x = xv(t); svg.appendChild(el('line', {x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: '#eee'})); svg.appendChild(el('text', {x: x, y: H - m.b + 15, 'text-anchor': 'middle', 'font-size': 11, fill: '#5d6577'}, String(t))); });
      else { var yrs = []; for (var y = Math.ceil(xlo); y <= xhi; y++) if (y >= yMin) yrs.push(y);
        yrs.forEach(function (y) { if (state.part1 && y % 2) return; svg.appendChild(el('text', {x: xv(y), y: H - m.b + 15, 'text-anchor': 'middle', 'font-size': 11, fill: '#5d6577'}, String(y))); }); }
      svg.appendChild(el('text', {x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 12, fill: '#1B2A33'}, map ? 'Energy (K − S): capacity left after stress' : 'Year'));
      var yt = el('text', {transform: 'translate(14,' + ((m.t + H - m.b) / 2) + ') rotate(-90)', 'text-anchor': 'middle', 'font-size': 12, fill: '#1B2A33'}, map || state.view === 'focus' ? 'Focus (A × C): clarity × coherence' : 'Energy (K − S)');
      svg.appendChild(yt);
      // reference lines
      function ref(x1, x2, y1, y2, label, lx, ly, anchor) { svg.appendChild(el('line', {x1: x1, x2: x2, y1: y1, y2: y2, stroke: '#888', 'stroke-dasharray': '5 4'})); if (label) svg.appendChild(el('text', {x: lx, y: ly, 'font-size': 10.5, fill: '#888', 'text-anchor': anchor || 'start'}, label)); }
      if (map) { ref(xv(X0), xv(X0), m.t, H - m.b, 'energy 0', xv(X0) + 4, m.t + 11); ref(m.l, W - m.r, yv(Y0), yv(Y0), 'focus 30.25', W - m.r - 4, yv(Y0) - 4, 'end'); }
      else if (state.view === 'energy') ref(m.l, W - m.r, yv(0), yv(0), 'energy 0', W - m.r - 4, yv(0) - 4, 'end');
      else ref(m.l, W - m.r, yv(Y0), yv(Y0), 'focus 30.25 (analyst-chosen line)', W - m.r - 4, yv(Y0) - 4, 'end');
      if (!map && state.part1) { var sx = xv(2020.5); ref(sx, sx, m.t, H - m.b, null); svg.appendChild(el('text', {x: sx + 4, y: m.t + 24, 'font-size': 10.5, fill: '#888'}, 'method seam →')); svg.appendChild(el('text', {x: sx - 4, y: m.t + 24, 'font-size': 10.5, fill: '#888', 'text-anchor': 'end'}, '← Part 1 baseline')); }
      // series
      names.forEach(function (n) {
        var P = pts[n], c = COLORS[n];
        function xy(p) { return map ? [xv(p[1]), yv(p[2])] : [xv(p[0]), yv(state.view === 'energy' ? p[1] : p[2])]; }
        function path(arr, dash, op) { if (arr.length < 2) return; var d = arr.map(function (p, i) { var q = xy(p); return (i ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' ');
          svg.appendChild(el('path', {d: d, fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-dasharray': dash || '', opacity: op || 1})); }
        var p2 = P.filter(function (p) { return p[0] >= 2021; }), p1 = P.filter(function (p) { return p[0] <= 2020; });
        path(p1, '3 3', 0.5); path(p2, '', 1);
        if (p1.length && p2.length && !map) path([p1[p1.length - 1], p2[0]], '1 4', 0.45);
        P.forEach(function (p) { var q = xy(p), last = p[0] === 2024, e = el('circle', {cx: q[0].toFixed(1), cy: q[1].toFixed(1), r: last ? 5 : 3.2, fill: p[0] >= 2021 ? c : '#fff', stroke: c, 'stroke-width': 1.5, opacity: p[0] >= 2021 ? 1 : 0.6});
          e.appendChild(el('title', {}, n + ' ' + p[0] + ': energy ' + f1(p[1]) + ', focus ' + f1(p[2]) + (p[0] <= 2020 ? ' (Part 1 baseline)' : ''))); svg.appendChild(e); });
        var L = P[P.length - 1], q = xy(L); svg.appendChild(el('text', {x: (q[0] + 8).toFixed(1), y: (q[1] + 4).toFixed(1), 'font-size': 11, 'font-weight': 700, fill: c}, n));
      });
      holder.appendChild(svg);
      note.textContent = (map ? 'Each line runs 2021 → 2024; the big dot is 2024. ' : '') + 'Primary pool, 7 scorer-runs. Dashed lines split the chart into quadrants; the focus line (30.25) was an analyst choice, not part of the sealed predictions.' +
        (state.part1 ? ' Open dots and thin dashes are the Part 1 baseline, scored by a different method: treat the step from 2020 to 2021 as a seam, not a real change.' : '');
      // table
      var rows = '<tr><th>Entity</th><th>Energy 2021</th><th>Energy 2024</th><th>Focus 2021</th><th>Focus 2024</th></tr>';
      names.forEach(function (n) { var a = DATA[n].filter(function (p) { return p[0] === 2021; })[0], b = DATA[n].filter(function (p) { return p[0] === 2024; })[0]; rows += '<tr><td>' + n + '</td><td>' + f1(a[1]) + '</td><td>' + f1(b[1]) + '</td><td>' + f1(a[2]) + '</td><td>' + f1(b[2]) + '</td></tr>'; });
      tbl.innerHTML = '<summary>Show the numbers (2021 and 2024)</summary><table>' + rows + '</table>';
    }
    draw();
  }
  function boot() { var n = document.querySelectorAll('.a02-chart'); for (var i = 0; i < n.length; i++) init(n[i]); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
