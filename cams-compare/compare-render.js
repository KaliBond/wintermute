/* ================================================================
   CAMS Â· Compare â€” rendering + UI wiring (part 2)
   ================================================================ */

/* ---------- Stat banner ---------- */
function renderBanner(){
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  const cell = (series, col) => {
    const v   = atYear(series,'meanV',YEAR);
    const sv  = atYear(series,'sigmaV',YEAR);
    const k   = atYear(series,'kappa',YEAR);
    const l2  = atYear(series,'lambda2',YEAR);
    const att = atYear(series,'attractor',YEAR);
    const al  = atYear(series,'alarm',YEAR);
    const has = v !== null;
    const kHex = has ? kappaTierHex(k) : INK5;
    const aHex = has ? (ATTRACTOR_HEX[att]||INK5) : INK5;
    return `
      <div class="cmp-soc-card">
        <div class="cmp-soc-name" style="color:${col}">${series ? series.soc : 'â€”'}</div>
        ${ has ? `
        <div class="cmp-stat-row">
          <div class="cmp-stat"><div class="cmp-sv" style="color:${col}">${fmt(v,1)}</div><div class="cmp-sl">VÌ„ mean value</div></div>
          <div class="cmp-stat"><div class="cmp-sv">${fmt(sv,1)}</div><div class="cmp-sl">Ïƒ_V spread</div></div>
          <div class="cmp-stat"><div class="cmp-sv" style="color:${kHex}">${fmt(k,2)}</div><div class="cmp-sl">Îº criticality</div></div>
          <div class="cmp-stat"><div class="cmp-sv">${fmt(l2,2)}</div><div class="cmp-sl">Î»â‚‚ connectivity</div></div>
        </div>
        <div class="cmp-att" style="border-color:${aHex}">
          <span class="cmp-att-dot" style="background:${aHex}"></span>
          <span class="cmp-att-name">${att}</span>
          <span class="cmp-att-alarm">alarm Â· ${al}</span>
        </div>` : `<div class="cmp-nodata">no data for ${YEAR}</div>` }
      </div>`;
  };
  const dvWrap = document.getElementById('cmp-banner');
  if(dvWrap) dvWrap.innerHTML = cell(sa, COL_A) + cell(sb, COL_B);
}

/* ---------- Overview tab ---------- */
function renderOverview(){
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  makeLine('c-health', [ dsA(sa,'meanV'), dsB(sb,'meanV') ],
    { scales:{ y:{ title:{ display:true, text:'VÌ„(t)  mean node value', color:INK5, font:{ size:10 } } } } });

  // Îº chart â€” clean two-line comparison (kernel Îº spans ~1â€“4 on this data,
  // so the 0.3â€“0.6 tier shading from the calibration table isn't drawn here;
  // tier labels live in the banner instead).
  makeLine('c-kappa', [ dsA(sa,'kappa'), dsB(sb,'kappa') ],
    { scales:{ y:{ min:0, title:{ display:true, text:'Îº(t)  cognitive criticality', color:INK5, font:{ size:10 } } } } });
  renderSoulGrid();
}

/* ---------- Soul / node snapshot grid (Overview) ---------- */
function renderSoulGrid(){
  const g = document.getElementById('soul-grid');
  if(!g) return;
  const blocks = [[SOC_A, COL_A], [SOC_B, COL_B]].map(([soc,col])=>{
    const series = computeSeries(soc);
    const ny = nearestYear(series, YEAR);
    const cards = NODES_CMP.map(node=>{
      const d = nodeAtYear(series, node, ny);
      if(!d) return `<div class="nc"><div class="nc-t">${node}</div><div class="nc-nd">â€”</div></div>`;
      const viHex = d.Vi>8 ? '#2E8B6B' : d.Vi>3 ? '#2F7ECC' : d.Vi>-1 ? '#D98B2B' : '#C84C3E';
      const bars = ['C','K','S','A'].map(m=>
        `<div class="nbr"><span class="nbl">${m}</span><span class="nbo"><span class="nbi" style="width:${d[m]/10*100}%;background:${MODE_HEX[m]}"></span></span></div>`).join('');
      return `<div class="nc">
        <div class="nc-layer" style="background:${NODE_HEX_CMP[node]}"></div>
        <div class="nc-t">${node}</div>
        <div class="nc-bars">${bars}</div>
        <div class="nc-vi" style="color:${viHex}">${fmt(d.Vi,1)}</div>
      </div>`;
    }).join('');
    return `<div class="soul-col">
      <div class="soul-head" style="color:${col}">${soc} <span>Â· ${ny ?? 'â€”'}</span></div>
      <div class="ng">${cards}</div>
    </div>`;
  }).join('');
  g.innerHTML = blocks;
}

/* ---------- Node Ã— metric grid tab ---------- */
let NODE_METRIC = 'Vi';
function renderNodeGrid(){
  const con = document.getElementById('nc-container');
  if(!con) return;
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  con.innerHTML = NODES_CMP.map(n=>
    `<div class="card">
       <h3><span class="dot" style="background:${NODE_HEX_CMP[n]}"></span>${n}
         <span class="layer-tag">${NODE_LAYER_CMP[n]}</span></h3>
       <div class="cw-sm"><canvas id="ncm-${n}"></canvas></div>
     </div>`).join('');
  const yTitle = NODE_METRIC==='Vi' ? 'V_i node value' : MODE_NAME[NODE_METRIC];
  NODES_CMP.forEach(n=>{
    makeLine(`ncm-${n}`, [
      Object.assign({ label:SOC_A, data:nodeLineData(sa,n,NODE_METRIC), borderColor:COL_A }, LINE_BASE, { borderWidth:1.8 }),
      Object.assign({ label:SOC_B, data:nodeLineData(sb,n,NODE_METRIC), borderColor:COL_B }, LINE_BASE, { borderWidth:1.8 }),
    ], { scales:{ y:{ title:{ display:true, text:yTitle, color:INK5, font:{ size:9 } } } } });
  });
}

/* ---------- Coordination tab (Î»â‚‚ + bond) ---------- */
function renderCoordination(){
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  makeLine('c-lambda', [ dsA(sa,'lambda2'), dsB(sb,'lambda2') ],
    { scales:{ y:{ min:0, title:{ display:true, text:'Î»â‚‚  algebraic connectivity', color:INK5, font:{ size:10 } } } } });
  makeLine('c-bond', [
    Object.assign({ label:SOC_A, data:lineData(sa,'Bagg'), borderColor:COL_A, backgroundColor:COL_A_FILL, fill:true }, LINE_BASE),
    Object.assign({ label:SOC_B, data:lineData(sb,'Bagg'), borderColor:COL_B, backgroundColor:COL_B_FILL, fill:true }, LINE_BASE),
  ], { scales:{ y:{ min:0, max:1, title:{ display:true, text:'B(t)  aggregate bond strength', color:INK5, font:{ size:10 } } } } });
  makeLine('c-tau', [ dsA(sa,'tau'), dsB(sb,'tau') ],
    { scales:{ y:{ title:{ display:true, text:'Ï„ = KÌ„ / SÌ„  capacity-to-stress', color:INK5, font:{ size:10 } } } } });
}

/* ---------- Phase space tab ---------- */
function phaseScatter(id, soc, col){
  const el = document.getElementById(id);
  if(!el) return;
  if(charts[id]) charts[id].destroy();
  const series = computeSeries(soc);
  if(!series) return;
  const pts = series.years.map((y,i)=>({
    x: series.meanV[i], y: series.sigmaV[i], yr:y, att:series.attractor[i]
  }));
  const colors = pts.map(p=> ATTRACTOR_HEX[p.att] || col);
  charts[id] = new Chart(el, {
    type:'scatter',
    data:{ datasets:[{ data:pts, backgroundColor:colors, pointRadius:3, borderColor:'rgba(0,0,0,0.18)', borderWidth:0.5 }] },
    options: baseOpts({
      scales:{
        x:{ type:'linear', title:{ display:true, text:'VÌ„(t)', color:INK5, font:{ size:10 } } },
        y:{ title:{ display:true, text:'Ïƒ_V', color:INK5, font:{ size:10 } } },
      },
      plugins:{ tooltip:{ callbacks:{ label:(c)=>` ${c.raw.yr}: VÌ„=${fmt(c.raw.x,1)}, Ïƒ=${fmt(c.raw.y,1)} Â· ${c.raw.att}` } } },
    }),
  });
  // mark current year
  const cy = series.years.indexOf(YEAR) >= 0 ? YEAR : nearestYear(series, YEAR);
  const i = series.years.indexOf(cy);
  if(i>=0){
    charts[id].data.datasets.push({
      data:[{ x:series.meanV[i], y:series.sigmaV[i] }],
      backgroundColor:'transparent', borderColor:col, borderWidth:2, pointRadius:8, pointStyle:'circle'
    });
    charts[id].update();
  }
}
function renderPhase(){
  phaseScatter('c-ps-a', SOC_A, COL_A);
  phaseScatter('c-ps-b', SOC_B, COL_B);
  // basin legend
  const lg = document.getElementById('basin-legend');
  if(lg){
    lg.innerHTML = Object.entries(ATTRACTOR_HEX).map(([name,hex])=>
      `<span class="bl-item"><span class="bl-dot" style="background:${hex}"></span>${name}</span>`).join('');
  }
}

/* ---------- Divergence + structural events tab ---------- */
function renderDivergence(){
  const sa = computeSeries(SOC_A), sb = computeSeries(SOC_B);
  // build aligned diff over intersection years
  const yset = sa && sb ? sa.years.filter(y=> sb.years.indexOf(y)>=0) : [];
  const diff = (key)=> yset.map(y=>({ x:y, y: atYear(sa,key,y) - atYear(sb,key,y) }));
  makeLine('c-div-v', [
    Object.assign({ label:'Î”VÌ„', data:diff('meanV'), borderColor:'#2F6F7D', backgroundColor:'rgba(47,111,125,0.08)', fill:true }, LINE_BASE),
  ], { scales:{ y:{ title:{ display:true, text:`Î”VÌ„  (${SOC_A} âˆ’ ${SOC_B})`, color:INK5, font:{ size:10 } } } } });
  makeLine('c-div-k', [
    Object.assign({ label:'Î”Îº', data:diff('kappa'), borderColor:'#C84C3E', backgroundColor:'rgba(200,76,62,0.08)', fill:true }, LINE_BASE),
  ], { scales:{ y:{ title:{ display:true, text:`Î”Îº  (${SOC_A} âˆ’ ${SOC_B})`, color:INK5, font:{ size:10 } } } } });

  renderEvents();
}

// structural events = years where the attractor label changes
function structuralEvents(series){
  if(!series) return [];
  const ev = [];
  for(let i=1;i<series.years.length;i++){
    if(series.attractor[i] !== series.attractor[i-1]){
      ev.push({ year:series.years[i], from:series.attractor[i-1], to:series.attractor[i] });
    }
  }
  return ev;
}
function renderEvents(){
  [[SOC_A,'evl-a',COL_A],[SOC_B,'evl-b',COL_B]].forEach(([soc,id,col])=>{
    const el = document.getElementById(id);
    if(!el) return;
    const series = computeSeries(soc);
    const ev = structuralEvents(series);
    el.previousElementSibling && (el.previousElementSibling.style.color = col);
    if(!ev.length){ el.innerHTML = `<div class="ev-empty">No attractor transitions â€” ${soc} holds one basin across its record.</div>`; return; }
    el.innerHTML = ev.map(e=>{
      const toHex = ATTRACTOR_HEX[e.to] || INK5;
      return `<div class="er">
        <div class="ey">${e.year}</div>
        <div class="et"><span style="color:${ATTRACTOR_HEX[e.from]||INK5}">${e.from}</span>
          <span class="ev-arrow">â†’</span>
          <span style="color:${toHex};font-weight:600">${e.to}</span></div>
      </div>`;
    }).join('');
  });
}

/* ---------- Tab routing ---------- */
const RENDERERS = {
  overview: renderOverview,
  nodes: renderNodeGrid,
  coordination: renderCoordination,
  phase: renderPhase,
  divergence: renderDivergence,
};
let ACTIVE_TAB = 'overview';
function renderActive(){
  renderBanner();
  (RENDERERS[ACTIVE_TAB] || (()=>{}))();
}

/* ---------- Year-rule refresh without full rebuild ---------- */
function refreshYearRules(){
  Object.values(charts).forEach(c=>{ if(c && c.config){ c.config._yearRule = YEAR; c.update('none'); }});
}

/* ---------- Wiring ---------- */
function initCompare(){
  // populate society selects
  const socs = Object.keys(DATA);
  const selA = document.getElementById('sel-a'), selB = document.getElementById('sel-b');
  socs.forEach(s=>{
    selA.insertAdjacentHTML('beforeend', `<option value="${s}" ${s===SOC_A?'selected':''}>${s}</option>`);
    selB.insertAdjacentHTML('beforeend', `<option value="${s}" ${s===SOC_B?'selected':''}>${s}</option>`);
  });
  const { lo, hi } = xRange();
  const slider = document.getElementById('yr-slider');
  slider.min = lo; slider.max = hi; slider.value = Math.min(YEAR, hi);
  YEAR = +slider.value;
  document.getElementById('yr-disp').textContent = YEAR;

  selA.addEventListener('change', e=>{ SOC_A = e.target.value; resyncRange(); renderActive(); });
  selB.addEventListener('change', e=>{ SOC_B = e.target.value; resyncRange(); renderActive(); });

  slider.addEventListener('input', e=>{
    YEAR = +e.target.value;
    document.getElementById('yr-disp').textContent = YEAR;
    renderBanner();
    if(ACTIVE_TAB==='overview') renderSoulGrid();
    if(ACTIVE_TAB==='phase'){ renderPhase(); } else { refreshYearRules(); }
  });

  document.querySelectorAll('.tab').forEach(t=> t.addEventListener('click', ()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tp').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    ACTIVE_TAB = t.dataset.tab;
    document.getElementById('tab-'+ACTIVE_TAB).classList.add('active');
    renderActive();
  }));

  document.querySelectorAll('.mb').forEach(b=> b.addEventListener('click', ()=>{
    document.querySelectorAll('.mb').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    NODE_METRIC = b.dataset.m;
    renderNodeGrid();
  }));

  renderActive();
}
function resyncRange(){
  const { lo, hi } = xRange();
  const slider = document.getElementById('yr-slider');
  slider.min = lo; slider.max = hi;
  if(YEAR < lo){ YEAR = lo; } if(YEAR > hi){ YEAR = hi; }
  slider.value = YEAR;
  document.getElementById('yr-disp').textContent = YEAR;
}

