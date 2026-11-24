/* SDRL interface and numerical repair layer.
   Loaded after app.js and before init(). Element IDs are retained for compatibility. */
(() => {
  "use strict";

  // Replace the defective self-loop Laplacian used by the live system calculator.
  canon.lambda2 = window.SDRL_MATH.lambda2;
  canon.system = window.SDRL_MATH.system;

  Object.assign(BASINS.PEAK_EXPANSION, {
    sub: "High-coherence expansion basin",
    meaning: "The institutional web is confident, well coupled and capable of long commitments. Its strength is real. So is the temptation to mistake present coordination for a permanent condition."
  });
  Object.assign(BASINS.NAVIGATIONAL, {
    sub: "Reconstruction and confident adjustment",
    meaning: "Purpose and institutional capacity reinforce one another. The society can still correct course without first breaking the institutions that carry the correction."
  });
  Object.assign(BASINS.EARLY_STRAIN, {
    sub: "Contested transition basin",
    meaning: "Stress is becoming visible before the institutional web has settled on a response. Old expectations remain in force, but their fit to present conditions is increasingly disputed."
  });
  Object.assign(BASINS.PRAETORIAN, {
    sub: "Security-dominant crisis basin",
    meaning: "Operational coordination is shifting towards organised force while stress outruns capacity. Short command cycles gain authority at the expense of deliberation and institutional memory."
  });
  Object.assign(BASINS.ARCHIPELAGO, {
    sub: "Fragmented-coordination basin",
    meaning: "Cooperation persists in separated institutional islands. The formal whole remains visible, but its parts increasingly coordinate through local networks rather than through a common centre."
  });
  Object.assign(BASINS.MASS_FORMATION, {
    sub: "Threat-coordinated mass-response basin",
    meaning: "Material weakness and cognitive separation coincide with intense coordination around a threatening object. This is a structural reading, not a diagnosis of citizens or a population estimate."
  });
  Object.assign(BASINS.PHANTOM, {
    sub: "Cultural persistence with material decline",
    meaning: "Cognitive and cultural functions remain comparatively viable while the material substrate weakens. The forms of continuity survive, but their material support is becoming less reliable."
  });

  Object.assign(ZEIT, {
    PEAK_EXPANSION: "The institutional climate is expansive. Shared purposes appear obvious, long commitments feel affordable, and coordination needs little coercion. The shadow of this confidence is closure: success makes the present order difficult to criticise and its assumptions difficult to inspect.",
    NAVIGATIONAL: "The climate is purposeful and comparatively open. Institutions can treat disagreement as information because cooperation still appears reciprocal. Practical repair carries more authority than nostalgia, and the recent past supplies lessons rather than a demand for restoration.",
    EARLY_STRAIN: "The climate is unsettled rather than uniformly pessimistic. Public expectations formed under earlier conditions now meet weaker delivery and thinner cooperation. Competing accounts of the same strain become politically important because each proposes a different place to put responsibility.",
    PRAETORIAN: "Public life becomes organised around threat, loyalty and immediate response. The tempo of decision quickens while its horizon shortens. Force does not merely protect the institutional web; it increasingly supplies the web’s working definition of order.",
    ARCHIPELAGO: "The common climate thins into local climates. Commitment remains intense inside trusted networks, while institutions outside those networks appear remote or procedural. Irony can become a form of self-protection, but it does not replace the coordination that has been lost.",
    MASS_FORMATION: "Attention narrows around a common object of fear or resentment. Coordination becomes intense, but its range contracts. Contradictory evidence is treated as a challenge to belonging rather than as information. This is a reading of institutional configuration, not a clinical claim about individuals.",
    PHANTOM: "Cultural life can remain eloquent while the systems that sustain it deteriorate. Ritual, scholarship and public memory continue to describe a coherent whole, even as material delivery becomes less dependable. The resulting mood is not simple denial. It is continuity sustained by forms whose foundations are changing."
  });

  Object.assign(SURV, {
    PEAK_EXPANSION: "Use the available coordination to build institutions that will still function when confidence weakens. Independent records, memory and legitimate dissent are most valuable before they become necessary.",
    NAVIGATIONAL: "The constructive opening is worth using, but not assuming. Strengthen Archive and Lore alongside {top}, so that present problem-solving becomes durable institutional learning rather than a temporary administrative success.",
    EARLY_STRAIN: "Distinguish weak delivery from weak legitimacy. Work across institutional boundaries and protect {top} without making every other function subordinate to it. Lateral bonds matter most while the next configuration is still unsettled.",
    PRAETORIAN: "Separate formal authority from operational coordination. Protect independent records, civilian deliberation and long-horizon decision forums. A security apparatus can coordinate a crisis; it cannot by itself replenish the capacities that the crisis consumes.",
    ARCHIPELAGO: "Map where coordination actually works. Local networks can carry real functions, but bridges between them determine whether fragmentation remains pluralism or becomes isolation. Treat symbolic unity and actual capacity as different questions.",
    MASS_FORMATION: "Keep evidential channels open and avoid turning every disagreement into a loyalty test. Preserve independent Archive and Lore functions, and maintain material networks that do not depend on a single coordinating story.",
    PHANTOM: "Value the surviving cultural capacity without asking it to substitute for material repair. Connect Archive, Lore and Craft to reliable delivery. The task is to let memory inform reconstruction rather than use memory to postpone it."
  });

  Object.assign(DELIB, {
    PEAK_EXPANSION: "Deliberation benefits from confidence and low coordination costs. Its blind spot is agreement itself: assumptions go untested because challenge feels unnecessary.",
    NAVIGATIONAL: "Deliberation can still change the course of the system. Evidence, practical judgment and institutional trust reinforce one another.",
    EARLY_STRAIN: "Deliberation becomes a contest over interpretation. Evidence remains available, but competing accounts of decline and responsibility narrow its audience.",
    PRAETORIAN: "Deliberation is compressed by emergency tempo. Threat framing rewards immediate response and treats reflection as delay.",
    ARCHIPELAGO: "Deliberation persists within enclaves and weakens between them. The public square remains, but fewer participants expect it to coordinate action.",
    MASS_FORMATION: "Deliberation is subordinated to belonging. Contradiction is received as disloyalty, and shared attention becomes narrower as coordination intensifies.",
    PHANTOM: "Deliberation can remain sophisticated in cultural enclaves while losing contact with material delivery. Eloquence and systemic health diverge."
  });

  function sourceShort(s) {
    const src = s.source || {};
    if (src.kind === "annual-ensemble") return `${s.years.length} annual rows · five-pass ensemble`;
    if (src.kind === "annual-ens") return `${s.years.length} annual source rows · repeated states retained`;
    return `${s.years.length} sampled observations · unified panel`;
  }

  function sourceLong(s) {
    const src = s.source || {}, repeated = Number.isFinite(src.repeated) ? src.repeated : 0;
    if (src.kind === "annual-ensemble") {
      return `${src.label}, ${s.years[0]}–${s.years[s.years.length - 1]}. Mean non-NA scores are used when at least three of five scorers are present, rounded to one decimal. This lineage differs from the unified panel; no 2026 row is manufactured.`;
    }
    if (src.kind === "annual-ens") {
      return `${src.label}, ${s.years[0]}–${s.years[s.years.length - 1]}. ${repeated} adjacent rows repeat the previous scored state. They are explicit repository rows, not dashboard interpolation, and should not be read as independent annual rescoring.`;
    }
    return `${src.label || "JUNO unified panel"}, ${s.years.length} scored observations. ${src.reason || "Source sampling retained"}. Missing years remain absent.`;
  }

  function scopeMixRepaired() {
    const s = soc();
    const kLo = Math.max(0, s.years.findIndex(y => y >= state.lo));
    const kHi = s.years.findIndex(y => y > state.hi);
    const hi = kHi === -1 ? s.years.length : kHi;
    const counts = REGIME.map(() => 0);
    for (let k = kLo; k < hi; k++) counts[s.R[k]]++;
    const total = Math.max(1, hi - kLo);
    document.getElementById("scopeMix").innerHTML =
      `<div class="mix-row"><span class="lbl" style="color:var(--cyan)">scored observations</span><span class="bar"></span><span class="pct">${hi - kLo}</span></div>` +
      REGIME.map(r => {
        const n = counts[r.i];
        return `<div class="mix-row"><span class="lbl">${r.label}</span><span class="bar"><i style="width:${(n / total * 100).toFixed(1)}%;background:${r.color}"></i></span><span class="pct">${(n / total * 100).toFixed(0)}%</span></div>`;
      }).join("");
  }
  scopeMix = scopeMixRepaired;

  function renderDossierRepaired() {
    const k = yidx(), body = document.getElementById("dossierBody");
    if (k < 0) { body.innerHTML = ""; return; }
    const s = soc(), d = basinInputs(k), basin = classifyBasin(d), bInfo = BASINS[basin];
    const threat = threatLevel(d), regime = REGIME_OF(s.R[k]);
    const V = s.nodeV[k];
    const ordered = NODES.map((name, i) => ({ name, value: V[i] })).sort((a, b) => b.value - a.value);
    const topNode = ordered[0].name, lowNode = ordered[ordered.length - 1].name;
    const hollow = NODES.filter((n, i) => s.A[k][i] > s.K[k][i] * 1.3);
    const sig = NODES.map((_, i) => s.A[k][i] * s.C[k][i] * (s.K[k][i] - s.S_v[k][i]));
    const inverted = NODES.filter((_, i) => sig[i] < 0);
    const src = sourceLong(s);

    let trajectory = "Two earlier observations are required before a trajectory can be described.";
    if (k >= 2) {
      const prior = basinInputs(k - 2);
      const skDelta = d.sk - prior.sk, bDelta = d.b - prior.b;
      const direction = skDelta > 0.1 ? "stress is rising relative to capacity" : skDelta < -0.1 ? "stress is easing relative to capacity" : "the stress-to-capacity ratio is broadly unchanged";
      const bondDirection = bDelta < -0.03 ? "bond strength is contracting" : bDelta > 0.03 ? "bond strength is expanding" : "bond strength is broadly unchanged";
      trajectory = `Across the prior two observations, ${direction} (Δsk ${skDelta >= 0 ? "+" : ""}${skDelta.toFixed(2)}) and ${bondDirection} (ΔB ${bDelta >= 0 ? "+" : ""}${bDelta.toFixed(2)}). This describes the sampled path; it is not a forecast.`;
    }

    const powerReading = d.praetorian > 0.25
      ? "Shield viability is materially above Helm viability. The index is consistent with security-dominant coordination, but does not by itself identify who holds political power."
      : d.praetorian < -0.15
      ? "Helm viability is materially above Shield viability. Executive capacity is stronger than the security counterweight in this scored state."
      : "Helm and Shield are comparatively balanced in this scored state.";

    const dynamicPortrait = ` In ${state.year}, ${topNode} carries the highest node viability and ${lowNode} the lowest. That contrast gives this general basin reading its society-year-specific tension.`;
    document.getElementById("dossierSub").textContent = `${state.society} · ${state.year} · ${sourceShort(s)}`;
    body.innerHTML = `
      <div class="basin-row portrait-chips">
        <span class="basin-chip" style="color:${regime.color};border-color:${regime.color}55">JUNO · ${regime.label}</span>
        <span class="basin-chip" style="color:${bInfo.col};border-color:${bInfo.col}55">BASIN · ${bInfo.name}</span>
        <span class="threat-chip" style="color:${threat.col};border-color:${threat.col}55">PRESSURE ${threat.level}</span>
      </div>
      <section class="level measured">
        <h3><span>01</span> Measured state</h3>
        <div class="level-grid">
          <div class="block"><h4>Scored configuration</h4>
            <p>V̄ ${d.sys.V_mean.toFixed(2)} · B_eff ${d.sys.B_eff.toFixed(3)} · V_min ${d.sys.V_min.toFixed(2)}</p>
            <p>${topNode} is highest at ${ordered[0].value.toFixed(1)}. ${lowNode} is lowest at ${ordered[ordered.length - 1].value.toFixed(1)}.</p>
            ${inverted.length ? `<p>Negative cognitive activation: ${inverted.join(", ")}.</p>` : ""}
            ${hollow.length ? `<p class="hline">Abstraction exceeds 1.3× capacity in ${hollow.join(", ")}.</p>` : ""}
          </div>
          <div class="block"><h4>Source and sampling</h4><p>${src}</p></div>
        </div>
      </section>
      <section class="level inferred">
        <h3><span>02</span> Inferred dynamics</h3>
        <div class="level-grid">
          <div class="block"><h4>Trajectory</h4><p>${trajectory}</p>
            <p class="fineprint">sk ${d.sk.toFixed(2)} · praetorian ${d.praetorian.toFixed(2)} · cog_gap ${d.cog_gap.toFixed(2)}</p></div>
          <div class="block"><h4>Institutional configuration</h4><p>${powerReading}</p>
            <p>The JUNO regime is the locked classification of viability, bond strength and minimum activation. The basin is a separate dashboard heuristic using sk, praetorian balance and cognitive–material separation. They can legitimately differ.</p></div>
        </div>
      </section>
      <section class="level interpretive">
        <h3><span>03</span> Interpretive portrait</h3>
        <div class="level-grid">
          <div class="block"><h4>Zeitgeist</h4><p>${ZEIT[basin]}${dynamicPortrait}</p>
            <p class="fineprint">This is a model-generated reading of institutional configuration, not survey evidence of public emotion.</p></div>
          <div class="block"><h4>Working implication</h4><p>${SURV[basin].replace("{top}", topNode)}</p></div>
        </div>
      </section>`;
  }
  renderDossier = renderDossierRepaired;

  const baseRenderSysVit = renderSysVit;
  renderSysVit = function () {
    baseRenderSysVit();
    const el = document.getElementById("sysvit");
    const heading = el && el.querySelector(".sysvit-head h4");
    if (heading) heading.textContent = "Systemic Pressures";
    const downstream = el && el.querySelector(".downstream h5");
    if (downstream) downstream.textContent = "The Deliberative Climate";
  };

  function updateProbeSource() {
    const note = document.getElementById("probeSrc");
    if (!note) return;
    if (!state.probe) { note.textContent = "Open Model Laboratory to construct an editable state."; note.classList.remove("stale"); return; }
    const sourceYear = state.probeAt == null ? state.year : state.probeAt;
    const sourceSociety = state.probeSociety || state.society;
    const stale = sourceYear !== state.year || sourceSociety !== state.society;
    note.classList.toggle("stale", stale);
    note.textContent = `${state.probeModified ? "Modified laboratory state" : "Laboratory state"} sourced from ${sourceSociety} ${sourceYear}.` +
      (stale ? ` The live selection is ${state.society} ${state.year}; reset to load it.` : "");
  }

  function formatFaders() {
    document.querySelectorAll("#scorerCards input[type=range]").forEach(input => {
      input.step = "0.1";
      const value = Number(input.value);
      input.style.setProperty("--range-progress", `${value * 10}%`);
      const output = document.getElementById(`o-${input.dataset.node}-${input.dataset.vital}`);
      if (output) output.textContent = value.toFixed(1);
      input.setAttribute("aria-valuetext", value.toFixed(1));
    });
  }

  const baseResetProbe = resetProbe;
  resetProbe = function () {
    baseResetProbe();
    state.probeAt = state.year;
    state.probeSociety = state.society;
    state.probeModified = false;
    formatFaders();
    updateProbeSource();
  };

  function syncYearSliderRepaired() {
    const s = soc(), slider = document.getElementById("yearSlider");
    if (!slider) return;
    const k = Math.max(0, yidx()), last = Math.max(0, s.years.length - 1);
    slider.min = "0"; slider.max = String(last); slider.step = "1"; slider.value = String(k);
    slider.disabled = last === 0;
    slider.style.setProperty("--range-progress", `${last ? k / last * 100 : 0}%`);
    slider.setAttribute("aria-valuetext", `${state.year}, observation ${k + 1} of ${s.years.length}`);
    const input = document.getElementById("yearInput");
    if (input) {
      input.min = String(s.years[0]); input.max = String(s.years[last]); input.value = String(state.year);
      input.setAttribute("aria-label", `${state.society} year; nearest scored observation is used`);
    }
    const samplingNote = document.getElementById("yearSampling");
    if (samplingNote) samplingNote.textContent = sourceShort(s);
    const prev = document.getElementById("yearPrev"), next = document.getElementById("yearNext");
    if (prev) prev.disabled = k <= 0;
    if (next) next.disabled = k >= last;
  }
  syncYearSlider = syncYearSliderRepaired;

  selectSociety = function (name) {
    state.society = name;
    togglePlay(false);
    markActiveLink();
    fullRange();
    state.probe = null;
    setYear(soc().years[soc().years.length - 1]);
    resetProbe();
  };

  const baseRenderAll = renderAll;
  renderAll = function () {
    baseRenderAll();
    syncYearSliderRepaired();
    updateProbeSource();
  };

  const baseTogglePlay = togglePlay;
  togglePlay = function (force) {
    baseTogglePlay(force);
    const button = document.getElementById("playBtn");
    if (button) {
      const label = state.playing ? "Pause this society" : "Follow this society";
      button.setAttribute("aria-label", label); button.title = label;
    }
  };

  function updateFooter() {
    const mount = window.SDRL_MOUNT;
    const observations = Object.values(D.societies).reduce((n, s) => n + s.years.length, 0);
    const years = Object.values(D.societies).flatMap(s => s.years);
    const foot = document.getElementById("sourceFoot");
    if (foot) {
      foot.textContent = `Source: JUNO unified panel with ${mount.mounted.length} annual-mounted societies and ${mount.fallback.length} unified fallbacks. ${observations.toLocaleString()} scored observations, ${Math.min(...years)}–${Math.max(...years)}. No missing years are interpolated. Mounted rows use JUNO v1.2-Final; fallback rows retain their original panel labels.`;
    }
    const detail = document.getElementById("sourceDetail");
    if (detail) {
      const usa = D.societies.USA && D.societies.USA.source;
      detail.textContent = `Annual ENS rows are explicit repository states. Repeated adjacent states are retained and disclosed, not treated as proof of independent annual rescoring. The Observed Adaptive Corridor is recomputed from the mounted distribution as within-bin p16, p50 and p84 values. USA source: ${usa ? usa.label : "unified panel"}.`;
    }
    const note = document.getElementById("dataNote");
    if (note) note.textContent = "The year control steps through scored observations. Direct year entry snaps to the nearest available row and reports that row everywhere. Laboratory presets retain their source year until reset.";
  }

  function setupControls() {
    const input = document.getElementById("yearInput");
    if (input) {
      const commit = () => { if (input.value !== "") setYear(Number(input.value)); };
      input.addEventListener("change", commit);
      input.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); commit(); input.blur(); } });
    }
    const step = direction => {
      const s = soc(), k = yidx(), next = Math.max(0, Math.min(s.years.length - 1, k + direction));
      setYear(s.years[next]);
    };
    const prev = document.getElementById("yearPrev"), next = document.getElementById("yearNext");
    if (prev) prev.addEventListener("click", () => step(-1));
    if (next) next.addEventListener("click", () => step(1));
    const slider = document.getElementById("yearSlider");
    if (slider) slider.setAttribute("aria-describedby", "yearSampling");

    const cards = document.getElementById("scorerCards");
    if (cards) cards.addEventListener("input", event => {
      const target = event.target;
      if (!target.matches("input[type=range]")) return;
      state.probeModified = true;
      const value = Number(target.value);
      target.style.setProperty("--range-progress", `${value * 10}%`);
      target.setAttribute("aria-valuetext", value.toFixed(1));
      const output = document.getElementById(`o-${target.dataset.node}-${target.dataset.vital}`);
      if (output) output.textContent = value.toFixed(1);
      updateProbeSource();
    });
    const play = document.getElementById("playBtn");
    if (play) { play.title = "Follow this society"; play.setAttribute("aria-label", "Follow this society"); }
    updateFooter();
    syncYearSliderRepaired();
    updateProbeSource();
  }

  // Smooth the machine's node radius before positioning its labels. This removes
  // label jumping when a long node value changes the geometry between frames.
  function stabiliseMachineLabels() {
    if (constState.els) {
      constState.els.nodeEls.forEach(({ body, nameTxt }) => {
        const target = Number(body.getAttribute("r"));
        if (!Number.isFinite(target)) return;
        const previous = Number(body.dataset.smoothR);
        const size = Number.isFinite(previous) ? previous * 0.82 + target * 0.18 : target;
        body.dataset.smoothR = String(size);
        body.setAttribute("r", size.toFixed(2));
        nameTxt.setAttribute("y", (-(size + 8)).toFixed(2));
      });
    }
    requestAnimationFrame(stabiliseMachineLabels);
  }

  const baseInit = init;
  init = function () {
    if (window.__sdrlInitialised) return;
    baseInit();
    window.__sdrlInitialised = true;
    setupControls();
    formatFaders();
    updateProbeSource();
    stabiliseMachineLabels();
  };
})();
