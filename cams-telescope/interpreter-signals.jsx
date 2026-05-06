/* ================================================================
   Signal generation engine v0.2 — calibrated to Aus 1895–2026
   Thresholds at 25th/75th percentiles of empirical operator ranges.
   Groups A–G per formal mapping document.
   ================================================================ */

const NODES = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'];

/* ── Confidence badge ── */
// HIGH / MED / LOW per mapping doc

function generateSignals(r, matrix) {
  const signals = {
    'Joint-State (Group A)': [],
    'System-Level (Group B)': [],
    'Slow-Loop Nodes (Group C)': [],
    'Fast-Loop Nodes (Group D)': [],
    'Pathologies (Group F)': [],
    'Secondary Phenomena': [],
  };

  const helmIdx = 0, shieldIdx = 1, loreIdx = 2, stewardsIdx = 3;
  const craftIdx = 4, handsIdx = 5, archiveIdx = 6, flowIdx = 7;
  const meanB = r.B.reduce((a,b) => a+b, 0) / r.B.length;
  const epsilon = r.meanA * r.meanS / Math.max(r.meanK, 0.001);
  const praetGap = r.V[shieldIdx] - r.V[helmIdx];
  const phantomGap = (r.V[loreIdx] + r.V[archiveIdx]) / 2 - (r.V[handsIdx] + r.V[craftIdx]) / 2;

  // ═══ GROUP A — Joint-state (highest confidence) ═══

  // Crisis composite: τ < 1.1 ∧ ε > 8.5 ∧ σ_V > 4.0
  if (r.tau < 1.1 && epsilon > 8.5 && r.sigmaV > 4.0) {
    signals['Joint-State (Group A)'].push({
      title: `Crisis composite — τ=${r.tau.toFixed(2)}, ε=${epsilon.toFixed(1)}, σ_V=${r.sigmaV.toFixed(2)}`,
      body: 'Decline-anxiety, gridlock framings, conspiracy genres, scapegoating waves co-occurring. Multiple stress indicators converge.',
      confidence: 'HIGH', exemplar: '1917, 1931, 1975, 1991, 2023',
      trove: '"broken" + "decline" co-frequency',
    });
  }

  // Praetorian state: Helm V < 6 ∧ Shield V > 15 ∧ Praet gap > +5
  if (r.V[helmIdx] < 6 && r.V[shieldIdx] > 15 && praetGap > 5) {
    signals['Joint-State (Group A)'].push({
      title: `Praetorian state — Helm V=${r.V[helmIdx].toFixed(1)}, Shield V=${r.V[shieldIdx].toFixed(1)}`,
      body: 'Securitisation of unrelated domains, strongman appeal, foreign-interference panic, leadership-as-spectacle.',
      confidence: 'HIGH', exemplar: '1975, 2010s, 2023',
      trove: '"strong leader" + "foreign interference"',
    });
  }

  // Buffering attractor: meanV > 15 ∧ σ_V < 2.5 ∧ τ > 1.5
  if (r.meanV > 15 && r.sigmaV < 2.5 && r.tau > 1.5) {
    signals['Joint-State (Group A)'].push({
      title: `Buffering attractor — V̄=${r.meanV.toFixed(1)}, τ=${r.tau.toFixed(2)}, σ_V=${r.sigmaV.toFixed(2)}`,
      body: 'Civic-progress, reformist-technocratic register, foreign-policy quiet. System operating in surplus.',
      confidence: 'HIGH', exemplar: '1955, 2004',
      trove: '"lucky country" + "consensus"',
    });
  }

  // Phantom Type I: Archive V > 19 ∧ Hands V < 9 ∧ Stewards V high
  if (r.V[archiveIdx] > 19 && r.V[handsIdx] < 9 && r.V[stewardsIdx] > 16) {
    signals['Joint-State (Group A)'].push({
      title: `Phantom Type I — Archive=${r.V[archiveIdx].toFixed(1)}, Hands=${r.V[handsIdx].toFixed(1)}, Stewards=${r.V[stewardsIdx].toFixed(1)}`,
      body: 'Heritage politics, ceremonial controversies, "way of life" rhetoric crowding out material grievance.',
      confidence: 'MED', exemplar: '1920s, 2010s late',
      trove: '"Australian values" + "tradition"',
    });
  }

  // Pre-bifurcation: ε > 8 ∧ Λ > 1.9 ∧ τ < 1.2
  if (epsilon > 8 && r.Lambda > 1.9 && r.tau < 1.2) {
    signals['Joint-State (Group A)'].push({
      title: `Pre-bifurcation — ε=${epsilon.toFixed(1)}, Λ=${r.Lambda.toFixed(3)}, τ=${r.tau.toFixed(2)}`,
      body: 'Conspiracy genres, "secret enemy" framings, anti-elite intensification. System approaching bifurcation point.',
      confidence: 'HIGH', exemplar: '1931, 1975, 2010s',
      trove: '"secret influence" + "elite"',
    });
  }

  // ═══ GROUP B — System-level operators ═══

  if (r.tau < 1.1) {
    signals['System-Level (Group B)'].push({
      title: `Low affective tone — τ=${r.tau.toFixed(2)}`,
      body: 'Decline-anxiety, civilisational-twilight rhetoric, "going to ruin" framings.',
      confidence: 'HIGH', exemplar: '1917, 1931, 1975, 2023',
      trove: '"national decline", "civilisation"',
    });
  }
  if (r.tau > 1.5) {
    signals['System-Level (Group B)'].push({
      title: `High affective tone — τ=${r.tau.toFixed(2)}`,
      body: 'Civic-progress, expansionist, "advancement of" register.',
      confidence: 'HIGH', exemplar: '1900s, 1950s–60s, 2000s',
      trove: '"great future", "march of progress"',
    });
  }

  if (epsilon > 8) {
    signals['System-Level (Group B)'].push({
      title: `Overreach — ε=${epsilon.toFixed(1)}`,
      body: 'Ideological-combat exhaustion, satire collapse, ironic-nihilist register when sustained.',
      confidence: 'MED', exemplar: '1970s, 2010s',
      trove: '"satire" + "dead", "post-political"',
    });
  }
  if (epsilon >= 4.5 && epsilon <= 7) {
    signals['System-Level (Group B)'].push({
      title: `Balanced overreach — ε=${epsilon.toFixed(1)}`,
      body: 'Pragmatist register, technical-managerial, "common sense" framings.',
      confidence: 'MED', exemplar: '1920s, 1960s',
      trove: '"common sense" + "practical"',
    });
  }

  if (r.Lambda > 1.9) {
    signals['System-Level (Group B)'].push({
      title: `High Λ (coordination failure) — λ₂=${r.Lambda.toFixed(3)}`,
      body: 'Gridlock / "broken system" framings, treason discourse, factional combat. Node heterogeneity is extreme.',
      confidence: 'HIGH', exemplar: '1975, 2010s, 2023',
      trove: '"gridlock", "broken parliament"',
    });
  }
  if (r.Lambda < 1.3) {
    signals['System-Level (Group B)'].push({
      title: `Low Λ (coordination success) — λ₂=${r.Lambda.toFixed(3)}`,
      body: 'Consensus politics, "adults in the room", reformist energy.',
      confidence: 'HIGH', exemplar: '1955–65, 1983–95, 1996–2007',
      trove: '"consensus", "bipartisan"',
    });
  }

  if (meanB < 0.52) {
    signals['System-Level (Group B)'].push({
      title: `Low bond strength — B̄=${meanB.toFixed(3)}`,
      body: 'Atomised political identity, "two Australias", regional-secession rhetoric.',
      confidence: 'MED', exemplar: '1972, 1982',
      trove: '"two nations", "us and them"',
    });
  }

  if (r.sigmaV > 4.5) {
    signals['System-Level (Group B)'].push({
      title: `High dispersion — σ_V=${r.sigmaV.toFixed(2)}`,
      body: 'Simultaneous incompatible discourses; press fragments by audience.',
      confidence: 'HIGH', exemplar: '1975, 1982, 2017',
      trove: '(sentiment dispersion in coverage)',
    });
  }
  if (r.sigmaV < 2.5) {
    signals['System-Level (Group B)'].push({
      title: `Low dispersion — σ_V=${r.sigmaV.toFixed(2)}`,
      body: 'Single coherent national narrative dominant in press.',
      confidence: 'HIGH', exemplar: '1911, 1955, 2004',
      trove: '(sentiment uniformity)',
    });
  }

  if (r.meanV < 8) {
    signals['System-Level (Group B)'].push({
      title: `Existential vitality — V̄=${r.meanV.toFixed(1)}`,
      body: 'Existential-stakes rhetoric ("survival of the nation"), apocalyptic register.',
      confidence: 'HIGH', exemplar: '1931',
      trove: '"survival of"',
    });
  }
  if (r.meanV > 17) {
    signals['System-Level (Group B)'].push({
      title: `Peak vitality — V̄=${r.meanV.toFixed(1)}`,
      body: 'Triumphalist register, "envy of the world".',
      confidence: 'HIGH', exemplar: '1999–2007',
      trove: '"lucky country", "envy"',
    });
  }

  if (praetGap > 5) {
    signals['System-Level (Group B)'].push({
      title: `Praetorian gap — Shield−Helm = ${praetGap.toFixed(1)}`,
      body: 'Security-bracket expansion to non-security domains.',
      confidence: 'HIGH', exemplar: '1975, 2010s, 2020s',
      trove: '"national security" + (non-security domain)',
    });
  }
  if (praetGap < 0) {
    signals['System-Level (Group B)'].push({
      title: `Civilian authority — Shield−Helm = ${praetGap.toFixed(1)}`,
      body: 'Genuine civilian-political authority register.',
      confidence: 'HIGH', exemplar: '1920s, 1940s',
      trove: '"the prime minister" + governance vocabulary',
    });
  }

  // ═══ GROUP C — Slow-loop nodes ═══

  if (r.V[archiveIdx] > 21) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Archive flourishing — V=${r.V[archiveIdx].toFixed(1)}`,
      body: 'Heritage-flourishing, ceremonial saturation; Anzac-style memorialisation.',
      confidence: 'HIGH', exemplar: '1925 (Gallipoli), 2014–18 (centenary)',
      trove: '"Anzac", "tradition"',
    });
  }

  if (r.Gamma[loreIdx] > 3) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Lore drift — Γ=${r.Gamma[loreIdx].toFixed(2)} (A/C ratio)`,
      body: 'Ideological fragmentation, culture-war saturation, identitarian combat.',
      confidence: 'HIGH', exemplar: '2010s',
      trove: '"culture war", "values"',
    });
  }
  if (r.V[loreIdx] < 12) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Lore thinning — V=${r.V[loreIdx].toFixed(1)}`,
      body: 'Symbolic-system thinning; "no shared story" framings.',
      confidence: 'MED', exemplar: '2010s',
      trove: '"no shared", "fragmented"',
    });
  }

  if (r.V[stewardsIdx] > 16) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Steward dominance — V=${r.V[stewardsIdx].toFixed(1)}`,
      body: 'Property/asset panic, NIMBY discourse, "boomers vs young" framing.',
      confidence: 'HIGH', exemplar: '1999–2007, 2018–24',
      trove: '"property crisis", "house prices"',
    });
  }

  if (r.V[helmIdx] < 6) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Helm decoupling — V=${r.V[helmIdx].toFixed(1)}`,
      body: '"Out of touch elite", anti-establishment populism, leader-as-clown discourse.',
      confidence: 'HIGH', exemplar: '1931, 2010s',
      trove: '"out of touch", "ivory tower"',
    });
  }
  if (r.V[helmIdx] > 16) {
    signals['Slow-Loop Nodes (Group C)'].push({
      title: `Helm authority — V=${r.V[helmIdx].toFixed(1)}`,
      body: 'Legitimist register, deferential coverage, "the prime minister has decided".',
      confidence: 'HIGH', exemplar: '1955, 2004',
      trove: '(deferential leader vocabulary)',
    });
  }

  // ═══ GROUP D — Fast-loop nodes ═══

  if (r.V[shieldIdx] > 16 && r.V[helmIdx] < 12) {
    signals['Fast-Loop Nodes (Group D)'].push({
      title: `Shield dominant — V=${r.V[shieldIdx].toFixed(1)}, Helm=${r.V[helmIdx].toFixed(1)}`,
      body: 'Securitisation rhetoric, militarised metaphor in domestic policy.',
      confidence: 'HIGH', exemplar: '1975, 2010s, 2020s',
      trove: '"national security" + (domestic)',
    });
  }

  if (r.V[craftIdx] < 12) {
    signals['Fast-Loop Nodes (Group D)'].push({
      title: `Craft hollowing — V=${r.V[craftIdx].toFixed(1)}`,
      body: 'Manufacturing-nostalgia, "hollowing out" rhetoric, regional-grievance politics.',
      confidence: 'HIGH', exemplar: '1975–95',
      trove: '"manufacturing" + "decline"',
    });
  }

  if (r.V[handsIdx] < 9 && matrix[handsIdx][2] > 7) {
    signals['Fast-Loop Nodes (Group D)'].push({
      title: `Hands squeezed — V=${r.V[handsIdx].toFixed(1)}, S=${matrix[handsIdx][2]}`,
      body: '"Battler" populism, class-conflict framing, strike-wave coverage.',
      confidence: 'HIGH', exemplar: '1931, 1975, 1991',
      trove: '"battler", "strikes"',
    });
  }

  if (r.V[flowIdx] > 16) {
    signals['Fast-Loop Nodes (Group D)'].push({
      title: `Flow dominant — V=${r.V[flowIdx].toFixed(1)}`,
      body: 'Financialisation discourse, "the markets demand" register.',
      confidence: 'HIGH', exemplar: '1999–2007',
      trove: '"the markets" + "demand"',
    });
  }
  if (r.V[flowIdx] < 6) {
    signals['Fast-Loop Nodes (Group D)'].push({
      title: `Flow collapse — V=${r.V[flowIdx].toFixed(1)}`,
      body: '"Markets in panic", debt-personification, credit-rating-as-moral-verdict.',
      confidence: 'HIGH', exemplar: '1929–32, 1991, 2008',
      trove: '"credit rating", "markets"',
    });
  }

  // ═══ GROUP F — Pathologies ═══

  // Executive Decoupling
  if (r.V[helmIdx] < 6) {
    signals['Pathologies (Group F)'].push({
      title: `Executive Decoupling — V_Helm=${r.V[helmIdx].toFixed(1)}`,
      body: 'Personality-fixated coverage; "leader unable to govern"; cabinet-reshuffle obsession.',
      confidence: 'HIGH', exemplar: '1931, 2010s',
      trove: '"spill", "dismissed", "resigned"',
    });
  }

  // Praetorian Condition
  if (praetGap > 5 && r.V[shieldIdx] > r.V[helmIdx]) {
    signals['Pathologies (Group F)'].push({
      title: `Praetorian Condition — gap=${praetGap.toFixed(1)}`,
      body: 'Security-bracket expansion to non-security domains; militarised metaphor in domestic policy.',
      confidence: 'HIGH', exemplar: '1975, 2010s, 2020s',
      trove: '"national security" + (non-security domain)',
    });
  }

  // Mythic-Material Decoupling
  if (phantomGap > 5) {
    signals['Pathologies (Group F)'].push({
      title: `Mythic-Material Decoupling — gap=${phantomGap.toFixed(1)}`,
      body: 'Culture-war saturation displacing material grievance; ceremonial controversies on front pages.',
      confidence: 'HIGH', exemplar: '2014–22',
      trove: '"culture war" + "values"',
    });
  }

  // Cognitive Inversion
  const inversions = [];
  NODES.forEach((n, i) => {
    if (r.alpha[i] < -3 && r.gamma[i] > 20) inversions.push(n);
  });
  if (inversions.length > 0) {
    signals['Pathologies (Group F)'].push({
      title: `Cognitive Inversion in ${inversions.join(', ')}`,
      body: 'Weaponised competence: high-quality rhetoric oriented destructively; ressentiment register.',
      confidence: 'MED', exemplar: '1916 (Hughes), 2013–15 (Abbott)',
      trove: '(ressentiment + high-rhetoric)',
    });
  }

  // Sisu activation
  const sisuNodes = [];
  NODES.forEach((n, i) => { if (r.Sigma[i] > 4) sisuNodes.push({ name: n, val: r.Sigma[i] }); });
  if (sisuNodes.length >= 2) {
    signals['Pathologies (Group F)'].push({
      title: `Symbolic Persistence ("Sisu") — ${sisuNodes.map(s => `${s.name}: Σ=${s.val.toFixed(1)}`).join('; ')}`,
      body: 'Heroic-endurance discourse against material decline; "we will get through this".',
      confidence: 'HIGH', exemplar: '1942, 2020 (COVID)',
      trove: '"endure", "through this"',
    });
  }

  // Network Fragmentation
  if (meanB < 0.52 && r.Lambda > 1.9) {
    signals['Pathologies (Group F)'].push({
      title: `Network Fragmentation — B̄=${meanB.toFixed(3)}, Λ=${r.Lambda.toFixed(3)}`,
      body: 'Tribal/regional discourse simultaneous; press fragments by audience.',
      confidence: 'MED', exemplar: '1975–82, 2017–22',
      trove: '(discourse fragmentation)',
    });
  }

  // ═══ SECONDARY — lower priority / single-node flags ═══

  // Abstraction drift (Γ > 3) in non-Lore nodes
  const driftNodes = [];
  NODES.forEach((n, i) => { if (i !== loreIdx && r.Gamma[i] > 3) driftNodes.push({ name: n, val: r.Gamma[i] }); });
  if (driftNodes.length > 0) {
    signals['Secondary Phenomena'].push({
      title: `Abstraction drift in ${driftNodes.map(d => d.name).join(', ')}`,
      body: `Abstraction-to-coherence ratio exceeds 3. ${driftNodes.map(d => `${d.name}: Γ=${d.val.toFixed(2)}`).join('; ')}.`,
      confidence: 'MED', exemplar: '',
      trove: '',
    });
  }

  // Sisu in single node
  const sisuSingle = [];
  NODES.forEach((n, i) => { if (r.Sigma[i] > 0 && r.Sigma[i] <= 4) sisuSingle.push({ name: n, val: r.Sigma[i] }); });
  if (sisuSingle.length > 0 && sisuNodes.length < 2) {
    signals['Secondary Phenomena'].push({
      title: `Mild Sisu activation in ${sisuSingle.map(s => s.name).join(', ')}`,
      body: `Antifragile activation detected. ${sisuSingle.map(s => `${s.name}: Σ=${s.val.toFixed(1)}`).join('; ')}.`,
      confidence: 'MED', exemplar: '',
      trove: '',
    });
  }

  return signals;
}

/* ── Basin classification (v0.2 calibrated) ── */
function classifyBasin(r, matrix) {
  const praetGap = r.V[1] - r.V[0]; // Shield - Helm
  const meanB = r.B.reduce((a,b) => a+b, 0) / r.B.length;
  const epsilon = r.meanA * r.meanS / Math.max(r.meanK, 0.001);
  const phantomGap = (r.V[2] + r.V[6]) / 2 - (r.V[5] + r.V[4]) / 2;

  // Peak Expansion: meanV>15 ∧ σ_V<2.5 ∧ τ>1.5 ∧ Λ<1.3
  if (r.meanV > 15 && r.sigmaV < 2.5 && r.tau > 1.5 && r.Lambda < 1.3) return 'Peak Expansion';
  // Mass Formation: Helm low ∧ Shield high ∧ ε > 9
  if (r.V[0] < 12 && r.V[1] > 16 && epsilon > 9) return 'Mass Formation';
  // Praetorian: praet gap > +5 ∧ Helm < 12
  if (praetGap > 5 && r.V[0] < 12) return 'Praetorian';
  // Archipelago: meanB < 0.52 ∧ σ_V > 4 ∧ ε > 7
  if (meanB < 0.52 && r.sigmaV > 4 && epsilon > 7) return 'Archipelago';
  // Phantom Type I: Archive>20 ∧ Hands<9 ∧ Stewards high
  if (r.V[6] > 20 && r.V[5] < 9 && r.V[3] > 16) return 'Phantom I';
  // Phantom Type II: Archive declining ∧ Hands low ∧ ε > 9
  if (r.V[6] < 14 && r.V[5] < 9 && epsilon > 9) return 'Phantom II';
  // Early Strain: τ 1.1–1.3 ∧ ε rising
  if (r.tau >= 1.1 && r.tau <= 1.3 && epsilon > 6) return 'Early Strain';
  // Navigational: meanV 13–15 ∧ τ 1.3–1.5
  if (r.meanV >= 13 && r.meanV <= 15 && r.tau >= 1.3 && r.tau <= 1.5) return 'Navigational';
  // Navigational fallback for healthy-ish
  if (r.meanV > 12 && r.tau > 1.2) return 'Navigational';
  return 'Early Strain';
}

window.NODES = NODES;
window.generateSignals = generateSignals;
window.classifyBasin = classifyBasin;
