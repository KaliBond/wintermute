// ───── CAMS v3.2-R Compute Kernel ──────────────────────────────────────
// Canonical formulation April 2026. All operators per spec.

const NODES_COMPUTE = ['Helm','Shield','Lore','Stewards','Craft','Hands','Archive','Flow'];

function computeAll(matrix){
  // matrix[i] = [C, K, S, A] for each node

  // §2 — Core Scalar Operators
  // V_i = C + K - S + ½A
  const V = matrix.map(([c,k,s,a]) => c + k - s + 0.5*a);

  // ESCH Activation Index: σ_i = (A·C)·(K−S)
  const sigma = matrix.map(([c,k,s,a]) => (a * c) * (k - s));

  // Legacy operators (still useful for display)
  const alpha = matrix.map(([c,k,s,a]) => k - s);           // affect
  const mu    = matrix.map(([c,k,s,a]) => (c + a) / 2);     // structural mean
  const gamma = matrix.map(([c,k,s,a]) => a * c);            // interaction product
  const Gamma = matrix.map(([c,k,s,a]) => a / (c + 0.001)); // abstraction drift ratio
  const Sigma = matrix.map(([c,k,s,a]) => Math.max(a-7,0) * Math.max(s-6,0)); // Sisu

  // §2 — Gaussian RBF kernel W
  const dist = (x,y) => Math.sqrt(x.reduce((s,xi,i) => s + (xi-y[i])**2, 0));
  const Dmat = matrix.map(x => matrix.map(y => dist(x,y)));
  const flat = [];
  for(let i=0;i<matrix.length;i++) for(let j=i+1;j<matrix.length;j++) flat.push(Dmat[i][j]);
  flat.sort((a,b)=>a-b);
  const med = flat[Math.floor(flat.length/2)] || 1;
  const sw = Math.max(med, 0.5);
  const W = Dmat.map(row => row.map(d => Math.exp(-(d*d)/(2*sw*sw))));

  // B_i = (1/N) Σ_j W_ij  (per node bond strength)
  const N = matrix.length;
  const B = W.map((row,i) => {
    let s=0;
    for(let j=0;j<N;j++) { if(j!==i) s += row[j]; }
    return s / (N-1);
  });

  // B(t) = aggregate bond strength = 2/(N(N-1)) Σ_{i<j} W_ij
  let Bagg = 0;
  for(let i=0;i<N;i++) for(let j=i+1;j<N;j++) Bagg += W[i][j];
  Bagg = 2 * Bagg / (N * (N-1));

  // §3 — Coordination Laplacian L = D - W, algebraic connectivity λ₂
  // Degree matrix D_ii = Σ_j W_ij
  const degW = W.map(row => row.reduce((a,b) => a+b, 0));
  const L = W.map((row, i) => row.map((w, j) => (i === j ? degW[i] : 0) - w));

  const {eigs: Leigs, vecs: Lvecs} = jacobiEigen(L);
  // Sort eigenvalues ascending (Laplacian: smallest is 0, second-smallest is λ₂)
  const Lidx = Leigs.map((e,i) => [e,i]).sort((a,b) => a[0] - b[0]);
  const eigsSorted = Lidx.map(p => p[0]);
  const vecsSorted = Lidx.map(p => Lvecs.map(row => row[p[1]]));
  const lambda2 = eigsSorted[1] || 0;  // algebraic connectivity

  // Also compute W eigenvalues for spectral display
  const {eigs: Weigs, vecs: Wvecs} = jacobiEigen(W);
  const Widx = Weigs.map((e,i) => [e,i]).sort((a,b) => b[0] - a[0]);
  const WeignsSorted = Widx.map(p => p[0]);
  const WvecsSorted = Widx.map(p => Wvecs.map(row => row[p[1]]));

  // §3 — Master Stability Function R = λ_max / λ₂
  const lambdaMax = eigsSorted[eigsSorted.length - 1] || 1;
  const R = lambda2 > 0.0001 ? lambdaMax / lambda2 : Infinity;

  // §3 — Rate Dispersion ω(t) = std(ΔS_i/Δt)
  // Single snapshot: use S values as proxy for rate dispersion
  const Svals = matrix.map(r => r[2]);
  const meanS_local = Svals.reduce((a,b) => a+b, 0) / N;
  const omega = Math.sqrt(Svals.reduce((a,s) => a + (s - meanS_local)**2, 0) / N);

  // §3 — Structural Synchronisability χ_K = B(t) / ω(t)
  const chiK = omega > 0.001 ? Bagg / omega : Infinity;

  // §3 — Cognitive-Plane Criticality κ(t) = B(t) / ω_μ(t)
  const muVals = mu;
  const meanMu = muVals.reduce((a,b) => a+b, 0) / N;
  const omegaMu = Math.sqrt(muVals.reduce((a,m) => a + (m - meanMu)**2, 0) / N);
  const kappa = omegaMu > 0.001 ? Bagg / omegaMu : Infinity;

  // Kappa tier (v3.2-R calibrated)
  let kappaTier = 'NOMINAL';
  if (kappa >= 0.57) kappaTier = 'EXTREME';
  else if (kappa >= 0.42) kappaTier = 'CRITICAL';
  else if (kappa >= 0.35) kappaTier = 'WARNING';
  else if (kappa >= 0.30) kappaTier = 'WATCH';

  // §5 — Library Attractor η_loop = (BS_Lore × BS_Archive) / S_Hands
  const loreIdx = 2, archiveIdx = 6, handsIdx = 5;
  const S_Hands = matrix[handsIdx][2];
  const etaLoop = S_Hands > 0 ? (B[loreIdx] * B[archiveIdx]) / S_Hands : Infinity;

  // §5 — Headroom x(t) = log(B_eff) - 0.6·log(1+Ω_eff) - 0.2·log(1+D+)
  // B_eff = Bagg, Ω_eff = omega, D+ = max(0, meanS - meanK) as stress surplus
  const meanC = matrix.map(r=>r[0]).reduce((a,b)=>a+b,0)/N;
  const meanK = matrix.map(r=>r[1]).reduce((a,b)=>a+b,0)/N;
  const meanS = matrix.map(r=>r[2]).reduce((a,b)=>a+b,0)/N;
  const meanA = matrix.map(r=>r[3]).reduce((a,b)=>a+b,0)/N;
  const Dplus = Math.max(0, meanS - meanK);
  const headroom = Math.log(Math.max(Bagg, 0.001)) - 0.6 * Math.log(1 + omega) - 0.2 * Math.log(1 + Dplus);

  // x_min(t) = min_i [log(BS_i) - 0.6·log(1+ω_i) - 0.2·log(1+d_i)]
  const headroomMin = Math.min(...matrix.map((r, i) => {
    const di = Math.max(0, r[2] - r[1]); // node-level stress surplus
    const omegaI = Math.abs(r[2] - meanS_local); // node deviation from mean S
    return Math.log(Math.max(B[i], 0.001)) - 0.6 * Math.log(1 + omegaI) - 0.2 * Math.log(1 + di);
  }));

  // Aggregates
  const meanV = V.reduce((a,b)=>a+b,0)/N;
  const sigmaV = Math.sqrt(V.reduce((a,v)=>a+(v-meanV)**2,0)/N);
  const tau = meanK / Math.max(meanS, 0.001);

  // ε (overreach) = meanA · meanS / meanK  (v0.2 calibrated)
  const epsilon = meanA * meanS / Math.max(meanK, 0.001);

  // §4 — Phase space Φ(t) = (V̄, σ_V)
  // Crisis thresholds: V̄ < 12 and σ_V > 3.5

  // Bond pairs for display
  const pairs = [];
  for(let i=0;i<N;i++) for(let j=i+1;j<N;j++) pairs.push({i,j,w:W[i][j]});
  pairs.sort((a,b)=>b.w-a.w);

  // §7 — Disregard trigger: V̄ > 15 ∧ σ_V < 2.0
  const disregard = meanV > 15 && sigmaV < 2.0;

  // §6 — Attractor classification
  const helmV = V[0];
  let attractor;
  if (meanV < 0) attractor = 'Thermodynamic Freeze';
  else if (disregard) attractor = 'Buffering';
  else if (helmV < 6) attractor = 'Fracture';
  else if (sigmaV > 4.5 || (kappa >= 0.35 && helmV < 6)) attractor = 'Fracture';
  else if (kappa >= 0.35) attractor = 'Oscillation';
  else attractor = 'Re-synchronisation';

  // §8 — Composite alarm
  let alarmConfidence = 'NOMINAL';
  if (kappa >= 0.35 && helmV < 6) alarmConfidence = 'HIGH';
  else if (kappa >= 0.35) alarmConfidence = 'MEDIUM';
  else if (lambda2 < 0.5) alarmConfidence = 'LOW'; // λ₂ degrading

  // Legacy regime for backward compat
  let regime;
  if (helmV < 6) regime = 'EXECUTIVE_DECOUPLING';
  else if (disregard) regime = 'BUFFERING';
  else if (meanV > 12 && sigmaV < 3.5) regime = 'HEALTHY';
  else if (meanV < 8 || sigmaV > 4.5 || tau < 1.0) regime = 'CRITICAL';
  else regime = 'STRAINED';

  return {
    V, sigma, alpha, mu, gamma, Gamma, Sigma,
    B, Bagg, W,
    // Laplacian spectral
    eigs: eigsSorted, vecs: vecsSorted, lambda2,
    // W spectral
    Weigs: WeignsSorted, Wvecs: WvecsSorted,
    // Criticality
    R, omega, chiK, kappa, kappaTier,
    // Library Attractor & Headroom
    etaLoop, headroom, headroomMin,
    // Phase space
    meanV, sigmaV, meanC, meanK, meanS, meanA,
    tau, epsilon,
    // Pairs & classification
    pairs, disregard, attractor, alarmConfidence, regime,
    // Derived
    Lambda: lambda2,  // alias for backward compat
  };
}

function jacobiEigen(A){
  const n = A.length;
  const M = A.map(row => row.slice());
  const V = Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));
  for(let iter=0; iter<200; iter++){
    let p=0,q=1,max=0;
    for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){
      if(Math.abs(M[i][j])>max){max=Math.abs(M[i][j]);p=i;q=j;}
    }
    if (max < 1e-11) break;
    const Apq = M[p][q];
    const theta = (M[q][q]-M[p][p])/(2*Apq);
    let t;
    if (Math.abs(theta) > 1e15) t = 1/(2*theta);
    else t = (theta>=0?1:-1)/(Math.abs(theta)+Math.sqrt(theta*theta+1));
    const c = 1/Math.sqrt(1+t*t);
    const s = t*c;
    const App = M[p][p], Aqq = M[q][q];
    M[p][p] = App - t*Apq;
    M[q][q] = Aqq + t*Apq;
    M[p][q] = 0; M[q][p] = 0;
    for(let i=0;i<n;i++){
      if(i!==p && i!==q){
        const Aip = M[i][p], Aiq = M[i][q];
        M[i][p] = c*Aip - s*Aiq; M[p][i] = M[i][p];
        M[i][q] = s*Aip + c*Aiq; M[q][i] = M[i][q];
      }
      const Vip = V[i][p], Viq = V[i][q];
      V[i][p] = c*Vip - s*Viq;
      V[i][q] = s*Vip + c*Viq;
    }
  }
  return {eigs: M.map((row,i)=>row[i]), vecs: V};
}
