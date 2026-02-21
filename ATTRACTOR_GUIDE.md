# CAMS Phase-Space Attractor - Quick Reference Guide

**Version:** 1.0
**Date:** 2026-02-20

## Overview

The **Phase-Space Attractor** visualizes institutional dynamics as 3D trajectories through M-Y-B space, revealing stability regimes, transition dynamics, and attractor basins.

## What It Shows

### Coordinate System

**3D Phase Space:**
- **X-axis (M)**: Metabolic Load - mean stress across material subsystem (Hands, Flow, Shield)
- **Y-axis (Y)**: Mythic Integration - mean coherence+abstraction across meaning subsystem (Lore, Archive, Stewards)
- **Z-axis (B)**: Bond Strength - mean coupling strength between institutions

**Trajectory**: Each point represents the system state in a given year. The line traces institutional evolution through time.

### Key Visualizations

#### 1. 3D Phase-Space Trajectory
- **Interactive 3D plot** with Plotly (zoom, rotate, pan)
- **Color gradient**: Time progression (dark → light)
- **Start point**: Green diamond
- **End point**: Red diamond
- **Hover**: Shows Year, M, Y, B values

**What to look for:**
- **Tight spirals**: System oscillating around attractor
- **Linear paths**: Directional change (reform, collapse)
- **Loops**: Cyclical dynamics
- **Jumps**: Shocks or rapid transitions

#### 2. Phase Density Projection (M-Y plane)
- **2D heatmap** showing time spent in each region
- **Hot spots**: Attractor basins (stable regimes)
- **Sparse areas**: Transition zones

**What to look for:**
- **Single hot spot**: Stable equilibrium
- **Multiple hot spots**: Multiple regimes
- **Elongated densities**: Transitional dynamics
- **Scattered density**: Chaotic or unstable system

#### 3. Regime Detection (K-means clustering)
- **Clusters** the trajectory into discrete regimes
- **Shows regime centers** (M, Y, B coordinates)
- **Occupancy statistics**: % of time in each regime

**What to look for:**
- **Regime 1 (high M, low Y)**: Metabolic stress dominance (crisis, mobilization)
- **Regime 2 (low M, high Y)**: Mythic integration dominance (stability, legitimacy)
- **Regime 3 (medium M, medium Y)**: Balanced or transitional
- **Transitions between regimes**: Institutional phase changes

#### 4. Phase Velocity (Rate of Change)
- **Time series** of ||dM/dt, dY/dt, dB/dt||
- **Measures speed** of institutional change

**What to look for:**
- **Low velocity**: Stability, institutional inertia
- **High velocity spikes**: Shocks, wars, revolutions, reforms
- **Gradual acceleration**: Slow-building crisis
- **Sustained high velocity**: Chronic instability

## Usage

### Running the Dashboard

```bash
cd C:\Users\julie\wintermute
streamlit run cams_advanced_analysis.py
```

Browser opens at `http://localhost:8501`

### Workflow

1. **Load dataset** (sidebar)
   - Upload CSV or select from `data/cleaned/`

2. **Go to "Phase-Space Attractor" tab** (4th tab)

3. **Adjust smoothing** (sidebar):
   - σ=0: Raw data (noisy but shows all detail)
   - σ=2: Default (slight smoothing)
   - σ=5+: Heavy smoothing (long-term trends)

4. **Explore 3D trajectory**:
   - Rotate plot to see different angles
   - Hover over points for year/values
   - Look for loops, spirals, linear segments

5. **View density projection**:
   - Identify attractor basins (hot spots)
   - Check for multiple regimes

6. **Detect regimes**:
   - Select number of regimes (2-5)
   - Click "Detect Regimes"
   - Review cluster centers and occupancy
   - View regime-colored 3D plot

7. **Analyze velocity**:
   - Check trajectory statistics
   - Identify rapid transitions

## Interpretation Examples

### Example 1: Single Attractor Basin
```
Density Projection: One bright hot spot at (M=3, Y=6)
3D Trajectory: Tight spiral around center
Regime Detection: 90% of time in Regime 1

Interpretation: Stable equilibrium. System gravitates toward balanced state.
Policy implication: Institutional resilience, reforms will face inertia.
```

### Example 2: Regime Shift
```
3D Trajectory: Sharp transition from (M=2, Y=7, B=0.8) to (M=6, Y=3, B=0.4)
Velocity: Spike in 1968
Regime Detection: Regime 1 (1900-1967) → Regime 2 (1968-2025)

Interpretation: Sudden shift from mythic dominance to metabolic stress.
Historical context: Social upheaval, war, economic crisis.
Policy implication: System crossed tipping point, new equilibrium.
```

### Example 3: Oscillatory Dynamics
```
3D Trajectory: Regular loops with period ~15 years
Density Projection: Circular or elliptical pattern
Velocity: Periodic spikes

Interpretation: Cyclical boom-bust or reform-backlash pattern.
Historical context: Economic cycles, political pendulum swings.
Policy implication: System has endogenous oscillations, timing matters.
```

### Example 4: Chaotic/Unstable System
```
3D Trajectory: Erratic path, no clear pattern
Density Projection: Diffuse, no clear hot spots
Regime Detection: Frequent regime switches, low occupancy
Velocity: Sustained high values

Interpretation: Institutional instability, no stable equilibrium.
Historical context: Civil war, state failure, chronic crisis.
Policy implication: System lacks stabilizing mechanisms, fragile.
```

## Technical Details

### Field Computations

**Metabolic Load (M):**
```python
M = mean(Stress_Hands, Stress_Flow, Stress_Shield)
```

**Mythic Integration (Y):**
```python
Y = mean(
    0.5 * Coherence_Lore + 0.5 * Abstraction_Lore,
    0.5 * Coherence_Archive + 0.5 * Abstraction_Archive,
    0.5 * Coherence_Stewards + 0.5 * Abstraction_Stewards
)
```

**Bond Strength (B):**
```python
B = mean(Bond_Strength across all nodes)
```

### Smoothing

**Gaussian filter:**
```python
M_smooth[t] = Σ M[t'] * exp(-(t-t')²/(2σ²))
```

- σ=2: Standard deviation of ~2 years
- Removes high-frequency noise
- Preserves long-term trends

### Regime Detection

**K-means clustering in M-Y-B space:**
1. Normalize features (optional)
2. Initialize k random centroids
3. Assign each year to nearest centroid
4. Recompute centroids
5. Iterate until convergence

**Choosing k (number of regimes):**
- k=2: Simple binary (stable vs unstable)
- k=3: Three-state model (common in political economy)
- k=4+: Fine-grained regime typology

### Phase Velocity

**Euclidean norm of derivative:**
```python
velocity[t] = sqrt(dM[t]² + dY[t]² + dB[t]²)

where:
  dM[t] = M[t+1] - M[t]
  dY[t] = Y[t+1] - Y[t]
  dB[t] = B[t+1] - B[t]
```

## Comparison with Other Analyses

| Analysis | Focus | Output |
|----------|-------|--------|
| **dDIG** | Causal influence (which nodes drive change) | Influence rankings |
| **Dyad Field** | M-Y mismatch and coupling | Scalar time series |
| **Attractor** | **Holistic system trajectory** | **3D phase portrait** |

**When to use Attractor:**
- Understand overall system dynamics
- Identify stability regimes
- Detect phase transitions
- Compare societies' trajectories
- Assess institutional resilience

**When to use dDIG:**
- Identify influential institutions
- Understand causal mechanisms
- Prioritize policy interventions

**When to use Dyad Field:**
- Track metabolic-mythic balance
- Monitor stress-coherence coupling
- Predict crises (high D, low B)

## Advanced Usage

### Comparing Multiple Societies

1. Generate attractors for Society A and Society B
2. Export 3D plots as images
3. Compare:
   - Attractor basin locations (where do they stabilize?)
   - Trajectory shapes (spirals vs linear vs chaotic)
   - Regime structures (similar or different?)
   - Velocity profiles (stable vs volatile)

### Detecting Tipping Points

1. Plot velocity time series
2. Identify sharp spikes (rapid transitions)
3. Cross-reference with historical events
4. Check regime detection for before/after clustering
5. Policy implication: Early warning signals

### Forecasting

**Caution**: Phase-space attractors show past dynamics, not future guarantees.

**Limited forecasting:**
- If trajectory is tightly spiraling: Likely to remain in basin
- If trajectory is approaching regime boundary: Possible transition
- If velocity is accelerating: Instability building

**Not suitable for:**
- Long-term prediction (chaos, exogenous shocks)
- Quantitative forecasts (use dDIG or time-series models)

## Troubleshooting

### "Insufficient data for attractor visualization"
- Need at least 2 valid years with M, Y, B data
- Check that CAMS nodes match (Hands, Flow, Shield for M; Lore, Archive, Stewards for Y)
- Verify Bond Strength column exists

### Plot looks noisy/chaotic
- Increase smoothing (σ=5 or higher)
- Check data quality (missing values, outliers)
- Some societies genuinely have chaotic dynamics

### Regime detection fails
- Need at least k*10 data points for k regimes
- Reduce number of regimes
- Check for NaN values in M, Y, B

### 3D plot not interactive
- Ensure Plotly is installed: `pip install plotly`
- Check browser compatibility (use Chrome/Firefox)
- Export as HTML if needed

## API Reference

```python
from src.cams_attractor import (
    compute_fields_from_wide,
    smooth,
    plot_attractor_3d_plotly,
    plot_density_projection_plotly,
    detect_regimes
)

# Compute M, Y, B from wide-format dataframe
M, Y, B = compute_fields_from_wide(wide_df)

# Smooth time series
M_smooth = smooth(M, sigma=2)

# Generate 3D plot
fig_3d = plot_attractor_3d_plotly(M_smooth, Y_smooth, B_smooth, years)
fig_3d.show()

# Generate density projection
fig_density = plot_density_projection_plotly(M_smooth, Y_smooth)
fig_density.show()

# Detect regimes
labels, centers = detect_regimes(M_smooth, Y_smooth, B_smooth, n_regimes=3)
```

## References

### Theoretical Background
- Dynamical systems theory
- Attractor basins and stability
- Phase-space analysis
- Regime detection and clustering

### Related CAMS Analyses
- Dyad Field Analysis (M-Y mismatch)
- dDIG (institutional influence)
- Entropy dynamics (see CAMS v2.1 docs)

### Data Requirements
- CAMS long-format CSV with:
  - Year, Node, Coherence, Capacity, Stress, Abstraction, Bond Strength
  - Minimum 3 nodes from each subsystem (metabolic, mythic)
  - Minimum 2 years (preferably 20+ for meaningful patterns)

---

**Last Updated**: 2026-02-20
**Version**: 1.0
**Author**: Claude Sonnet 4.5 + Julie Bond
