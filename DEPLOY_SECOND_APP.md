# Deploy CAMS GT as Second Dashboard

You now have TWO dashboards in your repository:

## Dashboard 1: CAMS GTS EV v2.0 (Existing)
- **File**: `streamlit_app.py`
- **URL**: https://wintermute-rgc7a5pgkcw8eg3luqinmq.streamlit.app
- **Features**:
  - EROEI-based thermodynamic engine
  - R ratio (Φ/Ψ) analysis
  - Model archaeology
  - Civilization type detection
  - 32 pre-loaded datasets

## Dashboard 2: CAMS GT Dual-Mode (New)
- **File**: `cams_gt_app.py`
- **URL**: (To be deployed)
- **Features**:
  - Upload CSV OR select pre-loaded datasets
  - Interactive network analysis
  - Enhanced time series
  - Phase classification (STABLE/STRESSED/FRAGILE/CRITICAL)
  - Network statistics

---

## Deploy Second Dashboard to Streamlit Cloud

### Step 1: Go to Streamlit Cloud
Navigate to: https://share.streamlit.io/

### Step 2: Click "New app"

### Step 3: Fill in Settings
- **Repository**: `KaliBond/wintermute`
- **Branch**: `main`
- **Main file path**: `cams_gt_app.py` ⚠️ (NOT streamlit_app.py)
- **App URL**: Choose a custom name like:
  - `cams-gt-analyzer`
  - `cams-gt-dualmode`
  - `neuralnations-cams-gt`

### Step 4: Click "Deploy!"
Wait 2-5 minutes for deployment

### Step 5: Get Your App URL
Once deployed, you'll get:
```
https://[your-chosen-name].streamlit.app
```

---

## Update Website to Link Both Dashboards

### Option A: Separate Landing Pages

**Create** `cams-gts-ev.html` for Dashboard 1
**Create** `cams-gt.html` for Dashboard 2

### Option B: Single Page with Two Buttons

Update `cams.html` with:
```html
<div class="dashboard-options">
  <div class="dashboard-card">
    <h3>CAMS GTS EV v2.0</h3>
    <p>Thermodynamic analysis with EROEI modeling</p>
    <a href="https://wintermute-rgc7a5pgkcw8eg3luqinmq.streamlit.app" class="btn">Open GTS EV Dashboard</a>
  </div>

  <div class="dashboard-card">
    <h3>CAMS GT Dual-Mode</h3>
    <p>Interactive network analysis with phase classification</p>
    <a href="https://[YOUR-NEW-APP].streamlit.app" class="btn">Open GT Dashboard</a>
  </div>
</div>
```

### Option C: Navigation Menu

Add to `index.html` navigation:
```html
<li><a href="cams.html">CAMS Dashboards</a>
  <ul>
    <li><a href="[URL1]">GTS EV v2.0</a></li>
    <li><a href="[URL2]">GT Dual-Mode</a></li>
  </ul>
</li>
```

---

## Comparison Table

| Feature | GTS EV v2.0 | GT Dual-Mode |
|---------|-------------|--------------|
| **Primary Metric** | R = Φ/Ψ | Ψ (Grand Metric) |
| **Energy Model** | EROEI-based | Thermodynamic |
| **Data Loading** | Pre-loaded only | Upload + Pre-loaded |
| **Network Viz** | Heatmap | Interactive Graph |
| **Phases** | 5 classes | 4 classes |
| **Best For** | EROEI research | General analysis |

---

## Maintenance

Both apps share the same:
- Data directory (`cleaned_datasets/`)
- Requirements file (`requirements.txt`)
- Git repository

Updates to datasets benefit both dashboards automatically!

---

## Quick Deploy Checklist

- [ ] Go to https://share.streamlit.io/
- [ ] Click "New app"
- [ ] Repository: `KaliBond/wintermute`
- [ ] Branch: `main`
- [ ] Main file: `cams_gt_app.py`
- [ ] Choose custom URL name
- [ ] Click "Deploy!"
- [ ] Wait 2-5 minutes
- [ ] Copy your new app URL
- [ ] Update website links
- [ ] Test both dashboards
- [ ] Share with colleagues!

---

**Both dashboards are now maintained in the same repository. Any git push updates both automatically!**
