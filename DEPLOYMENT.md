# CAMS GTS EV v2.0 Dashboard - Deployment Guide

## Overview

This guide walks you through deploying the CAMS GTS EV Dashboard to Streamlit Cloud and integrating it with neuralnations.org.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free - sign up at https://streamlit.io/cloud)
- Access to neuralnations.org repository

## Step 1: Prepare GitHub Repository

### 1.1 Check Git Status

```bash
cd wintermute
git status
```

### 1.2 Add New Files

```bash
git add streamlit_app.py
git add cams_engine.py
git add .streamlit/config.toml
git add exports/.gitkeep
git add cams.html
git add DASHBOARD_README.md
git add DEPLOYMENT.md
```

### 1.3 Update Changed Files

```bash
git add requirements.txt
git add .gitignore
git add index.html
```

### 1.4 Commit Changes

```bash
git commit -m "Add CAMS GTS EV v2.0 Dashboard with Streamlit deployment

- Add streamlit_app.py with 3-tab interface (single society, multi-society, export)
- Add cams_engine.py with canonical v1.95 thermodynamic engine
- Add .streamlit/config.toml for production deployment
- Add cams.html landing page for neuralnations.org
- Update index.html with dashboard navigation
- Update requirements.txt for Streamlit Cloud
- Add comprehensive documentation (DASHBOARD_README.md, DEPLOYMENT.md)

Features:
- Year range filtering (vs single year)
- Bond strength heatmap visualization
- Multi-society comparison with health-normalized trajectories
- CSV/JSON export functionality
- Real-time thermodynamic controls (EROEI, entropy offload)"
```

### 1.5 Push to GitHub

```bash
git push origin main
```

If you get errors, you may need to:
```bash
git pull origin main --rebase
git push origin main
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up for Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" and use your GitHub account
3. Authorize Streamlit to access your repositories

### 2.2 Create New App

1. Click "New app" in Streamlit Cloud dashboard
2. Select settings:
   - **Repository**: `KaliBond/wintermute` (or your username)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom name like `cams-gts-ev` or `neural-nations-dashboard`

3. Click "Deploy!"

### 2.3 Wait for Deployment

- First deployment takes 2-5 minutes
- You'll see build logs in real-time
- Dashboard will auto-start when ready

### 2.4 Get Your App URL

Your dashboard will be available at:
```
https://[your-app-name].streamlit.app
```

Example:
```
https://cams-gts-ev.streamlit.app
```

## Step 3: Update neuralnations.org

### 3.1 Update cams.html with Dashboard URL

Edit `cams.html` line 207:

```javascript
const DASHBOARD_URL = "https://YOUR-APP-NAME.streamlit.app";
```

Replace `YOUR-APP-NAME` with your actual Streamlit app name.

### 3.2 Push Website Updates

```bash
git add cams.html
git commit -m "Update CAMS dashboard URL with Streamlit Cloud deployment"
git push origin main
```

### 3.3 Deploy to Netlify

If you're using Netlify (which it looks like you are based on `_redirects` file):

1. Netlify will auto-deploy when you push to main
2. Check deployment at https://app.netlify.com
3. Your site should update within 1-2 minutes

## Step 4: Verify Deployment

### 4.1 Test Dashboard

1. Go to your Streamlit app URL
2. Verify:
   - ✅ Data loads correctly
   - ✅ Sidebar controls work
   - ✅ Time-series graphs display
   - ✅ Bond strength heatmap shows
   - ✅ Multi-society comparison works
   - ✅ CSV/JSON exports download

### 4.2 Test Website Integration

1. Go to https://neuralnations.org
2. Click "CAMS GTS EV Dashboard" button
3. Verify it redirects to `cams.html`
4. Click "Enter the Dashboard" button
5. Verify it opens your Streamlit app

## Step 5: Optional - Custom Domain

If you want `dashboard.neuralnations.org`:

### 5.1 In Streamlit Cloud

1. Go to app settings
2. Under "General" → "Custom subdomain"
3. Follow instructions to add CNAME record

### 5.2 Update DNS

Add CNAME record in your domain registrar:
```
dashboard.neuralnations.org → [your-app].streamlit.app
```

## Troubleshooting

### Data Files Missing

If you see "No data files found":

1. Ensure `cleaned_datasets/` folder is in repository
2. Add sample data files:
```bash
git add cleaned_datasets/*.csv
git commit -m "Add CAMS datasets for dashboard"
git push origin main
```

3. Streamlit Cloud will auto-redeploy

### Build Fails on Streamlit Cloud

Check requirements.txt versions:
- Streamlit Cloud uses Python 3.9+ by default
- All packages should install without conflicts

If issues persist:
```bash
# Test locally first
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Large CSV Files

If dataset files are very large (>100MB):

**Option A**: Use Git LFS
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git commit -m "Add Git LFS for large CSV files"
```

**Option B**: Host CSVs externally
- Upload to Google Drive/Dropbox
- Update `streamlit_app.py` to fetch from URL
- Use `@st.cache_data` to cache downloads

### Dashboard Runs Slowly

If dashboard is slow with large datasets:

1. Add caching to data loading (already implemented)
2. Reduce year range in sidebar
3. Sample data for visualization (every Nth year)

## Monitoring & Updates

### View Logs

In Streamlit Cloud:
1. Go to your app dashboard
2. Click "Manage app" → "Logs"
3. See real-time application logs

### Update Dashboard

Any push to GitHub main branch will:
1. Trigger auto-redeploy on Streamlit Cloud
2. Update live within 2-3 minutes
3. Keep existing URL (no downtime)

### Analytics

Streamlit Cloud provides:
- Viewer count
- Active sessions
- Resource usage
- Error tracking

## Next Steps

### Short Term (Week 1)
- [ ] Deploy to Streamlit Cloud
- [ ] Test all features live
- [ ] Share link on social media
- [ ] Collect user feedback

### Medium Term (Month 1)
- [ ] Add more societies to dataset
- [ ] Implement suggested comparisons
- [ ] Add scenario forecasting tab
- [ ] Create PDF report export

### Long Term (Quarter 1)
- [ ] Public API deployment
- [ ] Automated data updates
- [ ] Machine learning predictions
- [ ] Academic paper integration

## Support

**Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
**Deployment Forum**: https://discuss.streamlit.io

**Project Contact**: kari.freyr.4@gmail.com

---

**Dashboard Version**: CAMS GTS EV v2.0
**Last Updated**: December 2025
**Author**: Kari McKern
