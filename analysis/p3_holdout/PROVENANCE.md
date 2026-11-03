# P3d provenance
- The preregistration was committed to a local git repository before any P3d outcome was computed: commit `4199cc2e3fbe03305de7afb527c4beb850d74f54`, 2026-10-17T18:22:10+11:00. Results commit: `2421195`.
- `PREREG_P3d_17Oct2026.md` SHA-256: `94dc48621799ca812a96eaad95338d58e9b12a3975757fbdff11e4b27b2b4192`.
- `HOLDOUT_CAMS.csv` is not committed. It is `analysis/p3` FROZEN_CAMS (JUNO at 446928c) restricted to the societies in `holdout_societies.txt`, and `p3d_run.py` writes it. The script uses absolute box paths (`/workspace/analysis/p3`); point them at `analysis/p3/`.
