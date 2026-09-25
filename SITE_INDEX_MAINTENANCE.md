# Keeping the site indexes up to date

Three pages are **generated**, so don't edit them by hand:

| File | What it is | Built by |
|------|------------|----------|
| `sitemap.xml` | Search-engine index | `scripts/build_site_index.py` |
| `site-index.html` | **All Pages** list (neuralnations.org/site-index) | `scripts/build_site_index.py` |
| `nations.html` | **Reports by Nation** (neuralnations.org/nations) | `scripts/build_nations.py` |

## After adding, renaming or removing a page

1. Open `scripts/build_site_index.py`. Add the page's slug to the right group in `GROUPS`, or to `EXCLUDE` if it shouldn't be listed (tests, drafts, redirects, noindex pages).
   - A slug is the filename without `.html`. For example, `norway-arc-story`, `research/pr-gap2-findings`, or `cams-telescope/` for a folder's `index.html`.
2. Give the page a `<title>` and a `<meta name="description">`. Both indexes use them as the link text and summary.
3. Run both scripts from the repo root, **in this order**:
   ```
   python scripts/build_site_index.py
   python scripts/build_nations.py
   ```
4. Commit the regenerated files along with the new page.

If you forget step 1, `build_site_index.py` **stops with an error** that names the unlisted page. The problem shows up the next time either script runs, rather than a page quietly going missing.

## When to re-run the scripts (even with no new page)

- **A new country report or artefact** (a site page, or a new row in `explore-index.html`): re-run both scripts so it appears under its nation.
- **A new dataset** in `datasets.json` or a new nation card on `datasets.html`: re-run `build_nations.py`. If the society is new, add it to `NATIONS` (or `DATASET_ALIAS`) in `scripts/build_nations.py`. The script errors until you do.
- **A page's title or description changed:** re-run both scripts.
- **Monthly:** re-run both anyway, so the "updated" dates and the "Recently updated" list stay current.

## Datasets: every ensemble needs its envelope

Rule: every ensemble (`*_ENS*`) file in `cleaned_datasets/` or `data/nations/` has its envelope (`*_ENV*`), with the same name except ENS→ENV and exactly the same society/year/node rows. No ENV may exist without its ENS.

After adding, renaming or removing a dataset, run:
```
python scripts/check_envelopes.py
```
It rewrites `data/ENVELOPE_STATUS.md` and fails if an ENV has no ENS, if an ENV is empty or doesn't match its ENS row for row, or if an ENS has no ENV and isn't listed in `NO_ENVELOPE` with a reason. When a re-run supplies a missing envelope, add the file and delete its `NO_ENVELOPE` entry. Update `datasets.json` / `datasets.html` to match, then re-run `build_nations.py`.

## Nation matching

Reports are assigned to nations by the name patterns in `NATIONS` (`scripts/build_nations.py`). If a report lands under the wrong country, or is missing from the right one:
- to add a country: `PAGE_EXTRA = {'slug': {'Canada'}}`
- to remove a wrong match: `PAGE_SKIP = {'slug': {'China'}}`

Reports that name several countries are listed under each one and tagged *comparative*.

## Adding a link to the nav

The top nav is still hand-copied on about 135 pages. `site-chrome.js` drives it only on the few pages already migrated. A new nav item has to go in both places: edit `site-chrome.js`, then apply the same one-line insertion to every page's desktop and mobile nav. The "All Pages" link (commit b416b53) was added this way, as an example to follow.

The nav is close to full: it switches to the hamburger menu at 1340px and below, set in `style.css`. Check that a new item doesn't cause horizontal scrolling at about 1366–1440px.
