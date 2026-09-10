# Event coding — Epiphenomenon horse-race externals

**Date:** 10 Sep 2026 (Sydney)  
**Coder:** single transparent coding (no second coder). Treat as exploratory.

Coding rule: count only **discrete, publicly documented** PRC–Australia coercive/military incidents, US alliance escalation milestones, and Australian domestic security announcements. Prefer official Defence/DFAT/Parliamentary sources; news used only to locate official statements.

---

## 1. PRC policy actions & military incidents (`prc_actions_count`)

| Year | Count | Event (short) | Citation |
|------|------:|---------------|----------|
| 2006–2008 | 0 | — | — |
| 2009 | 1 | Stern Hu / Rio Tinto case — major bilateral legal-diplomatic crisis | Contemporary DFAT/media record of Rio Tinto employee detentions (2009) |
| 2010–2015 | 0 | — | — |
| 2016 | 1 | South China Sea arbitral award period; sharp PRC–Australia diplomatic confrontation over SCS | PCA award 12 Jul 2016; Australian government statements supporting rules-based order |
| 2017 | 1 | Foreign-influence crisis peak (Dastyari resignation; interference debate preceding FITS) | Parliamentary / public record 2017 |
| 2018 | 2 | (a) Huawei excluded from 5G (Aug 2018); (b) Foreign Interference Transparency Scheme / espionage Act package — year of acute PRC reaction | ABC 23 Aug 2018; Commonwealth legislation 2018 |
| 2019 | 1 | Dalian port restrictions on Australian coal imports (Feb 2019) | Reuters commodities timeline |
| 2020 | 6 | Coercive trade package clusters: barley tariffs (May); beef suspensions; wine AD investigation/tariffs; informal restrictions on cotton, lobster, timber, coal | USSC “China’s trade restrictions on Australian exports”; UTS ACRI timeline PDF; Reuters timeline |
| 2021 | 1 | Sanctions maintained + diplomatic freeze (continuation coded as 1, not re-counting 2020 measures) | USSC; DFAT public record |
| 2022 | 1 | Unsafe PLA intercept of RAAF P-8 Poseidon, 26 May 2022 (chaff/flare) | [defence.gov.au release 5 Jun 2022](https://www.defence.gov.au/news-events/releases/2022-06-05/chinese-interception-p-8a-poseidon-26-may-2022) |
| 2023 | 0 | Thaw / partial lifts dominate; no new major coercive wave coded | USSC (barley tariff lift Aug 2023) |
| 2024 | 0 | Wine tariffs lifted 28 Mar 2024; thaw continues | USSC |
| 2025 | 2 | (a) Feb 2025 PLA flares near RAAF P-8; (b) Oct 2025 flares near RAAF P-8 | ABC Defence reporting 13 Feb 2025; 20 Oct 2025 |
| 2026 | MISSING | Partial year at coding date | — |

### Derived: `prc_sanctions_active` (binary)

- **1** for calendar years **2020–2023** while major PRC trade restrictions on Australian exports remained in force (USSC: restrictions from May–Nov 2020; wine lifted Mar 2024; barley Aug 2023 — 2023 still coded active for residual measures).
- **0** otherwise.
- This is a **contemporaneous regime dummy**, not a leading indicator.

---

## 2. US alliance pressure signals (`us_alliance_pressure`)

| Year | Count | Milestone | Citation |
|------|------:|-----------|----------|
| 2011 | 1 | US Marine Rotational Force – Darwin announced (Obama visit) | USSC Alliance at 70 / public record |
| 2014 | 1 | Australia–US Force Posture Agreement (AUSMIN 2014) | USSC; DFAT/Defence |
| 2017 | 1 | Quad revived (Manila / ASEAN Summit) | Public Quad statements 2017 |
| 2021 | 1 | **AUKUS** announced 15 Sep 2021 | White House / PMC transcripts; Wikipedia AUKUS (cross-check) |
| 2022 | 1 | ENNPIA / AUKUS implementation milestones | PMC AUKUS fact sheet (2022) |
| 2023 | 1 | AUKUS Optimal Pathway announced 13 Mar 2023 (San Diego) | CSIS / official trilateral announcement |
| 2024 | 1 | AUKUS enabling / SRF-West pathway progress (milestone year) | Public AUKUS implementation reporting |
| Other 2006–2025 | 0 | — | — |
| 2026 | MISSING | — | — |

---

## 3. Domestic political variables

### Election year dummy
Federal election years: **2007, 2010, 2013, 2016, 2019, 2022, 2025**  
Sources: APH / AustralianPolitics.com election dates.

### Labor vs Coalition (`labor_gov`)
| Years | Code | Government |
|------:|------|------------|
| 2006–2007 | 0 | Coalition (Howard; Labor from 3 Dec 2007 — 2007 coded Coalition majority) |
| 2008–2013 | 1 | Labor (Rudd / Gillard / Rudd) |
| 2014–2021 | 0 | Coalition (Abbott / Turnbull / Morrison) |
| 2022–2025 | 1 | Labor (Albanese from 23 May 2022; 2022 coded Labor) |

Source: List of Australian ministries (Wikipedia / APH Practice).

### Major security announcements (`security_announce_any` and component dummies)
| Year | Dummy columns | Document |
|------|---------------|----------|
| 2016 | `sec_dwp2016` | Defence White Paper 2016 |
| 2020 | `sec_su2020` | 2020 Defence Strategic Update |
| 2021 | `sec_aukus2021` | AUKUS announcement (also in US alliance) |
| 2023 | `sec_dsr2023` | Defence Strategic Review 2023 |

---

## 4. Trade exposure (`china_export_share_pct`)

- **2006–2023:** World Bank WITS *Export Partner Share (%)* Australia→China (`XPRT-PRTNR-SHR`), scraped from WITS Country Profile annual summaries (UNSD Comtrade underlying).
- **2024:** UN Comtrade public preview API `primaryValue` China / World = **30.11%**.
- **2025–2026:** **MISSING** (not retrieved at run time).
- Note: WITS includes a large “Unspecified” partner category in some years (esp. 2021–2022), which can depress China’s reported share relative to DFAT Composition of Trade figures (AMP cites goods-export peak ~42% in 2021; WITS 2021 = 34.15%). Series is internally consistent for horse-race use; do not mix with DFAT FY shares without reconciliation.

---

## Limits of coding
- Single coder; no second-coder κ.
- 2020 count=6 aggregates a sanctions *wave*; alternative codings (binary year=1) are available via `prc_sanctions_active`.
- Military incidents rely on **Australian Defence public releases** when available; unreported intercepts are invisible by construction.
