# CAMS Protocol Note — China national + Evergrande, Huawei, BYD, CATL, Vanke, Longfor, SMIC, 2010–2020

Protocol: CAMS Universal Instruction Set v1.4
Scorer: Kimi (Moonshot AI)
Run date: 2026-11-04
Years: 2010–2020 (condition at 31 December each year)
Outputs: `CAMS_ChinaSet_2010-2020.csv` (704 rows) and sibling `CAMS_ChinaSet_2010-2020_status.csv`
Companion generator: `make_cams_chinaset.py` (score matrices, fully reproducible)

General stance notes, applying to all eight entities:

- Blind rules applied: yes — prestige, labels and corporate/state titles ignored; each function scored
  as if the other seven did not exist; no cross-node reconciliation. For firms, the eight functions are
  read at the scale the firm itself operates (e.g. Shield = protection/coercion available to the firm —
  state backing, legal defence, regulatory shelter — not armed force; Flow = circulation of goods,
  components, capital, signals through the firm and its networks).
- Evidence basis: no pre-assembled packets existed; "the packet" is what web searches supplied per
  function and era, cited below with years. Evidence was examined at era level for the corporates and
  near-year level for China national; Score_Status cells marked `Carried_Forward` had no year-specific
  evidence examined this run and must not be treated as independent observations.
- Hostile-witness rule: financial tokens (revenue, sales, debt, subsidy, valuation) were used as anchors
  only with corroboration by organisational/physical/behavioural facts, or are listed as uncorroborated below.
- Later consequences (e.g. Evergrande's 2021 collapse, 2024 CSRC fraud findings, 2023 SMIC 7nm) were
  NOT back-cast into 2010–2020 scores. Where a pre-2020 fact was later documented (e.g. CSRC findings
  covering conduct in 2019–2020), it is used only as corroboration, never as a year-end condition.
- Known weakness: corporate Shield (protection) for private firms rests on thin, partly inferential
  evidence; scores there are deliberately conservative (4–6 band) and flagged where thin.

---

## 1. Entity: China (national)

Year(s): 2010–2020
Evidence basis (searches used): Xi Tian Yang, *The Political Economy of China's Anti-corruption Campaign*
(2021) — Hu-era fragmentation, punitive model under Xi; USCC testimony (Morgan, 2020) and SWP Berlin (2025)
and CRS R46808 (2021) — PLA reforms and modernisation; Chen, He & Liu, NBER w23598 / BFI (2017) — LGFV debt
6T→27T RMB 2008–2016, shadow banking; Morgan Stanley IM, *The Big Picture* — debt/GDP 203%→296% (2009–2017),
Lewis turning point, MIC2025; IW Köln (2025) — LGFV mechanics; NYU Stern Acharya et al. — credit 125%→240% of
GDP 2008–2013; EconStor/DIW (2015–16) — slowdown, Caixin PMI 47.0 (Sep 2015), official-GDP doubts; UNCRD HSR
case and MDPI Sustainability (2025) and Wikipedia *High-speed rail in China* — technology transfer, indigenous
capability, network build-out; AP/China Daily/ProQuest (2011–2013) — Wenzhou crash, 40 dead, design flaws,
54 blamed; Qiushi timeline (2020) and China Daily (2018) — targeted poverty registry 2014 (89.62M persons,
per-household database), 10M/yr pledge; AQLI/EPIC and Zou et al. — Air Quality Action Plan 2013, war on
pollution 2014, $270B, PM2.5 −41% (2013–2018), steel capacity cuts 115Mt (2016–17); World Bank *Urban China*
— grain self-sufficiency 92% (2010), 88% (2012), soybean import dependence; MERICS *Shaky China* (2023) —
state grain reserves scale; AP Carrie Lam timeline and CRS IF11295 — HK 2019; Britannica and Fajgelbaum et al.
and Lau/CDF (2020) — trade war tariffs (~$350B), Phase One (Jan 2020), 57% fulfilment, COVID rebound and
"lowest point since 1971" tension.

Era statements per function (one sentence per era; year-level notes at every score change):

- Helm. 2010–2012: Hu's second term governed through fragmented, faction-riven coordination that still
  delivered growth, until the Bo Xilai affair exposed open elite conflict in 2012 (C5→C4 at 2012). 2013–2016:
  Xi centralised agenda-setting (reform plan, leading small groups) but the 2015 stock crash and capital
  flight showed coordination limits (S→8 in 2015 on PMI 47.0 and credit-fuelled volatility). 2017–2019:
  "core" status and constitutional entrenchment made decisions bind; 2019 added HK and trade-war pressure
  (S→8 in 2019, AP/CRS HK timeline; Britannica tariff waves). 2020: COVID produced an initially reflexive,
  information-suppressing response that converted into massive coordinated containment by mid-year
  (C 8→7, K 7→8, S 9; Britannica on early-handling criticism and swift containment).
- Shield. 2010–2012: modernisation underway but the force was untested and internally compromised
  (USCC 2020 describes the long-term program). 2013–2016: island-building and the 2015–16 command
  reorganisation (four general departments abolished; five theatre commands; SSF) were executed as planned
  (C→7, K→7, A→6 at 2015–16; CRS 2021, SWP 2025). 2017–2019: first indigenous carrier, Djibouti base,
  expanded nuclear/conventional systems fielded (USCC 2020). 2020: Galwan clash and Taiwan pressure raised
  load (S→8).
- Lore. 2010–2012: growth-based legitimacy frayed; 2012 anti-Japan mobilisation showed the symbols still
  moved people but reactively. 2013–2016: the Chinese Dream/rejuvenation narrative consolidated, with the
  2015 civil-society crackdown as strain (S→7 in 2015). 2017–2019: Xi Thought written into constitution and
  party charter (C/K→7); 2019 HK rupture tested the symbolic order's reach (S→8, AP/CRS). 2020: COVID
  victory narrative sustained binding power under siege conditions.
- Stewards. 2010–2012: grain self-sufficiency slipped 92%→88% while social safety nets expanded and property
  boomed (World Bank; xitianyang 2021). 2013–2016: targeted-poverty registry tied allocation to per-person
  records; 2014 war on pollution declared, $270B committed, capacity cut (AQLI; Zou; China Daily). 2017–2019:
  deleveraging campaign and pollution-plan targets met (PM2.5 −41% by 2018; steel −115Mt 2016–17) raised
  capacity and abstraction (K→7, A→6 at 2017). 2020: pandemic strained stores and dependents, with food-security
  emphasis and ASF recovery (S→8; Britannica COVID strain).
- Craft. 2010: HSR build-out at scale with 380 km/h indigenous designs (UNCRD). 2011: Wenzhou collision —
  40 dead, signalling design flaws, 54 officials blamed — exposed technique subordinated to speed (C→5, K→6,
  S→7; AP/China Daily). 2012: recovery and renewed build-out. 2013–2016: Tianhe-2, indigenous train mastery,
  MIC2025 as governing plan (A→6). 2017–2019: Beidou, 5G, Chang'e 4; ZTE's 2018 near-death exposed chip
  dependency (S→7 2018, K→8 2019). 2020: Chang'e 5 sample return and Beidou completion (S→8 under sanctions).
- Hands. 2010–2012: ~250M+ migrant workers absorbed as the economy hit the Lewis turning point
  (Morgan Stanley 2021; NIFD via search). 2013–2016: labour input growth turned negative (2015, −0.9%),
  wages rose, the bond stretched but held (S→7 2013; NIFD). 2017–2019: manufacturing employment plateau,
  gig-economy absorption, automation push. 2020: lockdowns then resumption — rupture-level load on the
  work bond (S→9; Britannica on swift containment and reopening edge).
- Archive. 2010–2012: official statistics' reliability publicly doubted, local falsification incentives
  intact (C=4; EconStor 2015 on early release and smoothing doubts). 2013–2016: the 2014 poverty registry
  (89.62M persons identified, per-person database) turned records into classification (C→6 2014; Qiushi
  timeline); social-credit and big-data integration progressed. 2017–2019: integrated records and
  surveillance classifications scaled (A→6 2017). 2020: tracing and census systems under pandemic load (S→7).
- Flow. 2010–2012: large trade surpluses with managed RMB. 2013–2014: BRI and stock connect widened
  circuits (A→6). 2015: capital flight, reserve drawdown, equity crash — circulation of value under acute
  strain (C→5, S→8; EconStor on 2015 slump and doubts; NBER/BFI on credit overhang). 2016: controls bit,
  RMB joined SDR. 2017–2019: trade war tariffs on ~$350B of goods; exports rerouted and held (C→7; Britannica;
  Fajgelbaum). 2020: COVID export surge to near-record surplus (K→8; Britannica December 2020 rebound).

Fact-anchor rule (scores <5, >8, or moves ≥2, paired with supporting facts):
- Helm C4 (2012): Hu-term "power fragmentation and factional conflicts"; Bo Xilai scandal (xitianyang 2021).
- Helm S8 (2015): Caixin PMI 47.0 Sep 2015, slowest growth since 2008, credit-fuelled volatility (EconStor;
  NBER/BFI 2017).
- Helm C7/K8/S9 (2020): early-outbreak handling criticism, then swift containment and factory reopening
  (Britannica 2025).
- Hands S9 (2020): nationwide lockdowns with 11M-person Wuhan lockdown, followed by mass resumption
  (Britannica).
- Archive C4 (2010–2012): official GDP data doubted (early release, insufficient capture of slowdown;
  EconStor 2015).
- Flow C5/S8 (2015): 2015 capital-flight/crash complex; PMI 47.0 (EconStor; Acharya et al. on credit surge).
- All other China cells are within 5–8 with moves ≤1 from anchored neighbours.

Hostile-witness anchors:
- GDP growth figures — used minimally and only alongside physical indicators (PMI, power/rail proxies cited in
  EconStor); the Li Keqiang-index critique is treated as standing.
- Poverty-lift totals 2013–2020 (98.99M; official) — UNCORROBORATED as standalone financial/statistical
  anchors; scores rest instead on the registry mechanism (Qiushi 2020, a process fact) and pre-2013
  corroboration (World Bank joint study on 1978–2013 reduction).
- Debt/GDP ratios (BIS via Morgan Stanley/NBER) — corroborated by LGFV mechanics (Chen, He & Liu 2017)
  and the 2015–16 volatility record (EconStor).
- Grain self-sufficiency ratios (World Bank 2012 study) — corroborated by reserve-scale reporting
  (MERICS 2023).

NA justifications: none.

---

## 2. Entity: Evergrande

Year(s): 2010–2020
Evidence basis: Caixin Global (2021) — end-2017 borrowings +37%, short-term +76%, RMB 130bn strategic
investors with buyback rights, backdoor-listing plan; Baidu Baike EN — 2011 industry-first operating metrics,
200+ projects in 120+ cities, 2010 football club purchase, 2015 diversification listings and Evergrande
Wealth registration, 2016 Shengjing Bank 17.28%; timeline.vg — PBOC systemic-risk flag (Nov 2018), Three
Red Lines (Aug 2020), Sept 2020 Guangdong-government letter, Nov 2020 listing-plan termination; ResearchGate
(2024) — debt above $100bn by 2013, perpetual bonds; Yahoo/Bloomberg (2017) — 2016 contracted-sales #1;
Futunn (2024) — CSRC 2019–2020 fraud findings used ONLY as post-period corroboration of the financing
structure's integrity, not as a condition at any year-end.

Era statements per function:

- Helm. 2010–2014: Xu's directive control executed a nationwide land-bank expansion that reliably bound the
  organisation (2011 industry-first metrics; Baike). 2015–2016: diversification into health/culture/finance
  split direction (C→5 2015). 2017–2019: the backdoor-listing gambit and buyback obligations subordinated
  corporate direction to a financing trick (S→7 2017; Caixin 2021). 2020: Three Red Lines confrontation, the
  September cash-crunch letter and November buyback trigger put the executive under rupture-level stress
  (K→5, S→9, A→3; timeline.vg).
- Shield. 2010–2016: political connectedness and state-bank funding acted as protection; 2017–2019: regulators
  moved from enablers to constraint (late-2016 financing controls; PBOC systemic flag Nov 2018, K→5); 2020:
  the state imposed the red lines — protection inverted (C→4, K→4, S→8).
- Lore. 2010–2014: the football club (bought 2010; AFC Champions League 2013) built unusual symbolic capital
  for a developer (K→7 2013). 2015–2019: diversification diluted the story (K→6 2015). 2020: WMP-investor and
  supplier protests drained it (S→7).
- Stewards. 2010–2014: borrowing against presales became a perpetual-rollover store (debt >$100bn by 2013;
  ResearchGate) — custody untethered from constraint (C=4, A=3 throughout). 2015–2016: WMPs, insurance funds and
  bank stakes made the store a shell system (C→3, K→4, A→2 2015; Baike). 2017–2019: buyback obligations and
  short-term debt +76% deepened the mismatch (S→8 2017; Caixin). 2020: red-line breach plus the group's own
  cash-shortage warning (S→9; timeline.vg).
- Craft. 2010–2019: mass standardised housing delivery was genuine, repeatable technique at industry-leading
  volume (K=6→7; A=5 from mid-decade as industrialised building standardised). 2020: cash drought stalled sites
  and supplier payments, degrading output (C→5, K→6, S→8; timeline.vg letter).
- Hands. 2010–2019: a vast migrant construction workforce and contractor network were mobilised nationally with
  the presale engine funding the bond (S→6 2016 on sales-record strain). 2020: unpaid suppliers and halted
  projects pressed the bond hard (K→6, S→8).
- Archive. 2010–2019: design libraries and multi-city replication show usable plans, but from 2015 records
  increasingly served an off-balance-sheet financing structure rather than husbandry (C held 5; Caixin 2021 on
  the buyback structure as "off-balance-sheet borrowing"). 2020: record-integrity strained by the restructuring
  scramble (C→4, A→3). THIN EVIDENCE: no direct sources on Evergrande's internal records function; scores are
  floor estimates from observable replication and financing-structure reporting.
- Flow. 2010–2016: presale cash plus perpetual/offshore/WMP channels moved money at will — #1 contracted
  sales 2016 (C→7; Bloomberg 2017). 2017–2019: buyback obligations turned financing flows into a treadmill
  (S→7). 2020: the September letter (cash shortage, systemic-risk warning), WMP redemptions and the November
  buyback trigger are rupture facts (C→5, K→5, S→9, A→3; timeline.vg).

Fact-anchor rule: all Evergrande cells at 3–4 or 9, and all ≥2-point moves, are paired above with Caixin,
Baike, timeline.vg, ResearchGate or Bloomberg facts.

Hostile-witness anchors:
- Debt and borrowing-growth figures (Caixin 2021; ResearchGate 2024) — corporate disclosures corroborated by
  the physical fact of the 2020 cash-shortage letter and subsequent restructuring; the RMB 130bn buyback
  structure corroborated by its disclosed trigger event (Nov 2020 termination).
- "2016 contracted sales #1" (Bloomberg/Yahoo 2017) — corroborated by CRIC-ranking reporting cited therein;
  ranking tokens are not used alone as Capacity anchors (per title/budget/fame rule), only as Flow evidence.

NA justifications: none.

---

## 3. Entity: Huawei

Year(s): 2010–2020
Evidence basis: Baidu Baike EN (Ren Zhengfei) — revenue 185.2→891.4bn RMB (2010–2020), R&D 40.8→131.7bn
(2014–2019), rotating CEO system (2011), full ESOP (Dec 2011), Meng Wanzhou arrest (Dec 2018), DOJ
indictment (Jan 2019), Entity List (16 May 2019), Flex withholding ~$700m of materials, Intel/Google/Microsoft/
Facebook cutoffs, HarmonyOS (Aug 2019), 2.01m-RMB talent plan, 48% R&D-department reduction, "state of war"
mobilisation, May 2020 TSMC/FDPR cutoff; INSEAD case (2020) — P9/Kirin 955 (2016), 79k R&D staff 45% (2015),
R&D 15.1% of sales, 769 patents licensed to Apple (2015), Samsung suits (2016), smartphone shares; Statista
(2019) and Counterpoint (Jan 2020) — 206m smartphones 2018, #2 global brand CY2019 despite sanctions, ~40% China
share; truenetlab (2026) — May 2020 FDPR expansion and TSMC cutoff.

Era statements per function:

- Helm. 2010–2017: rotating-CEO governance plus founder authority produced unusually coherent strategy
  (connectivity + devices), with methodical institutional learning (IBM/IFS lineage; C→7 2011). 2018–2020:
  the Meng arrest and Entity List converted governance into war footing without loss of command cohesion —
  decisions still bound (C→8 2019; Baike; truenetlab).
- Shield. 2010–2017: protection rested on being strategically valuable to the Chinese state — real but
  untested. 2018–2020: tested and found asymmetric — PRC support was visible but could not unblock US
  chokepoints; stress rupture-level (S→8 2018 on Meng/DOJ; S→9 2019–2020 on Entity List and FDPR).
- Lore. 2010–2016: wolf-culture and engineer identity bound staff; brand ascent (P9/Leica 2016) widened
  symbolic reach (K→7 2014–2016). 2017–2020: national-champion symbolism fused with siege narrative; internal
  and domestic-market binding strengthened under pressure (C→7, S→8 2019; Counterpoint on domestic share).
- Stewards. 2010–2013: ESOP alignment and unlisted status freed allocation from quarterly reflex. 2014–2019:
  R&D allocation rose relentlessly (14–15%+ of revenue; Baike; INSEAD) — husbandry tied to long-term
  constraint even under 2019–2020 shock (S→8).
- Craft. 2010–2015: telecom-equipment technique world-class; Kirin SoCs close the loop. 2016–2019: smartphone
  craft at premium tier, 5G standard-essential leadership (K→8 2016; A→7 2018). 2020: design craft intact;
  manufacturing access severed (that loss is scored under Flow/Shield, per blind rule).
- Hands. 2010–2017: large engineering workforce, R&D-heavy (79k/45% in 2015). 2018–2020: siege mobilisation
  — "legion" rallies, 2.01m-RMB talent offers, workforce held ~197k under sanctions (C→8 2019; Baike).
- Archive. 2010–2014: document-heavy engineering culture, codified management systems. 2015–2017: patent
  records became reusable bargaining knowledge (769 patents licensed to Apple 2015; suits against Samsung 2016;
  A→6→7). 2018–2020: knowledge codification continued under stress.
- Flow. 2010–2017: global supplier and carrier circuits functioned at scale (K→8 2016–17). 2018: allied
  market exclusions began biting. 2019: supplier rupture — Flex, Intel, Google, Microsoft cutoffs (C→5, K→6,
  S→9). 2020: FDPR/TSMC cutoff — component circulation severed, partially offset by domestic demand surge.

Fact-anchor rule: all cells ≥8 (Helm C 2019–20; Shield S 2018–20; Hands C 2019–20; Craft K 2016–20;
Flow K 2016–17, S 2019–20) and all ≥2 moves are paired with Baike/INSEAD/Counterpoint/truenetlab facts above.

Hostile-witness anchors:
- Revenue and R&D figures (Baike table, from annual reports) — corroborated by independent IDC/Counterpoint
  shipment data (206m units 2018; #2 CY2019) and the organisational fact of 79k R&D staff (INSEAD).
- Smartphone shipment rankings (Counterpoint/Statista) — corroborated by the physical market-presence fact of
  ~40% China share in 2019.

NA justifications: none.

---

## 4. Entity: BYD

Year(s): 2010–2020
Evidence basis: CKGSB (2025) — Buffett 2008 stake (10%, $230m), vertical-integration strain 2011–2017,
battery-supply opening from 2017; Wikipedia *BYD Auto* and Orennia — F3DM retail flop (100 units by 2011),
e6 (2010), Qin (2012), Tang (2015), Han (2020), 2012 profit −94%, 2015 top China EV brand, ~$1bn 2016
subsidies exceeding net profit, Rhodium estimate $4.3bn state support 2015–2020, 2017–2019 subsidy-cut
slowdown, 2019 net profit RMB 1.6bn, Wang's "survive" remark, Egger design hire (2016), K9 buses; Britannica
— model years, FinDreams creation (2020), Blade battery (2020); Paulson Institute (2017) — US HQ 2011,
Lancaster plants 2013, Long Beach bus contract 2013; Humans of EV — timeline synthesis.

Era statements per function:

- Helm. 2010–2016: Wang's founder authority executed a single-minded EV strategy through profit collapse
  (−94% in 2012, S→7 2011–12) and subsidy peaks — plan-governed direction (A=5). 2017–2019: subsidy withdrawal
  forced reflexive cost cutting ("survive") and partial de-integration (C→6, S→7; Wikipedia). 2020: Blade
  battery and Han re-centred direction on demonstrated method (C→7, K→7, A→6).
- Shield. 2010–2016: protection was subsidy flow plus the Buffett halo — thin but adequate. 2017–2020:
  subsidy protection was deliberately withdrawn and nothing replaced it (S→6; Wikipedia/CKGSB).
- Lore. 2010–2013: weak consumer brand; identity centred on Wang the engineer-craftsman. 2014–2019: K9 bus
  exports and Lancaster manufacturing added industrial prestige (K→6), but the passenger brand stayed mid-tier.
  2020: Blade battery lifted symbolic standing (C→6, A→5).
- Stewards. 2010–2012: vertical integration strained the store (2012 profit collapse, C=4, A=3). 2013–2016:
  subsidy rents distorted husbandry — 2016 subsidies exceeded net profit (S→7; Wikipedia/Rhodium). 2017–2019:
  forced discipline; integration eased, survival husbandry (C→6, A→5). 2020: allocation re-tied to real
  technical constraint via the Blade platform (K→7, A→6).
- Craft. 2010–2012: genuine battery mastery; automotive craft catching up. 2013–2016: bus/e-bus manufacturing
  at global standard (Lancaster 2013), #1 domestic EV brand 2015, design hire 2016 (K→7 2014; A→6 2015).
  2017–2019: LFP depth accumulating under subsidy pressure. 2020: Blade battery — cell-to-pack LFP
  synthesis — a real method advance (C→7, K→8, A→7).
- Hands. 2010–2019: a ~200k+ strong integrated workforce executed unusually wide process coverage in-house
  (K→8 2020 with automation). 2020: pandemic-year output held.
- Archive. 2010–2020: process knowledge kept largely in-house (moulds, wafers, cells made internally;
  CKGSB) — usable but weakly documented externally; modest steady scores. THIN EVIDENCE: no direct sources on
  BYD's records/knowledge-management function; scores are conservative floor estimates.
- Flow. 2010–2012: retail EV demand absent (F3DM ~100 units; e6 fleet-only). 2013–2016: fleet/export flows
  (Long Beach 2013), then #1 domestic EV sales 2015 (K→7). 2017–2019: subsidy cuts hit sales flow three
  consecutive years (S→7; 2019 net profit RMB 1.6bn). 2020: FinDreams opened component/battery flows to other
  makers (A→6; Britannica).

Fact-anchor rule: all cells at 4 or 8, and all ≥2 moves, are paired with the facts above (profit collapse,
subsidy figures, Blade battery, FinDreams, Lancaster, subsidy cuts).

Hostile-witness anchors:
- Subsidy figures (Wikipedia citing Rhodium and filings; ~$1bn 2016) — corroborated by the behavioural fact
  of Wang's public "survive" stance and three years of reported profit decline.
- Net-profit figures (2012 −94%; 2019 RMB 1.6bn) — corroborated by model-cycle and subsidy-phase-out reporting.

NA justifications: none.

---

## 5. Entity: CATL

Year(s): 2011–2020 (2010 scored NA — entity did not exist)
Evidence basis: Wikipedia *CATL* and Baidu Baike EN — 2011 founding as ATL EV-battery spin-off (Zeng Robin
team), BMW Brilliance partnership 2012, Yutong 2013, MIIT white list 2015, Brunp recycling acquisition 2015,
#3 global 2016, #1 global 2017 (11.84GWh), Valmet 22% stake 2017, Shenzhen ChiNext IPO June 2018, BMW €4bn
contract 2018, Arnstadt (Germany) plant 2018, JVs with SAIC 2017 / Dongfeng & GAC 2018 / Geely & FAW 2019,
1M-mile cell claim June 2020; Wuttke et al., ScienceDirect (2025) — 85% buyout of ATL's EV business 2011;
greathandshake — 2016 two-year exclusion of foreign battery makers from subsidy lists, >RMB 1.8bn subsidies
2018–2022 (partly post-period); LightSource — TDK divestment 2015.

Era statements per function:

- Helm. 2011–2014: founder-team direction from ATL lineage; BMW validation disciplined early execution.
  2015–2016: white-list policy plus falsification-based engineering culture gave coherent direction (C→7 2015).
  2017–2020: JV-and-global-plant execution showed decisions binding at scale (K→7 2017; IPO 2018; JVs 2019).
- Shield. 2011–2015: minimal protection; survival on customer validation. 2016: state policy excluded foreign
  rivals from subsidy lists for two years — protection as industrial policy (S→6; greathandshake). 2017–2020:
  scale itself became the moat.
- Lore. 2011–2016: strong internal engineering identity, near-zero public symbolic capital. 2017–2020:
  #1-global standing and IPO fame built national-champion aura (C→6 2017; K→6, A→5 2018).
- Stewards. 2011–2014: tight custody of a small store against heavy R&D demands. 2015–2016: recycling
  integration (Brunp) tied allocation to material constraint (C→6, A→5). 2017–2020: JV network and IPO
  capital allocated to capacity and global footprint (K→7 2017; A→6 2018; S→7 2019 on subsidy phase-out).
- Craft. 2011: ATL polymer-battery lineage as foundation. 2012–2014: BMW automotive-grade validation — the
  difficult known work done to a demanding external standard (C→6, A→6 2012). 2015–2017: NCM scale-up to
  #1 global 2017 (K→8, A→7 2017). 2018–2020: €4bn BMW order and 1M-mile durability claim sustain
  leading-position craft (claim corroborated by independent industry reporting; see hostile-witness note).
- Hands. 2011–2014: workforce ramp from startup. 2015–2020: gigascale lines staffed and replicated
  (K→7 2017–18 at 11.84GWh and post-IPO build-out).
- Archive. 2011–2015: ATL heritage records plus automotive-grade documentation discipline. 2016–2020:
  falsification-based testing method institutionalised; own process recipes became the reusable memory
  (A→6 2016; C/K→7 2018).
- Flow. 2011–2012: BMW flow gate opened. 2013: Yutong commercial-vehicle flow. 2015: white-list gate.
  2017–2018: IPO and Germany plant widened capital/goods circuits (C→7). 2019: subsidy phase-out pressed
  customer flows (C→6, S→7). 2020: demand flows held at #1 with JV lock-ins.

Fact-anchor rule: no scores <5 or >8; all ≥2 moves paired with facts above (BMW 2012, white list 2015,
foreign exclusion 2016, #1 ranking 2017, IPO 2018, JVs 2019).

Hostile-witness anchors:
- 1M-mile battery claim (June 2020, company announcement) — LIGHTLY CORROBORATED by third-party industry
  reporting; used only to support an unchanged 2020 Craft row, not as a sole anchor for any move.
- Subsidy receipts (>RMB 1.8bn 2018–2022, filings via greathandshake) — partly post-period; used as context,
  not as an in-period anchor.
- 11.84GWh / #1-global ranking (2017) — corroborated by multiple independent market-share series (Wikipedia
  citations); ranking used as Flow/Craft evidence with the physical capacity build-out as corroboration.

NA justifications:
- 2010, all nodes: CATL did not exist (founded 2011); neither a performance fact nor a strain fact exists for
  the function at that scale — NA per the NA rule.

---

## 6. Entity: Vanke

Year(s): 2010–2020
Evidence basis: Yahoo/Bloomberg (2017) — Wang Shi's exit, Shenzhen Metro as largest shareholder, 2016 sales-rank
cession; Futunn industry research (2026) — 2015 financials (sales RMB 261.5bn +20.7%, net debt ratio 19.3%,
sales collection >90%, ROE 18.09%), dispersed equity (China Resources 14.89%), battle chronology, CIRC's
10-year ban on Yao Zhenhua (Feb 2017), Shenzhen Metro 29.38% control (2017); Caixin (2019) — Baoneng peak
25.4% reduced below 5% by end-2019, nine AMPs levered 1:2, RMB 13.7bn realised profit; arXiv 2003.06019 and
51percent timeline — June 2016 board-removal requisition, Jan 2017 Shenzhen Metro purchase of China Resources'
15.31% for RMB 37.2bn, Dec 2016 termination of the placement plan; Frontiers (2022) — 2014 Ying An partner
plan (3.26%); Seafarer Funds (2017) — share suspension Dec 2015, A-shares halted into 2016, Evergrande's
Aug 2016 entry (14.07%).

Era statements per function:

- Helm. 2010–2014: founder-led professional management with dispersed equity produced disciplined,
  method-governed direction (A=5→6 2014 on the partner plan). 2015: Baoneng's stealth accumulation to largest
  shareholder made direction contested (C→4, S→7). 2016: open control war — board-removal requisition, blocked
  white knight — decisions did not bind (C→3, K→5, S→8; arXiv/51percent). 2017: regulator-enforced resolution
  and state white knight restored binding direction at the cost of founder displacement (C→5, S→6; Futunn,
  Caixin). 2018–2020: stable state-anchored governance (C→6).
- Shield. 2010–2014: with no controlling shareholder, protection was reputation and market position only.
  2015–2016: protection failed against a leveraged raider — until regulators themselves intervened
  (C→4, S→7→8; Futunn). 2017–2020: CSRC/CIRC action plus Shenzhen Metro as anchor shareholder made
  protection durable (C→6, K→6, S→4).
- Lore. 2010–2014: Wang Shi's transparency-and-rules culture was the industry's reference identity (K=6).
  2015–2016: the "barbarian at the gate" war put the culture itself on trial (C→5, S→7→8). 2017: founder
  exit ruptured the symbolic order (C=5, S=6). 2018–2020: professional culture reconsolidated under Yu Liang
  (C→6).
- Stewards. 2010–2014: the industry's most disciplined custody — low leverage, >90% cash collection
  (K=7). 2015–2016: the battle consumed management attention and threatened the credit profile, but low net
  debt (19.3%) cushioned operations through a record-sales year (C→6, S→6; Futunn, arXiv). 2017–2020:
  rail+property pivot; steady custody (A→6 2018).
- Craft. 2010–2020: standardised, industrialised development technique with prefab pioneering — steady
  capability, modest strain until 2020's red-line regime (S→6 2020).
- Hands. 2010–2015: national contractor ecosystem mobilised with low friction (S→5 2015). 2016: sales
  record through the feud showed the work bond held under elite conflict (S→6). 2017–2020: normal load.
- Archive. 2010–2014: disclosure tradition codified (partner plan 2014 as incentive precedent; A→6).
  2015–2016: decision records and disclosure became battlegrounds (C→5, S→6→7). 2017–2020: records
  stabilised with control (C→6).
- Flow. 2010–2014: the industry's cleanest sales/cash-flow circuit (K=7). 2015–2016: share suspension and
  the AMP-leverage attack choked equity circulation and threatened credit (C→5, S→8; Seafarer; Caixin).
  2017–2020: normalisation, then pandemic-era pressure (S→6 2020).

Fact-anchor rule: all cells at 3–4 or 8, and all ≥2 moves, are paired with facts above (board-removal
requisition, AMP 1:2 leverage, CIRC ban, Shenzhen Metro purchases, trading halt, 2015 financials).

Hostile-witness anchors:
- 2015 financials (sales, net debt 19.3%, collection >90%; Futunn) — filing-derived figures corroborated by
  the behavioural fact that Baoneng targeted Vanke precisely for these cash-flow qualities (Caixin; arXiv).
- Sales-rank tokens (2016 cession of #1; Bloomberg) — used only as Flow/Hands context, not as Capacity anchors.

NA justifications: none.

---

## 7. Entity: Longfor

Year(s): 2010–2020
Evidence basis: China.org.cn (2009/2010) — PwC-audited transparency adopted early, five cornerstone
investors (GIC, Temasek, Ping An, BOCIG, Hong Kong Land; $197.5m), Wang Shi's "not the largest, but
definitely the best" assessment, repeated customer-satisfaction #1 rankings; Baidu Baike EN (Wu Yajun) —
2011 sales RMB 38.2bn ranked #8, Aug 2011 CEO handover to Shao Mingxiao (Wu stays chair), Nov 2012 divorce
with 28% stake transfer and −4.2% single-day stock dip, 2016 "Platform + End" concept, 2017 sales >RMB 100bn,
Nov 2018 transfer of 43.98% to daughter with voting per Wu's instructions, 2019 Midea-inspired decision to
institutionalise toward professional managers; China Biz Atlas — 2010 contracted sales RMB 33.3bn (+81.5%),
national top-10; Mingtiandi (2022) — strongest financials among private developers, all-three-IG-ratings status
(quoted as post-period corroboration of in-period conservatism).

Era statements per function:

- Helm. 2010–2011: founder discipline with early professionalisation (CEO handover 2011 while chair retained).
  2012: divorce-driven control uncertainty pressed direction (C→6, S→5; Baike). 2013–2016: steady top-10
  expansion; "Platform + End" management digitisation (K→7, A=5 2016). 2017–2020: >RMB 100bn scale with
  unchanged conservatism; succession engineering (2018 stake transfer with voting instructions; 2019
  institutionalisation decision) — method governing the founder's own exit (A→6 2018).
- Shield. 2010–2016: cornerstone state-linked investors and funding-market credibility as modest
  protection. 2017–2020: balance-sheet reputation became a real moat while peers' protection collapsed
  (K→6 2017; S→5 2019–20).
- Lore. 2010–2020: "best, not largest" identity and satisfaction rankings bound customers and staff; low-key
  founder persona; gradual national-brand consolidation (K=6; A→5 2017).
- Stewards. 2010–2016: PwC-transparency conservatism — the sector's cleanest custody (K=6→7).
  2017–2020: >RMB 100bn sales held with green-line discipline; commercial/rental mix stabilised the store
  (A→6 2018; corroborated post-period by Mingtiandi's "strongest financials" assessment).
- Craft. 2010–2016: mid/high-end residential craft, commercial-property operations (Beicheng Tianjie
  lineage). 2017–2020: refined operations craft across residential + mall portfolios (C→7 2017; S→5 2018).
- Hands. 2010–2020: Chongqing-rooted workforce and contractor network scaled nationally with the
  satisfaction-first model; no ruptures in the bond (S→5 from 2017 on scale).
- Archive. 2010–2017: transparency culture and quality systems codified (PwC since early years).
  2018–2020: succession and institutionalisation decisions documented and codified (A→6 2018).
- Flow. 2010: post-IPO capital plus +81.5% sales flow. 2012: divorce share-transfer shock — single-day −4.2%
  (C→5, S→5 2012; Baike). 2013–2020: uninterrupted funding-market access and sales flows; scale + credibility
  widened circuits (C/K→7 2017).

Fact-anchor rule: no scores <5 or >8; the only ≥2 moves (Flow C 2010→2012, Helm C 2010→2012) are paired
with the divorce share-transfer facts (Baike).

Hostile-witness anchors:
- Sales figures (RMB 33.3bn 2010; 38.2bn 2011; >100bn 2017; Biz Atlas/Baike) — corroborated by sustained
  top-10 industry rankings and customer-satisfaction survey results (China.org.cn).
- Post-period IG-ratings quote (Mingtiandi 2022) — used only as corroboration of in-period balance-sheet
  policy; not a condition of any year-end.

NA justifications: none.

---

## 8. Entity: SMIC

Year(s): 2010–2020
Evidence basis: ResearchGate/CUHK Press chapter (2022) — state-capital injection, consistent profits after
2012, "lack of long-term commitment to catch-up" widening the TSMC gap, 1/10 of TSMC revenue by 2017, 28nm
ramp struggles, co-CEO appointment 2017 (Zhao Haijun + Liang Mong Song), 14nm FinFET mass production
end-2019 (~six years behind TSMC), HiSilicon as major domestic customer, military-end-user designation and
Entity List (Dec 2020), ASML sub-10nm equipment cutoff, Liang resignation-letter circulation Dec 2020;
Wikipedia (Zhang Rujing; SMIC) and Asianometry (2022) — 2003 TSMC suit, 2005 $175m settlement, 2009 $200m
plus 10% equity settlement, three-year industry ban for Zhang, 15,000 stolen TSMC documents (500,000 pages),
"copied line for line"; HandWiki — Sept/Oct 2020 BIS supplier letters.

Era statements per function:

- Helm. 2010–2016: post-founder churn with state shareholders and no sustained catch-up commitment —
  direction fragmented (C=4 throughout; ResearchGate 2022). 2017: insider+technocrat co-CEO pairing gave
  direction a technocratic spine (C→5). 2018–2019: 14nm push coordinated to mass production. 2020: Entity
  List plus the Liang letter's public circulation showed internal friction surfacing under siege
  (C→4, S→8; ResearchGate 2022).
- Shield. 2010–2016: the 2009 settlement left protection weak — TSMC held 10% and a say; state support partial
  (C=4; Asianometry). 2017–2019: national-fund backing and strategic value thickened protection (C→5).
  2020: US coercion designation stripped protection against the one actor that mattered (C→4, K→5, S→9;
  ResearchGate; HandWiki).
- Lore. 2010–2016: mission identity ("China's chip hope") real but unfocused. 2017–2019: as Huawei sanctions
  loomed, the mission narrative bound engineers and state funders (C→6; S→7 2019). 2020: siege plus the
  Liang rift cracked the internal story (C→5, A→3).
- Stewards. 2010–2011: losses and settlement obligations — chronic shortage (C=4, K=4, A=3). 2012–2016:
  consistent but thin profitability; allocation to mature-node survival (ResearchGate). 2017–2019:
  big-fund-era capex at 1/10 TSMC scale — heavy store demands (K→6, S→7). 2020: cutoff threatened the
  equipment pipeline (S→8).
- Craft. 2010–2016: solid mature-node execution but the leading-edge craft was absent — 28nm ramp
  struggled for years (K=5, A=4; ResearchGate). 2017–2019: Liang's arrival built genuine FinFET capability;
  14nm entered mass production end-2019, six years behind (K→6 2017; S→7 2018; A→5 2018). 2020: technique
  pipeline choked by equipment cutoff even as existing craft held (S→8).
- Hands. 2010–2016: disciplined fab workforce executing mature-node volume. 2017–2020: multi-site 12-inch
  ramp with ~18k staff — the bond held under strain (C→6, K→7 2017; S→7 2018).
- Archive. 2010–2014: the founding shortcut left a corrupted memory function — 15,000 stolen documents had to
  be surrendered and purged; know-how built on them was unusable (C=3, K=4, A=3; Asianometry; Wikipedia).
  2015–2017: rebuilt, genuinely-owned process records matured (C→4 2015; C→5 2017). 2018–2020: own 14nm
  recipes documented and reusable (K→6 2018; A→5 2018).
- Flow. 2010–2016: domestic fabless demand flowed, but leading-edge equipment circulation was always
  externally gated. 2017–2019: HiSilicon demand rose while equipment access tightened (C→6 2017; S→7 2018;
  ResearchGate). 2020: equipment/materials cutoff under the 10nm rule — rupture (C→4, K→5, S→9, A→3).

Fact-anchor rule: all cells at 3–4 or 9, and all ≥2 moves (e.g. Archive C 2014→2017 3→5; Flow C 2019→2020
6→4; Lore A 2020; Shield S 2018→2020), are paired with facts above (document-surrender terms, co-CEO change,
14nm timeline, Entity List, Liang letter, equipment rules).

Hostile-witness anchors:
- "1/10 of TSMC revenue" (2017; ResearchGate estimate) — corroborated by the physical gap fact that SMIC was
  still ramping 28nm while TSMC mass-produced 7nm.
- 14nm mass-production timing and six-year lag (ResearchGate; HandWiki corroboration) — corroborated by
  independent foundry-tracking reporting in the same sources.

NA justifications: none.

---

## Revision log

- v1.0, 2026-11-04: initial run, all eight entities, 2010–2020. 704 rows. No prior versions.
