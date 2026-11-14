# CAMS Universal Instruction Set v1.4 -- China national + 7 corporates, 2010-2020
# Generates the raw score CSV and the Score_Status sibling file.
# Scorer: Kimi (Moonshot AI), run date 2026-11-04.
# Node order: Helm, Shield, Lore, Stewards, Craft, Hands, Archive, Flow.
# Years 2010..2020 inclusive.

import csv, os

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
YEARS = list(range(2010, 2021))
NA = ("NA", "NA", "NA", "NA")

def series(vals):
    """vals: list of 11 (C,K,S,A) tuples aligned to YEARS."""
    assert len(vals) == 11, f"need 11 year values, got {len(vals)}"
    return dict(zip(YEARS, vals))

# ---------------------------------------------------------------- China (national)
CHINA = {
 "Helm": series([
    (5,6,6,4), (5,6,6,4), (4,6,6,4),   # 2010-2012 Hu 2nd term: fragmentation; Bo affair 2012
    (7,6,6,5), (7,6,6,5),               # 2013-2014 Xi consolidation, reform plan
    (6,6,8,5), (6,6,7,5),               # 2015 stock crash/outflows; 2016 supply-side turn
    (8,7,7,6), (8,7,7,6), (8,7,8,6),    # 2017-2019 core status; 2019 HK + trade war
    (7,8,9,5),                          # 2020 COVID: early reflexive cover-up, then massive coordinated containment
 ]),
 "Shield": series([
    (5,6,6,4), (5,6,6,4), (6,6,6,4),    # 2010-2012 modernisation underway; 2012 Liaoning/Scarborough assertiveness
    (6,6,6,5), (6,6,6,5),               # 2013-2014 island building begins
    (6,7,6,5), (7,7,7,6),               # 2015-16 reforms announced/executed (theatre commands, SSF)
    (7,7,7,6), (7,7,7,6), (7,7,7,6),    # 2017-2019 Djibouti base, carriers, systems fielded
    (7,7,8,6),                          # 2020 Galwan clash, Taiwan pressure, mobilisation tested by COVID
 ]),
 "Lore": series([
    (5,6,5,4), (5,6,5,4), (5,6,6,4),    # 2010-2012 growth-legitimacy eroding; 2012 anti-Japan mobilisation
    (6,6,6,4), (6,6,6,4),               # 2013-2014 Chinese Dream narrative
    (6,6,7,4), (6,6,6,4),               # 2015 civil-society crackdown; 2016 confidence campaigns
    (7,7,7,5), (7,7,7,5), (7,7,8,5),    # 2017-2019 Xi Thought in constitution; 2019 HK rupture
    (7,7,8,5),                          # 2020 COVID victory narrative, nationalism
 ]),
 "Stewards": series([
    (5,6,7,4), (5,6,7,4), (5,6,7,4),    # 2010-2012 grain self-suff. 92->88%, safety-net build-out, property boom
    (6,6,7,5), (6,6,7,5),               # 2013-2014 targeted poverty registry; war on pollution prep
    (6,6,7,5), (6,6,7,5),               # 2015-2016 poverty campaign + capacity cuts (steel 115Mt)
    (6,7,7,6), (6,7,7,6), (6,7,7,6),    # 2017-2019 deleveraging, pollution plan targets met
    (6,7,8,6),                          # 2020 COVID strain on stores/dependents; pork/food security push
 ]),
 "Craft": series([
    (6,7,6,5), (5,6,7,5), (6,7,6,5),    # 2010 build-out; 2011 Wenzhou crash (design flaws, 40 dead); 2012 recovery
    (6,7,6,6), (6,7,6,6),               # 2013-2014 Tianhe-2, indigenous train mastery, patent spillovers
    (6,7,6,6), (6,7,6,6),               # 2015-2016 MIC2025, Chang'e/quantum milestones
    (7,7,7,6), (7,7,7,7), (7,8,7,7),    # 2017-2019 Beidou/5G/ZTE-shock; 2020 Chang'e 5, Beidou complete
    (7,8,8,7),                          # 2020
 ]),
 "Hands": series([
    (6,7,6,4), (6,7,6,4), (6,7,6,4),    # 2010-2012 migrant labour absorption near Lewis turning point
    (6,6,7,4), (6,6,7,4),               # 2013-2014 labour shortage visible, wages rising
    (6,6,7,4), (6,6,7,5),               # 2015-2016 labour input growth -0.9% (2015); gig economy rising
    (6,6,7,5), (6,6,7,5), (6,6,7,5),    # 2017-2019 manufacturing employment plateau, services absorption
    (6,6,9,5),                          # 2020 lockdowns then resumption
 ]),
 "Archive": series([
    (4,5,5,4), (4,5,5,4), (4,5,5,4),    # 2010-2012 GDP-data reliability doubts; local falsification incentives
    (5,6,5,5),                          # 2013
    (6,6,5,5), (5,6,6,5), (5,6,6,5),    # 2014 poverty registry (89.62M identified, per-person DB); 2015-16 social credit rollout
    (6,6,6,6), (6,6,6,6), (6,6,6,6),    # 2017-2019 integrated records/surveillance classifications
    (6,6,7,6),                          # 2020 tracing/census data systems under pandemic load
 ]),
 "Flow": series([
    (6,7,6,5), (6,7,6,5), (6,7,6,5),    # 2010-2012 trade surpluses, RMB pressure, controls
    (6,7,6,6), (6,7,6,6),               # 2013-2014 BRI launched, stock connect
    (5,6,8,6),                          # 2015 capital flight, reserve drawdown, stock crash
    (6,6,7,6),                          # 2016 RMB SDR, outflow controls bite
    (7,7,7,6), (7,7,8,6), (7,7,8,6),    # 2017-2019 trade war: tariffs on $350B, rerouting, resilient exports
    (7,8,8,6),                          # 2020 COVID export surge, record surplus
 ]),
}

# ---------------------------------------------------------------- Evergrande
EVERGRANDE = {
 "Helm": series([
    (7,7,5,4), (7,7,5,4), (7,7,5,4),    # 2010-2012 Xu Jiayin's directive expansion executes (200+ projects, 120+ cities by 2011)
    (7,7,6,4), (7,7,6,4),               # 2013-2014 perpetual-bond-fuelled growth
    (5,6,6,4), (5,6,6,4),               # 2015-2016 diversification (health/culture/finance) blurs direction
    (5,6,7,4), (5,6,7,4), (5,6,7,4),    # 2017-2019 backdoor-listing gambit, buyback obligations
    (5,5,9,3),                          # 2020 Three Red Lines, cash-crunch letter, buyback trigger
 ]),
 "Shield": series([
    (6,6,4,4), (6,6,4,4), (6,6,4,4),    # 2010-2012 political connections + state banks as protection
    (6,6,5,4), (6,6,5,4),               # 2013-2014
    (6,6,5,4), (6,6,5,4),               # 2015-2016 (Shengjing Bank stake 2016 deepens state ties)
    (5,6,6,4),                          # 2017 regulators flag concerns (late-2016 controls)
    (5,5,6,4),                          # 2018 PBOC names it systemic-risk conglomerate (Nov 2018)
    (5,5,7,4),                          # 2019 protection thinning
    (4,4,8,4),                          # 2020 Three Red Lines: the state becomes constraint, not shield
 ]),
 "Lore": series([
    (6,6,4,4), (6,6,4,4), (6,6,4,4),    # 2010-2012 football club purchase (2010) builds brand
    (6,7,4,4), (6,7,4,4),               # 2013-2014 AFC Champions League 2013, league titles
    (6,6,5,4), (6,6,5,4),               # 2015-2016 diversification dilutes the single story
    (6,6,5,4), (6,6,5,4), (6,6,5,4),    # 2017-2019
    (6,6,7,4),                          # 2020 WMP investors/supplier protests erode symbolic capital
 ]),
 "Stewards": series([
    (4,5,6,3), (4,5,6,3),               # 2010-2011 heavy borrowing against presales
    (4,5,7,3),                          # 2012
    (4,5,7,3), (4,5,7,3),               # 2013-2014 perpetuals; debt above $100bn by 2013
    (3,4,7,2), (3,4,7,2),               # 2015-2016 store custody becomes shell game (WMP, insurance funds, bank stakes)
    (3,4,8,2), (3,4,8,2), (3,4,8,2),    # 2017-2019 buyback obligations + ST debt +76% (2017)
    (3,4,9,2),                          # 2020 Three Red Lines breached; cash-crunch letter
 ]),
 "Craft": series([
    (6,6,5,4),                          # 2010
    (6,7,5,4), (6,7,5,4), (6,7,5,4),    # 2011-2013 mass standardised delivery at scale
    (6,7,5,5), (6,7,5,5),               # 2014-2015 industrialised building methods
    (6,7,6,5), (6,7,6,5), (6,7,6,5), (6,7,6,5),  # 2016-2019
    (5,6,8,5),                          # 2020 projects stall, contractor arrears
 ]),
 "Hands": series([
    (6,7,5,4), (6,7,5,4), (6,7,5,4), (6,7,5,4),  # 2010-2013 huge construction workforce mobilised nationally
    (6,7,5,4),                          # 2014
    (6,7,5,4), (6,7,6,4),               # 2015-2016 sales #1 2016
    (6,7,6,4), (6,7,6,4), (6,7,6,4),    # 2017-2019
    (6,6,8,4),                          # 2020 unpaid suppliers, stalled sites
 ]),
 "Archive": series([
    (5,6,4,4), (5,6,4,4), (5,6,4,4), (5,6,4,4),  # 2010-2013 standardised designs replicated across cities (usable plans)
    (5,6,4,4), (5,6,4,4),               # 2014-2015
    (5,6,4,4), (5,6,4,4),               # 2016-2017 records serve the financing structure (off-BS arrangements, Caixin)
    (5,6,5,4), (5,6,5,4),               # 2018-2019
    (4,5,6,3),                          # 2020 records under strain of restructuring scramble
 ]),
 "Flow": series([
    (6,6,5,4),                          # 2010
    (6,7,5,4), (6,7,5,4),               # 2011-2012 presale cash engine + HK listing
    (6,7,6,4), (6,7,6,4),               # 2013-2014 perpetual bond market access
    (6,7,6,4),                          # 2015
    (7,7,6,4),                          # 2016 contracted sales #1 (overtakes Vanke)
    (6,7,7,4), (6,7,7,4), (6,7,7,4),    # 2017-2019 buyback obligations strain financing flows
    (5,5,9,3),                          # 2020 cash-crunch warning letter; WMP redemptions; buyback trigger
 ]),
}

# ---------------------------------------------------------------- Huawei
HUAWEI = {
 "Helm": series([
    (6,6,5,5),                          # 2010
    (7,6,5,5),                          # 2011 rotating CEO system + full ESOP
    (7,6,6,5), (7,6,6,5), (7,6,6,5),    # 2012-2014
    (7,7,6,6), (7,7,6,6), (7,7,6,6),    # 2015-2017 governance matures; revenue 521.6bn (2016)
    (7,7,8,6),                          # 2018 Meng Wanzhou arrest (Dec 2018)
    (8,7,9,6),                          # 2019 Entity List; "state of war" mobilisation, HQ coherence
    (8,7,9,6),                          # 2020 FDPR expansion, TSMC cutoff absorbed
 ]),
 "Shield": series([
    (5,6,4,4), (5,6,4,4), (5,6,4,4), (5,6,4,4),  # 2010-2013 no coercive arm; state backstop assumed
    (5,6,4,4), (5,6,4,4), (5,6,4,4), (5,6,4,4),  # 2014-2017
    (6,6,8,4),                          # 2018 Meng arrest + US DOJ indictment
    (6,6,9,4),                          # 2019 Entity List; PRC state support visible but cannot unblock US chokepoints
    (6,6,9,4),                          # 2020
 ]),
 "Lore": series([
    (6,6,5,4), (6,6,5,4), (6,6,5,4), (6,6,5,4),  # 2010-2013 wolf culture, engineer identity
    (6,7,5,5), (6,7,5,5), (6,7,5,5),    # 2014-2016 global brand ascent (P9/Leica 2016)
    (6,7,6,5), (6,7,6,5),               # 2017-2018 national champion symbolism grows
    (7,7,8,5),                          # 2019 siege rallies internal/external identity; domestic share ~40%
    (7,7,8,5),                          # 2020
 ]),
 "Stewards": series([
    (6,6,5,5), (6,6,5,5), (6,6,5,5), (6,6,5,5),  # 2010-2013 ESOP alignment, no listing pressure
    (6,7,6,6),                          # 2014 R&D 14.2% of revenue (40.8bn yuan)
    (7,7,6,6), (7,7,6,6), (7,7,6,6),    # 2015-2017 R&D 76-90bn/yr
    (7,7,7,6),                          # 2018
    (7,7,8,6),                          # 2019 sanctions force reallocation of stores to survival priorities
    (7,7,8,6),                          # 2020
 ]),
 "Craft": series([
    (6,7,5,5), (6,7,5,5), (6,7,5,5),    # 2010-2012 telecom-equipment technique world-class
    (6,7,5,6),                          # 2013
    (6,7,6,6),                          # 2014
    (7,7,6,6),                          # 2015 Kirin 950 class SoCs
    (7,8,6,6), (7,8,6,6),               # 2016-2017 P9/Kirin 955; 5G leadership position
    (7,8,7,7),                          # 2018 5G standard-essential patents lead
    (7,8,8,7),                          # 2019 design craft intact under sanctions
    (7,8,8,7),                          # 2020
 ]),
 "Hands": series([
    (6,6,5,4), (6,6,5,4), (6,6,5,4), (6,6,5,4),  # 2010-2013 large engineering workforce
    (6,7,5,5),                          # 2014
    (7,7,6,5),                          # 2015 79k R&D staff (45% of employees)
    (7,7,6,5), (7,7,6,5),               # 2016-2017
    (7,7,7,5),                          # 2018
    (8,7,8,6),                          # 2019 "legion" mobilisation; 2.01m-yuan talent plan
    (8,7,8,6),                          # 2020 ~197k employees hold under siege
 ]),
 "Archive": series([
    (6,6,5,5), (6,6,5,5), (6,6,5,5),    # 2010-2012 document-heavy engineering culture
    (6,6,5,6), (6,6,5,6),               # 2013-2014 management systems codified (IFS/IBM lineage)
    (7,7,6,6),                          # 2015 769 patents licensed to Apple (records become bargaining knowledge)
    (7,7,6,7),                          # 2016 Samsung suits assert patent precedent
    (7,7,6,7),                          # 2017
    (7,7,7,7), (7,7,7,7),              # 2018-2019
    (7,7,7,7),                          # 2020
 ]),
 "Flow": series([
    (6,7,5,5), (6,7,5,5), (6,7,5,5), (6,7,5,5),  # 2010-2013 global supply + sales circuits
    (6,7,5,6),                          # 2014
    (6,7,6,5),                          # 2015
    (6,8,6,6), (6,8,6,6),               # 2016-2017 carrier + device distribution global
    (6,7,7,6),                          # 2018 US market entry already blocked (2012); allies warned
    (5,6,9,6),                          # 2019 supplier rupture (Flex $700m, Intel, Google)
    (5,6,9,6),                          # 2020 TSMC/FDPR cutoff; domestic sales compensate
 ]),
}

# ---------------------------------------------------------------- BYD
BYD = {
 "Helm": series([
    (7,6,6,5),                          # 2010 Wang Chuanfu's EV bet executes (e6, F3DM)
    (7,6,7,5),                          # 2011 profit collapse (-94% 2012 reported for 2011-12) but strategy holds
    (7,6,7,5),                          # 2012
    (7,6,6,5), (7,6,6,5),               # 2013-2014 fleet/bus pivot
    (7,7,6,5), (7,7,6,5),               # 2015-2016 top China EV brand; subsidies peak
    (6,6,7,5), (6,6,7,5), (6,6,7,5),    # 2017-2019 subsidy withdrawal; "survive" mode
    (7,7,7,6),                          # 2020 Blade battery + Han; recentring
 ]),
 "Shield": series([
    (5,5,5,4), (5,5,5,4), (5,5,5,4), (5,5,5,4),  # 2010-2013 subsidy + Buffett halo as main protection
    (5,5,5,4),                          # 2014
    (5,5,5,4),                          # 2015
    (5,6,5,4),                          # 2016 ~$1bn subsidies
    (5,5,6,4), (5,5,6,4), (5,5,6,4),    # 2017-2019 subsidy protection withdrawn
    (5,5,6,4),                          # 2020
 ]),
 "Lore": series([
    (5,5,5,4), (5,5,5,4), (5,5,5,4), (5,5,5,4),  # 2010-2013 weak consumer brand
    (5,6,5,4),                          # 2014 K9 buses / Lancaster plant prestige
    (5,6,5,4),                          # 2015
    (5,6,5,4),                          # 2016
    (5,6,6,4), (5,6,6,4), (5,6,6,4),    # 2017-2019 design uplift (Egger) but brand still mid-tier
    (6,6,6,5),                          # 2020 Blade battery/Han raise symbolic standing
 ]),
 "Stewards": series([
    (5,5,6,4),                          # 2010 vertical integration strains balance sheet
    (5,5,6,4),                          # 2011
    (4,5,6,3),                          # 2012 profit -94%
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (5,6,6,4),                          # 2015
    (5,6,7,4),                          # 2016 subsidies exceed net profit (~$1bn vs <$1bn)
    (6,6,7,5), (6,6,7,5), (6,6,7,5),    # 2017-2019 forced husbandry, integration eased
    (6,7,6,6),                          # 2020 Blade platform reused across products
 ]),
 "Craft": series([
    (5,6,5,5), (5,6,5,5), (5,6,5,5),    # 2010-2012 battery mastery; auto craft catching up
    (6,6,5,5),                          # 2013 US bus/battery plant (Lancaster)
    (6,7,5,5),                          # 2014
    (6,7,5,6),                          # 2015 #1 China EV brand
    (6,7,5,6),                          # 2016 Egger design hire
    (6,7,6,6), (6,7,6,6), (6,7,6,6),    # 2017-2019 LFP depth accumulates
    (7,8,6,7),                          # 2020 Blade battery (cell-to-pack LFP) -- method advance
 ]),
 "Hands": series([
    (6,7,5,4), (6,7,5,4), (6,7,5,4), (6,7,5,4),  # 2010-2013 large integrated workforce
    (6,7,5,4),                          # 2014
    (6,7,6,4),                          # 2015
    (6,7,6,4),                          # 2016
    (6,7,6,5), (6,7,6,5), (6,7,6,5),    # 2017-2019 ~230k employees; automation push
    (6,8,6,5),                          # 2020
 ]),
 "Archive": series([
    (5,6,5,4), (5,6,5,4), (5,6,5,4), (5,6,5,4),  # 2010-2013 process knowledge in-house (moulds, wafers)
    (5,6,5,5), (5,6,5,5), (5,6,5,5),    # 2014-2016
    (6,6,5,5), (6,6,5,5), (6,6,5,5),    # 2017-2019
    (6,6,5,5),                          # 2020
 ]),
 "Flow": series([
    (5,5,6,4),                          # 2010 F3DM retail flops (100 units by 2011)
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (6,6,5,5),                          # 2013 US fleet contracts (Long Beach)
    (6,6,5,5),                          # 2014
    (6,7,5,5),                          # 2015 #1 EV sales
    (6,7,6,5),                          # 2016
    (6,6,7,5), (6,6,7,5), (6,6,7,5),    # 2017-2019 subsidy cuts hit sales flow
    (6,7,6,6),                          # 2020 FinDreams opens battery supply to industry
 ]),
}

# ---------------------------------------------------------------- CATL (founded 2011; 2010 = NA)
CATL = {
 "Helm": series([
    NA,                                 # 2010 entity nonexistent
    (6,5,5,5),                          # 2011 founding team spins out of ATL
    (6,5,5,5),                          # 2012 BMW Brilliance validation
    (6,6,5,5),                          # 2013
    (6,6,5,5),                          # 2014
    (7,6,5,5),                          # 2015 white list; Brunp recycling
    (7,6,6,5),                          # 2016 foreign makers excluded from subsidies (2 yrs)
    (7,7,6,6),                          # 2017 #1 global (11.84GWh); SAIC JV; Valmet 22%
    (7,7,6,6),                          # 2018 Shenzhen IPO; BMW EUR4bn contract
    (7,7,7,6),                          # 2019 JVs with Geely/FAW; subsidy phase-out strain
    (7,7,7,6),                          # 2020 1M-mile cell claim
 ]),
 "Shield": series([
    NA,
    (5,5,5,4), (5,5,5,4), (5,5,5,4), (5,5,5,4),  # 2011-2014
    (5,5,5,4),                          # 2015
    (5,5,6,4),                          # 2016 subsidy-list exclusion of foreign rivals
    (5,6,6,4),                          # 2017
    (5,6,6,5),                          # 2018
    (5,6,6,5),                          # 2019
    (5,6,6,5),                          # 2020
 ]),
 "Lore": series([
    NA,
    (5,5,5,4), (5,5,5,4), (5,5,5,4), (5,5,5,4), (5,5,5,4),  # 2011-2015 engineering culture, no public brand
    (5,5,5,4),                          # 2016
    (6,6,5,4),                          # 2017 #1 global standing
    (6,6,6,5),                          # 2018 IPO fame; national-champion aura
    (6,6,6,5),                          # 2019
    (6,6,6,5),                          # 2020
 ]),
 "Stewards": series([
    NA,
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (6,6,6,5),                          # 2015 recycling integration (Brunp)
    (6,6,6,5),                          # 2016
    (7,7,6,5),                          # 2017 Valmet stake, SAIC JV
    (7,7,6,6),                          # 2018 IPO capital
    (7,7,7,6),                          # 2019 JV network locks customers
    (7,7,7,6),                          # 2020
 ]),
 "Craft": series([
    NA,
    (5,6,6,5),                          # 2011 ATL polymer-battery lineage
    (6,6,6,6),                          # 2012 BMW automotive-grade validation
    (6,6,6,6),                          # 2013 Yutong commercial vehicles
    (6,6,6,6),                          # 2014
    (7,7,6,6),                          # 2015
    (7,7,6,6),                          # 2016 #3 global
    (7,8,7,7),                          # 2017 #1 global
    (7,8,7,7),                          # 2018 EUR4bn BMW order
    (7,8,7,7),                          # 2019
    (7,8,7,7),                          # 2020 1M-mile durability claim
 ]),
 "Hands": series([
    NA,
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (5,6,6,4),                          # 2013
    (6,6,6,4),                          # 2014
    (6,6,6,5),                          # 2015
    (6,6,6,5),                          # 2016
    (6,7,7,5),                          # 2017 capacity ramp to 11.84GWh
    (6,7,7,5),                          # 2018
    (6,7,7,5),                          # 2019
    (6,7,7,5),                          # 2020
 ]),
 "Archive": series([
    NA,
    (5,5,5,5),                          # 2011 ATL records/heritage transfer
    (5,6,5,5),                          # 2012
    (5,6,5,5),                          # 2013
    (5,6,5,5),                          # 2014
    (6,6,5,5),                          # 2015
    (6,6,5,6),                          # 2016 falsification-based engineering method institutionalised
    (6,6,6,6),                          # 2017
    (6,7,6,6),                          # 2018
    (6,7,6,6),                          # 2019
    (6,7,6,6),                          # 2020
 ]),
 "Flow": series([
    NA,
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (6,6,6,4),                          # 2013 Yutong contract
    (6,6,6,5),                          # 2014
    (6,6,6,5),                          # 2015 white list opens subsidy-flow gate
    (6,6,6,5),                          # 2016
    (7,7,6,6),                          # 2017
    (7,7,6,6),                          # 2018 IPO + Germany plant
    (6,7,7,6),                          # 2019 subsidy phase-out pressure
    (7,7,7,6),                          # 2020
 ]),
}

# ---------------------------------------------------------------- Vanke
VANKE = {
 "Helm": series([
    (7,7,4,5), (7,7,4,5), (7,7,4,5),    # 2010-2012 founder-led professional management, dispersed equity
    (7,7,4,5),                          # 2013
    (7,7,4,6),                          # 2014 partner incentive plan (Ying An 3.26%)
    (4,6,7,5),                          # 2015 Baoneng raid -> largest shareholder Dec 2015; trading halt
    (3,5,8,5),                          # 2016 board-removal requisition; white-knight plan blocked
    (5,6,6,5),                          # 2017 Shenzhen Metro control 29.38%; Wang Shi exits; Yu Liang chair
    (6,6,5,5),                          # 2018
    (6,6,5,5),                          # 2019
    (6,6,6,5),                          # 2020 stability under state anchor shareholder
 ]),
 "Shield": series([
    (5,5,4,4), (5,5,4,4), (5,5,4,4), (5,5,4,4), (5,5,4,4),  # 2010-2014 no controlling shareholder
    (4,4,7,4),                          # 2015 protection fails vs leveraged raider
    (4,4,8,4),                          # 2016 regulators watch; resolution uncertain
    (6,6,4,5),                          # 2017 CSRC/CIRC intervene on Vanke's side; state white knight
    (6,6,4,5),                          # 2018
    (6,6,4,5),                          # 2019
    (6,6,5,5),                          # 2020
 ]),
 "Lore": series([
    (7,6,4,5), (7,6,4,5), (7,6,4,5),    # 2010-2012 Wang Shi culture: rules, transparency, mountaineer-founder
    (7,6,4,5),                          # 2013
    (7,6,4,5),                          # 2014
    (5,5,7,4),                          # 2015 identity crisis: "barbarian at the gate"
    (5,5,8,4),                          # 2016 open war narrative
    (5,5,6,4),                          # 2017 founder displaced; cultural rupture
    (6,6,4,5),                          # 2018 professional culture reconsolidates
    (6,6,4,5),                          # 2019
    (6,6,5,5),                          # 2020
 ]),
 "Stewards": series([
    (7,7,4,5), (7,7,4,5), (7,7,4,5), (7,7,4,5),  # 2010-2013 low-leverage, cash-collection discipline
    (7,7,4,6),                          # 2014
    (6,7,6,5),                          # 2015 net debt 19.3%, collection >90% cushions the battle
    (6,7,6,5),                          # 2016 operations intact through feud (sales record)
    (7,7,5,5),                          # 2017 rail+property pivot
    (7,7,5,6),                          # 2018
    (7,7,5,6),                          # 2019
    (7,7,6,6),                          # 2020
 ]),
 "Craft": series([
    (6,7,4,5), (6,7,4,5), (6,7,4,5), (6,7,4,5), (6,7,4,5),  # 2010-2014 standardised development technique, prefab pioneer
    (6,7,5,5),                          # 2015
    (6,7,5,5),                          # 2016
    (6,7,5,5),                          # 2017
    (6,7,5,6),                          # 2018
    (6,7,5,6),                          # 2019
    (6,7,6,6),                          # 2020 industrialised construction expands
 ]),
 "Hands": series([
    (6,7,4,4), (6,7,4,4), (6,7,4,4), (6,7,4,4),  # 2010-2013 national contractor ecosystem
    (6,7,4,4),                          # 2014
    (6,7,5,4),                          # 2015
    (6,7,6,4),                          # 2016 sales record despite control battle
    (6,7,5,5),                          # 2017
    (6,7,5,5),                          # 2018
    (6,7,5,5),                          # 2019
    (6,7,6,5),                          # 2020
 ]),
 "Archive": series([
    (6,7,4,5), (6,7,4,5), (6,7,4,5),    # 2010-2012 transparency/disclosure tradition studied by peers
    (6,7,4,5),                          # 2013
    (6,7,4,6),                          # 2014 partner plan codifies incentive precedent
    (5,6,6,5),                          # 2015 disclosure battlefield; decision records contested
    (5,6,7,5),                          # 2016
    (6,6,5,5),                          # 2017
    (6,6,5,5),                          # 2018
    (6,6,5,5),                          # 2019
    (6,6,5,5),                          # 2020
 ]),
 "Flow": series([
    (7,7,4,5), (7,7,4,5), (7,7,4,5), (7,7,4,5), (7,7,4,5),  # 2010-2014 strongest sales/cash collection in industry
    (5,6,8,4),                          # 2015 shares suspended; funding-structure attack; credit profile threatened
    (5,6,8,4),                          # 2016 A-shares halted ~1 yr; bond/channel flows continue
    (6,7,5,5),                          # 2017 normalisation
    (6,7,5,5),                          # 2018
    (6,7,5,5),                          # 2019
    (6,7,6,5),                          # 2020
 ]),
}

# ---------------------------------------------------------------- Longfor
LONGFOR = {
 "Helm": series([
    (7,6,4,5),                          # 2010 sales +81.5% to 33.3bn; top-10; founder discipline
    (7,6,4,5),                          # 2011 CEO handed to professional manager (Shao), Wu stays chair
    (6,6,5,5),                          # 2012 divorce; 28% stake transfer; control uncertainty (stock -4.2%)
    (7,6,4,5), (7,6,4,5), (7,6,4,5),    # 2013-2015 steady top-10 expansion
    (7,7,4,5),                          # 2016 "Platform + End" management digitisation
    (7,7,5,5),                          # 2017 sales >100bn while staying conservative
    (7,7,5,6),                          # 2018 stake to daughter, voting per Wu (succession engineering)
    (7,7,5,6),                          # 2019 Midea-model: decide to institutionalise to professional managers
    (7,7,5,6),                          # 2020
 ]),
 "Shield": series([
    (5,5,4,4), (5,5,4,4), (5,5,4,4),    # 2010-2012 cornerstone backing (GIC, Temasek, Ping An, BOCIG)
    (5,5,4,4),                          # 2013
    (5,5,4,4),                          # 2014
    (5,5,4,4),                          # 2015
    (5,5,4,4),                          # 2016
    (5,6,4,5),                          # 2017 scale + funding-market credibility as moat
    (5,6,4,5),                          # 2018
    (5,6,5,5),                          # 2019
    (5,6,5,5),                          # 2020 strongest private-developer balance sheet
 ]),
 "Lore": series([
    (6,6,4,4), (6,6,4,4), (6,6,4,4),    # 2010-2012 "not the largest but the best" (Wang Shi); satisfaction #1 surveys
    (6,6,4,4),                          # 2013
    (6,6,4,4),                          # 2014
    (6,6,4,4),                          # 2015
    (6,6,4,4),                          # 2016
    (6,6,4,5),                          # 2017
    (6,6,4,5),                          # 2018
    (6,6,4,5),                          # 2019
    (6,6,4,5),                          # 2020
 ]),
 "Stewards": series([
    (7,6,4,5),                          # 2010 PwC-audited transparency since founding
    (7,6,4,5),                          # 2011
    (7,6,4,5),                          # 2012
    (7,6,4,5),                          # 2013
    (7,6,4,5),                          # 2014
    (7,6,4,5),                          # 2015
    (7,6,4,5),                          # 2016
    (7,7,5,5),                          # 2017 >100bn sales with conservative leverage
    (7,7,5,6),                          # 2018
    (7,7,5,6),                          # 2019
    (7,7,5,6),                          # 2020
 ]),
 "Craft": series([
    (6,6,4,4),                          # 2010
    (6,7,4,4),                          # 2011
    (6,7,4,4),                          # 2012
    (6,7,4,4),                          # 2013
    (6,7,4,4),                          # 2014
    (6,7,4,4),                          # 2015
    (6,7,4,5),                          # 2016
    (7,7,4,5),                          # 2017
    (7,7,5,5),                          # 2018
    (7,7,5,5),                          # 2019
    (7,7,5,5),                          # 2020 mid/high-end residential + commercial (Paradise Walk) operations craft
 ]),
 "Hands": series([
    (6,7,4,4), (6,7,4,4), (6,7,4,4), (6,7,4,4),  # 2010-2013
    (6,7,4,4),                          # 2014
    (6,7,4,4),                          # 2015
    (6,7,4,4),                          # 2016
    (6,7,5,5),                          # 2017
    (6,7,5,5),                          # 2018
    (6,7,5,5),                          # 2019
    (6,7,5,5),                          # 2020
 ]),
 "Archive": series([
    (6,6,4,5), (6,6,4,5), (6,6,4,5),    # 2010-2012 PwC transparency culture; codified quality systems
    (6,6,4,5),                          # 2013
    (6,6,4,5),                          # 2014
    (6,6,4,5),                          # 2015
    (6,6,4,5),                          # 2016
    (6,6,4,5),                          # 2017
    (6,6,4,6),                          # 2018
    (6,6,4,6),                          # 2019 institutionalisation decision recorded/codified
    (6,6,4,6),                          # 2020
 ]),
 "Flow": series([
    (6,6,4,4),                          # 2010 IPO-fresh capital; sales +81.5%
    (6,6,4,4),                          # 2011
    (5,6,5,4),                          # 2012 divorce share-transfer shock (-4.2% day)
    (6,6,4,4),                          # 2013
    (6,6,4,4),                          # 2014
    (6,6,4,4),                          # 2015
    (6,6,4,4),                          # 2016
    (7,7,4,5),                          # 2017
    (7,7,4,5),                          # 2018
    (7,7,5,5),                          # 2019
    (7,7,5,5),                          # 2020
 ]),
}

# ---------------------------------------------------------------- SMIC
SMIC = {
 "Helm": series([
    (4,5,6,4), (4,5,6,4),               # 2010-2011 post-Zhang churn; state shareholders; no long-term catch-up commitment
    (4,5,6,4),                          # 2012
    (4,5,6,4),                          # 2013
    (4,5,6,4),                          # 2014
    (4,5,6,4),                          # 2015
    (4,5,6,4),                          # 2016
    (5,6,6,5),                          # 2017 co-CEO structure (Zhao Haijun + Liang Mong Song)
    (5,6,6,5),                          # 2018
    (5,6,7,5),                          # 2019 14nm mass production end-2019
    (4,6,8,4),                          # 2020 Entity List; Liang resignation letter circulates
 ]),
 "Shield": series([
    (4,5,6,4),                          # 2010 weakened by TSMC settlement terms
    (4,5,6,4),                          # 2011
    (4,5,6,4),                          # 2012
    (4,5,6,4),                          # 2013
    (4,5,6,4),                          # 2014
    (4,5,6,4),                          # 2015
    (4,5,6,4),                          # 2016
    (5,6,6,4),                          # 2017
    (5,6,6,4),                          # 2018
    (5,6,7,4),                          # 2019
    (4,5,9,4),                          # 2020 military-end-user designation + Entity List (Dec 2020)
 ]),
 "Lore": series([
    (5,5,6,4),                          # 2010
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (5,5,6,4),                          # 2015
    (5,5,6,4),                          # 2016
    (6,6,6,4),                          # 2017 "China's chip hope" mission identity strengthens
    (6,6,6,4),                          # 2018
    (6,6,7,4),                          # 2019
    (5,5,8,3),                          # 2020 sanctions strain + internal rift visible (Liang letter)
 ]),
 "Stewards": series([
    (4,4,7,3),                          # 2010 losses; settlement obligations
    (4,4,7,3),                          # 2011
    (5,5,6,4),                          # 2012 consistently profitable after 2012
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (5,5,6,4),                          # 2015
    (5,5,6,4),                          # 2016
    (5,6,7,4),                          # 2017 big-fund era capex; 1/10 TSMC scale
    (5,6,7,4),                          # 2018
    (5,6,7,4),                          # 2019
    (5,6,8,4),                          # 2020
 ]),
 "Craft": series([
    (5,5,6,4),                          # 2010 mature-node execution only
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (5,5,6,4),                          # 2015
    (5,5,6,4),                          # 2016 28nm ramp struggles
    (5,6,6,4),                          # 2017 Liang arrives; 14nm development
    (5,6,7,5),                          # 2018 14nm FinFET flow
    (5,6,7,5),                          # 2019 14nm mass production (~6 yrs behind TSMC)
    (5,6,8,5),                          # 2020 equipment cutoff threatens technique pipeline
 ]),
 "Hands": series([
    (5,6,6,4),                          # 2010
    (5,6,6,4),                          # 2011
    (5,6,6,4),                          # 2012
    (5,6,6,4),                          # 2013
    (5,6,6,4),                          # 2014
    (5,6,6,4),                          # 2015
    (5,6,6,4),                          # 2016
    (6,7,6,4),                          # 2017
    (6,7,7,4),                          # 2018
    (6,7,7,4),                          # 2019
    (6,7,7,4),                          # 2020 fab workforce ~18k, multi-site ramp
 ]),
 "Archive": series([
    (3,4,6,3),                          # 2010 stolen-records legacy: 15k TSMC docs surrendered; know-how unusable
    (3,4,6,3),                          # 2011
    (3,4,6,3),                          # 2012
    (3,4,6,3),                          # 2013
    (3,4,6,3),                          # 2014
    (4,4,6,3),                          # 2015 rebuilt R&D records mature
    (4,5,6,4),                          # 2016
    (5,5,6,4),                          # 2017 own process recipes documented
    (5,6,6,5),                          # 2018
    (5,6,6,5),                          # 2019
    (5,6,7,5),                          # 2020
 ]),
 "Flow": series([
    (5,5,6,4),                          # 2010
    (5,5,6,4),                          # 2011
    (5,5,6,4),                          # 2012
    (5,5,6,4),                          # 2013
    (5,5,6,4),                          # 2014
    (5,5,6,4),                          # 2015
    (5,5,6,4),                          # 2016
    (6,6,6,4),                          # 2017
    (6,6,7,4),                          # 2018 equipment access tightens; Huawei/Hisilicon demand rises
    (6,6,7,4),                          # 2019
    (4,5,9,3),                          # 2020 equipment/materials cutoff (Entity List, 10nm rule)
 ]),
}

ENTITIES = [
    ("China (national)", CHINA),
    ("Evergrande", EVERGRANDE),
    ("Huawei", HUAWEI),
    ("BYD", BYD),
    ("CATL", CATL),
    ("Vanke", VANKE),
    ("Longfor", LONGFOR),
    ("SMIC", SMIC),
]

# ---- evidence-status sets: years where year-specific evidence was examined ----
# change-years are added automatically. Series start = New_Evidence by rule.
EVIDENCE_YEARS = {
    "China (national)": set(range(2010, 2021)),   # dense year-by-year facts examined
    "Evergrande": {2010, 2011, 2013, 2015, 2016, 2017, 2018, 2020},
    "Huawei": {2010, 2011, 2014, 2015, 2016, 2018, 2019, 2020},
    "BYD": {2010, 2011, 2013, 2015, 2016, 2017, 2019, 2020},
    "CATL": {2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2020},
    "Vanke": {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019},
    "Longfor": {2010, 2011, 2012, 2016, 2017, 2018, 2019, 2020},
    "SMIC": {2010, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020},
}
REAFFIRMED_YEARS = {
    "China (national)": set(),
    "Evergrande": set(),
    "Huawei": set(),
    "BYD": set(),
    "CATL": set(),
    "Vanke": {2018, 2020},
    "Longfor": set(),
    "SMIC": set(),
}

def status_for(entity, node, year, prev, cur):
    if prev is None:
        return "New_Evidence"
    if cur != prev:
        return "New_Evidence"
    if year in EVIDENCE_YEARS[entity]:
        return "New_Evidence"
    if year in REAFFIRMED_YEARS[entity]:
        return "Reaffirmed"
    return "Carried_Forward"

out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "CAMS_ChinaSet_2010-2020.csv")
stat_path = os.path.join(out_dir, "CAMS_ChinaSet_2010-2020_status.csv")

rows, srows = [], []
for entity, matrix in ENTITIES:
    assert set(matrix.keys()) == set(NODES), f"node set mismatch for {entity}"
    for y in YEARS:
        for n in NODES:
            cur = matrix[n][y]
            prev = matrix[n][y - 1] if y - 1 in matrix[n] else None
            rows.append([entity, y, n, *cur])
            st = status_for(entity, n, y, prev, cur)
            srows.append([entity, y, n, st, st, st, st])

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "Coherence", "Capacity", "Stress", "Abstraction"])
    w.writerows(rows)

with open(stat_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Entity", "Year", "Node", "C_status", "K_status", "S_status", "A_status"])
    w.writerows(srows)

print(f"wrote {csv_path}: {len(rows)} rows")
print(f"wrote {stat_path}: {len(srows)} rows")
