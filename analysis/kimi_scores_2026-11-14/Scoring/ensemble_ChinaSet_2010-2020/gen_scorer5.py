# Scorer 5 raw score generator - CAMS Universal Instruction Set v1.4
# Entities x 2010-2020 x 8 nodes (canonical order: Helm, Shield, Lore, Stewards,
# Craft, Hands, Archive, Flow) x (C, K, S, A). Integers 1-10 or NA.
# CATL founded 2011 -> 2010 = NA on all nodes.

NODES = ["Helm", "Shield", "Lore", "Stewards", "Craft", "Hands", "Archive", "Flow"]
YEARS = list(range(2010, 2021))

# Each entity: node -> list of 11 (C,K,S,A) tuples for 2010..2020
S = {}

S["China (national)"] = {
    # Helm: Hu-era collective leadership 6; Xi consolidation +13; Xi Thought/1st era
    # plans 7-8; 2020 COVID early-signal failure (Li Wenliang reprimand, ~7-week
    # delay) pulls C to 7, rupture-level stress 9.
    "Helm": [(6,6,5,6),(6,6,5,6),(6,6,6,6),(7,7,6,7),(7,7,6,7),(7,7,6,7),
             (8,8,6,7),(8,8,6,8),(8,8,7,8),(8,8,7,8),(7,8,9,7)],
    # Shield: PLA modernising 6s; 2013-16 reforms (15 CMC bodies, 5 theater commands
    # Feb 2016, 300k cut) C/K +1; SCS island-building; joint ops outline Nov 2020.
    "Shield": [(6,6,5,5),(6,6,5,5),(6,6,5,6),(7,7,6,6),(7,7,6,6),(7,7,6,7),
               (8,7,6,7),(8,8,6,7),(8,8,7,7),(8,8,7,7),(8,8,7,7)],
    # Lore: Hu slogans waning 6; Chinese Dream/core values 7; Xi Thought in party
    # and state constitution (2017/18) 8; COVID victory narrative but early-2020
    # anger (Li Wenliang death) keeps S 7.
    "Lore": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(7,7,6,6),(7,7,6,6),(7,7,6,6),
             (8,7,6,7),(8,8,6,8),(8,8,6,8),(8,8,6,8),(8,8,7,8)],
    # Stewards: post-4tn local debt build-up 2010-12; debt swap 2015, overcapacity,
    # financial de-risking 2017, poverty campaign, three red lines Aug 2020;
    # 2020 triple load -> S 8.
    "Stewards": [(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,7,6),(6,6,7,6),(6,6,7,6),
                 (7,7,7,6),(7,7,7,7),(7,7,7,7),(7,7,7,7),(7,7,8,7)],
    # Craft: HSR 8,358 km (2010) -> 37,900 km (2020), Fuxing 2017; Mozi 2016,
    # Chang'e-4 2019, Chang'e-5 Dec 2020; 5G lead; tech-war stress 2019+.
    "Craft": [(6,6,5,5),(6,6,5,5),(6,6,5,6),(7,7,6,6),(7,7,6,7),(7,7,6,7),
              (8,7,6,7),(8,8,6,7),(8,8,7,7),(8,8,7,7),(8,8,7,7)],
    # Hands: migrant wage growth 10-15%/yr 2010-12, episodic shortages, working-age
    # share peaked 2010; informal share rising; COVID 2020: shutdown then 98%
    # resumption - bond held (C 8), rupture load (S 9).
    "Hands": [(7,7,7,5),(7,7,7,5),(7,7,7,5),(7,6,7,6),(7,6,7,6),(7,7,7,6),
              (7,7,7,6),(7,7,7,6),(7,7,7,6),(7,7,7,6),(8,7,9,7)],
    # Archive: censuses, guiding-cases system 2010, env monitoring nets 2013-15,
    # social-credit records (10.7bn items 2017); 2020: early-warning signal
    # suppressed (Li Wenliang) vs genome sequenced Jan 5 - S +2 to 8, C dips to 6.
    "Archive": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,7,5,6),(6,7,6,6),(6,7,6,6),
                (7,7,6,6),(7,7,6,7),(7,7,6,7),(7,7,6,7),(6,7,8,7)],
    # Flow: exporter scale, HSR/courier circulation; 2015-16 capital flight + stock
    # crash (C dips); BRI corridors, SDR 2016, CIPS; 2020: persons sealed, goods
    # resilient - S +2 to 8.
    "Flow": [(7,7,5,5),(7,7,5,5),(7,7,5,5),(7,7,5,6),(6,7,6,6),(6,7,6,6),
             (7,7,6,6),(7,8,6,7),(7,8,6,7),(7,8,6,7),(7,8,8,7)],
}

S["Evergrande"] = {
    # Helm: Hui 77.2% control, decisions bind absolutely (C 7); K falls 2018 as
    # Faraday/Season Smart fiasco + incoherent diversification; S rises to 8 2020
    # (buyback cliff, collapsed listing); A 4 - reflex leverage model throughout.
    "Helm": [(7,6,5,4),(7,6,5,4),(7,6,5,4),(7,6,5,4),(7,6,6,4),(7,6,6,4),
             (7,7,7,4),(7,7,7,4),(6,5,7,4),(6,5,7,4),(6,5,8,4)],
    # Shield: Shenzhen govt + CITIC state ties, 130bn state-linked strategic
    # investors 2017; shelter holds till 2019; 2020 red-lines pilot + denied
    # Guangdong letter - protection strained (K 6, S 7, C 6).
    "Shield": [(7,7,4,5),(7,7,4,5),(7,7,4,5),(7,7,4,5),(7,7,5,5),(7,7,5,5),
               (7,7,5,5),(7,7,5,5),(7,7,6,5),(7,6,6,5),(6,6,7,5)],
    # Lore: Guangzhou FC bought 2010, AFC Champions League 2013 & 2015, 7th
    # straight CSL title 2017; brand machinery peaks K 8 2017-18; strains 2019-20.
    "Lore": [(6,6,4,5),(7,7,4,5),(7,7,4,5),(7,7,4,5),(7,7,4,5),(7,7,4,5),
             (7,8,4,5),(7,8,5,5),(7,7,6,5),(7,7,6,5),(6,6,7,5)],
    # Stewards: 50% dividend payout 2011-2020 while leverage built; short-term
    # debt +76% end-2017; buyback clause on 130bn; all three red lines crossed
    # (83/159/0.6) - allocation divorced from constraint (C 4, K 4 2020, S 9).
    "Stewards": [(6,6,5,4),(6,6,5,4),(6,6,6,4),(6,6,6,4),(5,6,6,4),(5,6,6,4),
                 (5,6,7,4),(4,6,8,4),(4,5,8,4),(4,5,8,4),(4,4,9,4)],
    # Craft: high-leverage/high-turnover development model standardised
    # nationally (real technique, K 7); 2018+ EV entries with no product (K -2);
    # S 8 2020 (no car, red lines).
    "Craft": [(6,7,5,5),(6,7,5,5),(6,7,5,5),(6,7,5,5),(6,7,5,5),(6,7,6,5),
              (6,7,6,5),(6,7,6,4),(5,5,7,4),(5,5,7,4),(5,4,8,4)],
    # Hands: massive presale delivery machine (bond holds, C 7); strain rises
    # 2017+; 2020 S 8 (cash-flow warning letter, supplier ecosystem strain).
    "Hands": [(7,7,6,4),(7,7,6,4),(7,7,6,4),(7,7,6,4),(7,7,6,4),(7,7,6,4),
              (7,7,7,4),(7,7,7,4),(7,6,7,4),(7,6,7,4),(6,5,8,4)],
    # Archive: standardised repeatable playbook (land-bank cycle) = usable memory
    # (K 6); financial records later proven false (2024 CSRC) NOT back-cast;
    # record-system strain grows with refinancing opacity 2019-20.
    "Archive": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),
                (6,6,5,5),(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,7,5)],
    # Flow: presale->cash->land machine at national scale (sales #1 2017);
    # refinancing dependence (short-term +76% 2017); 2020: ban on new debt -
    # capital circulation throttled (C 5, K 5, S 8).
    "Flow": [(7,7,5,5),(7,7,5,5),(7,7,5,5),(7,7,5,5),(7,7,6,5),(7,7,6,5),
             (7,7,6,5),(7,7,7,5),(6,6,7,5),(6,6,7,5),(5,5,8,4)],
}

S["Huawei"] = {
    # Helm: founder authority + rotating chairs, decisions bind; crisis direction
    # held 2019-20 ("spend its way out"); K dips 2020 (growth stalls); S 9 2020.
    "Helm": [(7,7,6,7),(7,7,6,7),(7,7,6,7),(7,7,6,7),(7,8,6,7),(7,8,6,7),
             (7,8,7,8),(7,8,7,8),(7,8,8,8),(8,8,8,8),(8,7,9,8)],
    # Shield: state diplomatic backing + 2018-20 full legal-diplomatic machine
    # (Meng Wanzhou defence, Canadian detainees); protection fails vs US
    # jurisdiction over chips - S 8-9 2018-20.
    "Shield": [(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,7,6,5),
               (7,7,6,6),(7,8,7,6),(7,8,8,7),(7,7,9,7),(7,7,9,7)],
    # Lore: national-champion symbolism; post-2018 Meng/loyalty rallying binds
    # employees (union shareholding) and domestic public; S rises with siege.
    "Lore": [(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,7),
             (7,7,5,7),(7,7,6,7),(7,7,7,7),(8,8,8,7),(8,8,8,7)],
    # Stewards: deep cash stores, low debt; allocation discipline (R&D
    # intensity held as revenue fell - margins squeezed by choice); strain 2019+.
    "Stewards": [(8,8,5,6),(8,8,5,6),(8,8,5,7),(8,8,5,7),(8,8,5,7),(8,8,6,7),
                 (8,8,6,7),(8,8,6,7),(8,8,7,7),(8,7,8,7),(8,7,8,7)],
    # Craft: R&D $2.5bn (2010) -> $22bn/15.9% (2020); 87,805 patents granted
    # by 2018; 5G patent lead; Kirin, foldables, HarmonyOS 2019; Innovation 2.0.
    "Craft": [(6,7,6,6),(6,7,6,6),(7,7,6,6),(7,7,6,7),(7,8,6,7),(7,8,6,7),
              (7,8,6,7),(7,8,7,7),(7,8,7,7),(8,8,8,8),(8,7,9,8)],
    # Hands: large engineering workforce held through sanctions (no flight);
    # retention/mobilisation strain rises 2019-20 (S 8).
    "Hands": [(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,7,7),
              (7,7,7,7),(7,7,7,7),(7,7,8,7),(7,7,8,7),(7,7,8,7)],
    # Archive: patent/knowledge base built pre-restrictions (ITIF: R&D
    # overspending built knowledge to confront restrictions); standards
    # engagement; 2020: loss of advanced-chip know-how pathway - S 9.
    "Archive": [(7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,6,7),
                (7,7,6,7),(7,8,6,8),(7,8,7,8),(8,8,8,8),(8,8,9,8)],
    # Flow: global carrier/device circulation cut by Entity List (May 2019),
    # Google cut, Aug 2020 FDPR chip cut; domestic 5G buildout absorbs volume.
    "Flow": [(7,8,6,7),(7,8,6,7),(7,8,6,7),(7,8,6,7),(7,8,6,7),(7,8,7,7),
             (7,8,7,7),(7,8,7,7),(7,7,8,7),(6,6,9,8),(6,6,9,7)],
}

S["BYD"] = {
    # Helm: Wang Chuanfu centralised, two-decade NEV consistency through subsidy
    # cycles; 2020 mask pivot in one month = decision-binding (C/K 8, A 8).
    "Helm": [(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,7),(7,7,6,7),
             (7,7,6,7),(7,7,6,7),(7,7,7,7),(7,7,7,7),(8,8,8,8)],
    # Shield: Shenzhen champion, Buffett/Munger backing (2008), subsidy-era
    # regulatory shelter, state procurement (e-taxis, buses); 2020 state-scale
    # mask reliance deepens mutual dependence (K 8).
    "Shield": [(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,5,5),(7,6,5,5),(7,7,5,5),
               (7,7,5,5),(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,8,6,5)],
    # Lore: green-tech mission ("Cool the Earth by 1C"), engineer culture;
    # brand deepens with Blade/Han 2020 (C/K 8).
    "Lore": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(7,7,5,6),
             (7,7,5,6),(7,7,5,6),(7,7,6,6),(7,7,6,6),(8,8,6,7)],
    # Stewards: vertical integration (cells, IGBT, motors) = custody of stores;
    # conservative balance sheet vs peers; recurring capability grows (A up).
    "Stewards": [(6,6,5,6),(6,6,5,6),(6,6,5,6),(6,6,5,6),(6,7,5,6),(6,7,5,6),
                 (7,7,5,7),(7,7,5,7),(7,7,6,7),(7,7,6,7),(7,8,7,8)],
    # Craft: LFP/battery lineage, e6 2010, Qin 2013, Tang 2015, 7+4 2015, SkyRail
    # 2017, IGBT 4.0 2018, Blade battery + Han 2020 (nail-test no-fire), masks.
    "Craft": [(6,7,6,6),(6,7,6,6),(6,7,6,6),(6,7,6,7),(6,7,6,7),(7,7,6,7),
              (7,7,6,7),(7,7,6,7),(7,7,7,7),(7,7,7,7),(8,7,7,8)],
    # Hands: manufacturing workforce scales with plant network (Shenzhen 350k
    # capacity); no bond dissolution evidence; 2020 mobilisation adds load (S 7).
    "Hands": [(6,7,5,5),(6,7,5,5),(6,7,5,5),(6,7,5,5),(6,7,6,6),(7,7,6,6),
              (7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,8,7,8)],
    # Archive: vertically integrated process know-how (cells->packs->IGBT);
    # FinDreams standardisation 2019-20; record strain low (S 5-6).
    "Archive": [(6,7,5,6),(6,7,5,6),(6,7,5,6),(6,7,5,6),(6,7,5,6),(6,7,5,6),
                (6,7,5,6),(6,7,5,6),(6,7,6,6),(6,7,6,6),(7,7,6,7)],
    # Flow: e6 taxi circulation, global bus exports, Han 13,701 units Dec 2020;
    # 2020 mask output = goods-circulation flex (K 8).
    "Flow": [(7,6,5,5),(7,6,5,5),(7,6,5,5),(7,6,5,5),(7,6,5,6),(7,6,5,6),
             (7,6,5,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(8,8,7,7)],
}

S["CATL"] = {
    # 2010: NA all nodes (founded 2011).
    # Helm: Zeng-led consistent battery strategy; IPO 2018, BMW/Erfurt, JV
    # architecture (FAW 2019); decisions bind and scale.
    "Helm": [(None,None,None,None),(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,7,6,6),(6,7,6,6),
             (7,7,7,6),(7,7,7,7),(8,8,7,7),(8,8,7,7),(8,8,7,7)],
    # Shield: 2011 rule tying EV subsidies to domestic tech transfer = created
    # market; Ningde/local and Thuringia government welcome; no major threats.
    "Shield": [(None,None,None,None),(7,6,5,5),(7,6,5,5),(7,6,5,5),(7,6,5,5),(7,6,5,5),
               (7,6,5,5),(7,6,5,5),(7,7,5,5),(7,7,5,5),(7,7,6,5)],
    # Lore: B2B champion brand (BMW/Tesla association), engineer magnetism;
    # modest symbolic load.
    "Lore": [(None,None,None,None),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),
             (6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5)],
    # Stewards: capacity build with strong post-IPO balance sheet; R&D ~7%
    # income steady; custody aligned to national EV constraint.
    "Stewards": [(None,None,None,None),(6,5,6,5),(6,5,6,5),(6,5,6,5),(6,6,6,5),(6,7,6,6),
                 (7,7,7,6),(7,8,7,6),(8,8,7,6),(8,8,7,6),(8,8,7,7)],
    # Craft: BMW Brilliance 2012 first customer; #3 globally 2016; 11.84 GWh
    # world #1 2017; NCM811 industry-first 2018; million-mile battery June 2020;
    # Fraunhofer BITC life-cycle modelling 2020 (A 8).
    "Craft": [(None,None,None,None),(5,5,6,5),(6,6,6,5),(6,6,6,5),(6,6,6,6),(6,7,6,6),
              (7,8,7,7),(8,8,7,7),(8,8,7,7),(8,8,7,7),(8,8,7,8)],
    # Hands: gigafactory workforce ramp (Ningde; Erfurt 600 jobs planned);
    # expansion load S 6-7.
    "Hands": [(None,None,None,None),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,6,5),(6,7,6,6),
              (7,7,7,6),(7,7,7,6),(8,8,7,6),(8,8,7,6),(8,8,7,6)],
    # Archive: process know-how, ppb quality systems, Fraunhofer cooperation
    # 2020 (BattLife) = records becoming models.
    "Archive": [(None,None,None,None),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,7,6,5),(6,7,6,6),
               (7,7,6,6),(7,7,6,7),(7,7,6,7),(8,8,6,7),(8,8,6,8)],
    # Flow: cells to BMW/VW/Daimler/JLR/PSA pipeline; Erfurt localisation;
    # Tesla 2020; customer circulation widens steadily.
    "Flow": [(None,None,None,None),(6,5,6,5),(6,6,6,5),(6,6,6,5),(7,7,6,5),(7,7,6,6),
             (7,7,7,6),(7,8,7,6),(8,8,7,6),(8,8,7,7),(8,8,7,7)],
}

S["Vanke"] = {
    # Helm: Wang Shi professionalised governance, decisions bind, method
    # governs (A 8); 2015-16 Baoneng raid: dilution blocked, trading halt,
    # board-removal requisition -> C 4 K 6 S 8; 2017 SZ Metro 29% + regulator
    # bans end conflict (C +2); Yu Liang succession stable.
    "Helm": [(8,8,5,8),(8,8,5,8),(8,8,5,8),(8,8,5,8),(8,8,6,8),(4,6,8,5),
             (6,7,5,6),(7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,5,7)],
    # Shield: dispersed ownership left it exposed (K 6); defence failed
    # privately - settled by state regulators (CIRC ban on Yao, Feb 2017)
    # and SOE white knight -> K 8 2017+.
    "Shield": [(7,6,5,6),(7,6,5,6),(7,6,5,6),(7,6,5,6),(7,6,7,6),(5,5,9,7),
               (7,8,6,7),(7,8,5,6),(7,8,5,6),(7,8,5,6),(7,8,5,6)],
    # Lore: Wang Shi personal brand + quality benchmark bind industry and
    # staff ("no gifts" culture, transparency); takeover battle strains 2015-17.
    "Lore": [(8,7,4,6),(8,7,4,6),(8,7,4,6),(8,7,4,6),(8,7,5,6),(6,6,7,6),
             (7,7,6,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6)],
    # Stewards: industry benchmark custody (low leverage, cash-rich); 2017
    # sales +88% to 581bn (3rd) with logistics/rental allocation ahead of
    # peers; allocation tied to constraint (C 8 held).
    "Stewards": [(8,8,5,7),(8,8,5,7),(8,8,5,7),(8,8,5,7),(8,8,6,7),(7,8,7,7),
                 (7,8,5,7),(7,8,5,7),(7,8,5,7),(7,8,5,7),(7,8,5,7)],
    # Craft: construction-quality standard setting, industrialised building,
    # property-management technique; steady deepening.
    "Craft": [(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,7),(7,7,5,7),
              (7,7,5,7),(7,8,5,7),(7,8,5,7),(7,8,5,7),(7,8,5,7)],
    # Hands: delivery via contractors + large PM workforce; decade-long bond
    # held; 2020 COVID construction halt adds load (S 6).
    "Hands": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),
              (6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,6,5)],
    # Archive: codified management systems (industry teaching material),
    # trusted reporting, managed 2017 succession = memory outlives founder;
    # 2016 record-contestation strain (S 6).
    "Archive": [(8,7,4,7),(8,7,4,7),(8,7,4,7),(8,7,4,7),(8,7,5,7),(6,6,6,7),
               (7,7,4,7),(7,7,4,7),(7,7,4,7),(7,7,4,7),(7,7,4,7)],
    # Flow: presale machine #1 2010-2015; Dec 2015-Jul 2016 trading halt
    # blocks equity circulation (C/K drop); recovery 2017+; 2020 industry
    # red-lines + COVID tighten flows again (C 6, S 6).
    "Flow": [(7,8,5,7),(7,8,5,7),(7,8,5,7),(7,8,5,7),(7,8,6,7),(5,6,7,7),
             (7,8,6,7),(7,8,5,7),(7,8,5,7),(7,8,5,7),(6,8,6,7)],
}

S["Longfor"] = {
    # Helm: Wu Yajun disciplined strategy (control scale 2013), Platform+End
    # 2016, Leadership Development Center 2015, professional-succession
    # conception 2019 (Midea study) - method governs.
    "Helm": [(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),
             (7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,5,7),(7,7,6,7)],
    # Shield: clean private developer, investment-grade ratings 2016; three
    # red lines 2020: green status preserves credit access while sector
    # protection demand spikes (S +2 to 6, K 8).
    "Shield": [(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,5,5),
               (7,6,5,5),(7,7,5,5),(7,7,5,5),(7,7,5,5),(7,8,6,5)],
    # Lore: "top student" of private developers, garden-home quality brand,
    # low-turnover culture; modest strain throughout.
    "Lore": [(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,4,5),(7,6,4,5),
             (7,6,4,5),(7,6,4,5),(7,6,5,5),(7,6,5,5),(7,6,6,5)],
    # Stewards: lowest-leverage custody among majors, malls/apartments
    # recurring income, record-low bond coupons 2015; allocation tied to
    # constraint (C 8 decade-long).
    "Stewards": [(8,7,4,7),(8,7,4,7),(8,7,4,7),(8,7,4,7),(8,7,4,7),(8,7,4,7),
                 (8,7,5,7),(8,7,5,7),(8,8,5,7),(8,8,5,7),(8,8,6,7)],
    # Craft: development + Sky Street commercial operations + PM external
    # expansion 2016; quality standards white paper 2012.
    "Craft": [(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),
              (7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6)],
    # Hands: steady delivery workforce; no strain or rupture evidence -
    # scored conservatively.
    "Hands": [(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),
              (6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5),(6,6,5,5)],
    # Archive: Property Standards White Paper 2012, Leadership Development
    # Center 2015, vocational training school lineage - know-how codified.
    "Archive": [(7,7,4,6),(7,7,4,6),(7,7,4,7),(7,7,4,7),(7,7,4,7),(7,7,4,7),
                (7,7,4,7),(7,7,4,7),(7,7,4,7),(7,7,4,7),(7,7,4,7)],
    # Flow: sales 40bn 2012 -> 100bn 2017; diversified funding (USD notes
    # 2011, record RMB bonds 2014-15); top-10 circulation; 2020 tightening.
    "Flow": [(7,6,5,6),(7,6,5,6),(7,6,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),
             (7,7,5,6),(7,7,5,6),(7,8,5,6),(7,8,5,6),(7,8,6,6)],
}

S["SMIC"] = {
    # Helm: post-2009 lawsuit resettlement (Zhang Rujing exit, David Wang),
    # first profit 2010; National IC Fund seat 2015; Liang Mong Song co-CEO
    # 2017; Dec 2020 Liang resignation letter over Chiang appointment -
    # schism visible (C 5) though he stays.
    "Helm": [(6,6,7,5),(6,6,7,5),(6,6,6,5),(6,6,6,5),(6,7,6,6),(6,7,6,6),
             (6,7,7,6),(6,7,7,7),(7,7,8,7),(7,7,8,7),(5,6,9,7)],
    # Shield: state backing deepens (IC Fund 17% 2015; 45.7bn RMB into
    # subsidiaries; local JVs); 2020 DoD CCMC listing + Commerce letter +
    # Dec 18 Entity List - external protection fails (S 9) while domestic
    # shelter holds (STAR IPO July 2020 raises capital regardless).
    "Shield": [(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,7,6,5),(7,8,6,5),(7,8,6,6),
               (7,8,7,6),(7,8,7,6),(7,8,8,7),(7,7,9,7),(7,7,9,7)],
    # Lore: national-champion mission binds engineers; geopolitical identity
    # pressure rises with US designations.
    "Lore": [(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,6,5),(6,6,6,5),
             (6,6,7,5),(6,6,7,5),(6,6,8,5),(6,6,8,5),(6,6,8,5)],
    # Stewards: allocation aligned to national chip constraint; heavy capex
    # funded by state and STAR raise; 10nm+ R&D/capex blocked (S 9 2020).
    "Stewards": [(7,6,6,6),(7,6,6,6),(7,6,6,6),(7,6,6,6),(7,7,6,6),(7,7,7,6),
                 (7,7,7,6),(7,7,7,6),(7,7,8,6),(7,7,8,6),(7,7,9,6)],
    # Craft: 65nm mass production Aug 2010; 40nm late 2012; 28nm 2015;
    # 14nm FinFET volume Nov 2019 (independent IP); EUV order blocked 2018;
    # mature nodes 92% of revenue Q1 2020 - core job done, frontier cut.
    "Craft": [(6,7,7,6),(6,7,7,6),(6,7,7,6),(6,7,6,6),(7,7,6,6),(7,7,6,6),
              (7,7,6,6),(7,7,7,7),(7,7,8,7),(8,7,8,8),(7,7,9,7)],
    # Hands: fab workforce 24/7; talent wars (Liang 2017, Chiang late 2020);
    # COVID-era essential-industry operation; 2020 loyalty strain (S 8).
    "Hands": [(6,7,6,6),(6,7,6,6),(6,7,6,6),(6,7,6,6),(6,7,6,6),(6,7,6,6),
              (6,7,6,6),(6,7,7,6),(6,7,7,6),(6,7,7,6),(6,7,8,6)],
    # Archive: licensed recipes (IBM/Chartered) -> indigenous FinFET IP 2019
    # = abstraction rise; 2020: know-how pathway to sub-10nm blocked (S 9).
    "Archive": [(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,5,6),(7,7,6,6),(7,7,6,7),
                (7,7,6,7),(7,7,6,7),(7,7,7,7),(7,7,8,8),(7,7,9,8)],
    # Flow: Qualcomm 410 on 28nm 2015; HiSilicon ~20% of sales 2020; US
    # customers Qualcomm/Broadcom/TI; 2020: Sept 25 Commerce letter, Dec 18
    # Entity List, EUV already blocked - equipment/customer flows cut (K 5).
    "Flow": [(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),(7,7,6,6),
             (7,7,7,6),(7,7,7,6),(7,7,8,6),(7,7,8,6),(6,5,9,6)],
}

ORDER = ["China (national)", "Evergrande", "Huawei", "BYD", "CATL", "Vanke",
         "Longfor", "SMIC"]

def fmt(v):
    return "NA" if v is None else str(v)

lines = ["Entity,Year,Node,Coherence,Capacity,Stress,Abstraction"]
for ent in ORDER:
    for yi, year in enumerate(YEARS):
        for node in NODES:
            c, k, s, a = S[ent][node][yi]
            lines.append(f"{ent},{year},{node},{fmt(c)},{fmt(k)},{fmt(s)},{fmt(a)}")

out = r"C:\Users\julie\kimi_scores\Scoring\ensemble_ChinaSet_2010-2020\scorer5_raw.csv"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# Verification
n_rows = len(lines) - 1
assert n_rows == 704, n_rows
# node order check per entity-year
for ent in ORDER:
    for year in YEARS:
        idx = [i for i, l in enumerate(lines[1:]) if l.startswith(f"{ent},{year},")]
        assert len(idx) == 8
        block = lines[1+idx[0]:1+idx[0]+8]
        got = [b.split(",")[2] for b in block]
        assert got == NODES, (ent, year, got)
# integer range check
for l in lines[1:]:
    p = l.split(",")
    for v in p[3:]:
        assert v == "NA" or (v.isdigit() and 1 <= int(v) <= 10), l
# CATL 2010 all NA
for node in NODES:
    row = [l for l in lines if l.startswith(f"CATL,2010,{node},")][0]
    assert row.endswith(",NA,NA,NA,NA"), row
print("OK:", n_rows, "data rows; node order, integer range, CATL 2010 NA verified.")
