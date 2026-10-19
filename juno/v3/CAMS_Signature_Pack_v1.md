# CAMS Signature Pack v1 (20 Oct 2026)

**Data:** JUNO_Unified_Dataset_v3_2026-10-19.csv (41,914 rows, 42 societies). Node value V = C + K + A/2 − S on eight nodes: Helm, Shield, Lore, Stewards, Flow, Craft, Hands, Archive.

## Framing
Societies are complex adaptive systems, and functionally scale-invariant. The node scores are primary measurements, the monks' paperwork. A signature is a regularity in how the scored node topology changes over time. Scorers rate only C, K, S and A per node per year, and the signatures are inferred afterwards from repeated recovery of the same shape across nations. They are not something an LLM imposes while scoring. The metrics are the signal. Similar patterns and structural failure modes within a class are sufficient evidence of a regularity.

**Hit lists.** Primary lists come from Lab Chief's frozen-threshold run (prereg commit 1b9b9f1, results 58fcaaf). Thresholds were set as pooled percentiles over changing years before counting, giving 696 firing years. Draft-threshold hits (sig.py) are shown alongside for comparison. Frozen-run exclusions: Denmark, India, Turkey, Israel before 2015, Canada and Thailand. Draft exclusions: Denmark and Israel before 2015 (India, Turkey, Thailand and Canada are not annual, so annual-only rules skip them). Societies marked STEP are step-coded series (Iraq, Italy, Lebanon, Indonesia, Pakistan, Palestine, Israel from 2015), so their hits rest on fewer changing years.

## Decade Zeitgeist context
| Decade | Zeitgeist reading |
|---|---|
| 1880s–1900s | Lore dominant |
| 1910s | 79% of nations falling |
| 1930s | Shield +0.71, Flow −0.62 |
| 1950s–60s | Lowest node dispersion (1.9) |
| 1970s | Shield up |
| 1980s–90s | Lore rebound |
| 2000s | Flow-led |
| 2020s | Shield +0.90, leading in 27% of societies; cross-national synchrony 0.35, the highest on record |

## Headline (frozen run, shuffle null: circular shift within society, 10,000 reps)
| Signature | Firing years | Hits in named windows (obs / chance) | p (windows) | Firings before a big fall (obs / chance) | p (fall) |
|---|---|---|---|---|---|
| Archive amnesia | 22 | 15 / 0.75 | 0.0001 | 5 / 4.65 | 0.4933 |
| Helm Incompetence | 25 | 4 / 0.18 | 0.0223 | 14 / 9.25 | 0.0739 |
| Helm isolation | 225 | 17 / 8.41 | 0.0043 | 57 / 55.94 | 0.4673 |
| Shield inversion | 268 | 24 / 10.37 | 0.006 | 67 / 63.52 | 0.3616 |
| Late abstraction collapse | 104 |  |  | 62 / 38.56 | 0.0002 |
| Narrative–material gap | 52 |  |  | 28 / 20.43 | 0.0718 |

All four signatures with named windows concentrate in those windows far beyond chance: Archive amnesia p=0.0001, Helm isolation 0.0043, Shield inversion 0.006, Helm Incompetence 0.022. Late abstraction collapse precedes big falls well above chance (62 firings vs 38.6 expected, p=0.0002). Helm Incompetence (p=0.074) and the narrative–material gap (p=0.072) lean the same way. These are raw p-values; the prereg specifies Holm correction across signatures, which leaves the window results under 0.05 and Late abstraction collapse at about 0.001 (0.0002 × 6).

## Signatures

### Archive amnesia
*Institutional memory is stripped while the command nodes gain: Archive falls hard as Helm or Shield rises.*

**Rule.** Draft: 5-yr ΔV(Archive) ≤ −3 AND (5-yr ΔV Helm > 0 OR 5-yr ΔV Shield > 0). Frozen: 5-yr ΔV(Archive) ≤ P5 = -5.77, same Helm/Shield condition.

**Counts.** Frozen: 22 firing years in 10 societies. Draft: 61 firing years in 21 societies.

**Named windows (survives if the frozen rule fires anywhere from start−1 to end+1):**

- Argentina 1976–1978: SURVIVED (1978)
- Chile 1975–1977: SURVIVED (1975–1976)
- Cambodia 1975–1978: SURVIVED (1975–1978)
- Greece 1967–1971: SURVIVED (1967–1969, 1971)
- Laos 1975: did not survive
- Germany 1931: did not survive
- Hong Kong 1997: did not survive
- Hong Kong 2020–2024: SURVIVED (2020–2023)

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Argentina | 1978 | 1976–1978 |
| Australia | — | 1976–1980, 2003 |
| Cambodia | 1975–1978 | 1975–1978 |
| Chile | 1975–1976 | 1975–1977 |
| China | 1899, 1961–1962 | 1855–1856, 1858, 1899, 1936, 1961–1962 |
| Finland | — | 1913–1914 |
| Germany | — | 1931 |
| Greece | 1967–1969, 1971 | 1967–1969, 1971 |
| Hong Kong | 2020–2023 | 1997, 2020–2024 |
| Iran | 1911 | 1911 |
| Japan | — | 1932, 2024 |
| Laos | — | 1975 |
| Mongolia | 1939 | 1937, 1939, 1994 |
| Netherlands | — | 1945, 2022 |
| Poland | — | 1915–1916 |
| Russia | 1920 | 1920, 1986, 1988 |
| Singapore | — | 1946 |
| Sweden | 2015 | 2015–2016 |
| Syria | — | 1902–1903, 1966 |
| UK | — | 1941, 1943 |
| Venezuela | — | 1987, 1995, 2012 |

### Helm Incompetence
*Capable society, incapable helm (informally, 'idiots in charge'): Helm loses capacity and gains stress while its abstraction holds, and Craft/Flow capacity is still at or above trend.*

**Rule.** Draft: Helm 3-yr ΔK ≤ −1, ΔS ≥ +1, ΔA ≥ 0, mean K(Craft,Flow) ≥ trailing 10-yr mean. Frozen: ΔK ≤ P10 = -1.0, ΔS ≥ P90 = +2.0, other conditions unchanged.

**Counts.** Frozen: 25 firing years in 8 societies. Draft: 67 firing years in 18 societies.

**Named windows (survives if the frozen rule fires anywhere from start−1 to end+1):**

- UK 2016–2019: SURVIVED (2016–2019)
- Germany 1939: did not survive
- Pakistan 1968–1970: did not survive

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Argentina | — | 2009 |
| Australia | 1948 | 1948 |
| Brazil | 1887–1888 | 1887–1888, 1975 |
| China | — | 1883, 1910, 1931 |
| France | — | 1865–1867 |
| Germany | — | 1939 |
| Hong Kong | 1969 | 1969, 1991 |
| New Zealand | — | 2017–2018 |
| Norway | — | 1965–1967 |
| Pakistan (STEP) | — | 1968–1970 |
| Russia | — | 1856, 1895, 1948 |
| Saudi Arabia | 1928, 1958 | 1928, 1953–1955, 1958–1960, 2011 |
| Singapore | — | 1956 |
| South Africa | 1907–1908, 1941, 1973, 1993 | 1907–1908, 1941, 1973, 1993 |
| Sweden | 1924, 1967–1968, 1986–1987, 2010–2011, 2014 | 1923–1924, 1966–1968, 1986–1987, 2010–2014, 2016 |
| Syria | — | 1979–1982 |
| UK | 2016–2019 | 1885, 1892–1893, 1990, 2001, 2016–2019 |
| Venezuela | 1977, 2008 | 1977, 2008 |

### Helm isolation
*Helm's coherence detaches from the rest of the body politic.*

**Rule.** Draft: |C(Helm) − mean C(other 7)| ≥ 2. Frozen: ≥ P95 = 1.86.

**Counts.** Frozen: 225 firing years in 18 societies. Draft: 200 firing years in 18 societies.

**Named windows (survives if the frozen rule fires anywhere from start−1 to end+1):**

- Venezuela 1999–2013: SURVIVED (1998–2001, 2003–2006, 2008–2012, 2014)
- Australia 2010–2018: SURVIVED (2010, 2015, 2018)

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Australia | 2010, 2015, 2018 | 2010, 2015, 2018 |
| Brazil | 1893 | 1893 |
| Chile | 1976–1978 | 1976–1978 |
| China | 1917–1919, 1923, 1939, 1949–1950, 1960, 1966–1968, 1970 | 1917–1919, 1923, 1939, 1949–1950, 1960, 1966–1968, 1970 |
| Finland | 1901–1902, 1904–1905, 1910–1916 | 1901–1902, 1904–1905, 1910–1915 |
| Germany | 1890–1891, 1902, 1907–1908, 1919, 1935 | 1891, 1907, 1919 |
| Iran | 1917, 1931–1933, 1942–1943, 1978, 2001 | 1917, 1931, 1942–1943, 1978 |
| Israel (STEP) | 2020 | — |
| Laos | 1976–1978 | 1976–1978 |
| Netherlands | 1977, 2002, 2017, 2021, 2023–2025 | 1977, 2002, 2017, 2021, 2023–2025 |
| New Zealand | 1894, 1896, 1900, 1909, 1945 | 1896 |
| Poland ⚑ constant firer | 1875–1904, 1920 | 1875–1904, 1920 |
| Russia ⚑ constant firer | 1827–1828, 1830, 1832–1845, 1847, 1849–1852, 1860, 1883–1884, 1889–1891, 1893, 1919, 1928–1940, 1944, 1946, 1948–1949, 1952, 1990, 2016 | 1827–1828, 1830, 1832–1835, 1837–1845, 1847, 1849–1851, 1891, 1893, 1919, 1929–1940, 1946, 1948–1949, 1952 |
| Saudi Arabia ⚑ constant firer | 1921–1923, 1925–1927, 1929–1944, 1962–1963 | 1921–1923, 1925–1927, 1929–1944, 1962–1963 |
| Sweden ⚑ constant firer | 1887, 1905, 1914, 1922–1931, 1973, 1976–1981, 1986, 2009, 2014–2019, 2021 | 1887, 1905, 1914, 1922–1931, 1973, 1976–1981, 1986, 2014–2016, 2018–2019 |
| Syria | 1893–1894 | 1893–1894 |
| Turkey | — | 1925, 1930 |
| UK | 2018 | 2018 |
| Venezuela ⚑ constant firer | 1985, 1987–1988, 1998–2001, 2003–2006, 2008–2012, 2014, 2016–2025 | 1985, 1987–1988, 1998–2001, 2003–2006, 2008–2012, 2014, 2016–2025 |

### Shield inversion
*Shield becomes the leading node while Flow sinks below the society mean (the 1930s and 2020s pattern).*

**Rule.** Draft: Shield highest V AND V(Flow) < mean V. Frozen: Shield highest V (ties count) AND V(Flow) − mean V ≤ P25 = -1.18.

**Counts.** Frozen: 268 firing years in 23 societies. Draft: 404 firing years in 25 societies.

**Named windows (survives if the frozen rule fires anywhere from start−1 to end+1):**

- Iran 2015–2025: SURVIVED (2018, 2020–2026)
- Iraq 2015–2025: did not survive
- Russia 2015–2025: SURVIVED (2014–2016, 2020, 2022)
- Venezuela 2015–2025: SURVIVED (2015–2020, 2024–2025)
- Finland 2023–2025: SURVIVED (2023–2025)

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Argentina ⚑ constant firer | 1952–1954, 1962, 1980–1982, 1989, 2000–2002, 2018–2019, 2022–2024 | 1952–1954, 1962, 1967, 1971–1972, 1980–1982, 1989, 2000–2002, 2019, 2022–2026 |
| Australia | 1990 | 1945, 1990, 1992 |
| Brazil | — | 1981 |
| Cambodia | 1975–1978 | 1975–1978 |
| Chile | 1919–1920, 1930–1931, 1982–1983 | 1919–1920, 1923, 1930–1931, 1933, 1973–1974, 1982–1984 |
| China | 1958–1963, 1966, 1971, 1974 | 1958–1963, 1965–1968, 1970–1971, 1974, 1977–1978, 1988–1989, 1996, 1999, 2020 |
| Finland | 2023–2025 | 2023–2025 |
| Germany | 1914, 1916 | 1914, 1916, 1934–1935, 1939, 1942 |
| Hong Kong | 1943 | 1943 |
| Iran | 1922–1925, 1938, 1959, 1977, 2010–2011, 2013, 2018, 2020–2026 | 1922–1926, 1929–1930, 1958–1961, 1963, 1977, 2010–2011, 2013, 2015–2018, 2020–2026 |
| Iraq (STEP) | — | 1981–1983, 1985–1994, 1996–2001, 2010–2025 |
| Israel (STEP) | 2024 | — |
| Japan ⚑ constant firer | 1930–1939, 1941–1943, 1992–2001, 2008–2011, 2020–2021 | 1930–1938, 1943, 1992–2001, 2008–2011, 2020–2021 |
| Laos | 1975, 1979, 1997–2000, 2020–2023 | 1975, 1979, 1997–2000, 2020–2023 |
| Mongolia | 1930–1931, 1990, 2013–2016, 2020–2021 | 1928–1931, 1933–1936, 1940–1955, 1990, 2013–2021 |
| Netherlands | 1944 | 1942, 1944, 1982 |
| New Zealand | 1880–1881, 1883–1889 | 1880–1881, 1884–1889, 1915–1917 |
| Pakistan (STEP) | 1958–1967 | — |
| Poland | 1943–1944, 1982–1983, 1985 | 1944, 1947, 1953, 1982–1985 |
| Russia ⚑ constant firer | 1800–1804, 1808–1815, 1817, 1819, 1821–1822, 1824–1835, 1837–1840, 1842–1844, 1846, 1848–1849, 1851–1852, 1854, 1865, 1877, 1885, 1912, 1914, 1916, 1920–1921, 1929–1932, 1935, 1943–1954, 1959–1961, 1963–1982, 1984, 1994, 1998, 2008–2009, 2014–2016, 2020, 2022 | 1800–1804, 1808–1835, 1837–1846, 1848–1849, 1851–1852, 1854, 1857–1858, 1860–1861, 1865, 1869–1872, 1876–1878, 1880, 1882, 1884–1886, 1889–1891, 1893, 1896–1897, 1903, 1910, 1912, 1914, 1920–1921, 1929–1931, 1935, 1943–1954, 1959–1961, 1963, 1965–1982, 1984, 1994, 1998–1999, 2008–2009, 2014–2022 |
| Saudi Arabia | 1921–1923, 1926, 1985–1986 | 1985–1988 |
| Singapore | 1931–1932, 1942–1944 | 1931–1933, 1942–1944, 1946 |
| Sweden | 1973–1980 | 1974–1980 |
| Syria | 2025 | — |
| Thailand | — | 1935, 1950, 1955, 1975, 2020, 2025 |
| Turkey | — | 1895, 1915, 1960, 1980, 2020 |
| USA | — | 1932 |
| Venezuela ⚑ constant firer | 1983–1984, 1987–1988, 1994–1995, 2015–2020, 2024–2025 | 1982–1984, 1986–1988, 1994–1995, 2008–2010, 2013–2021, 2024–2025 |

### Late abstraction collapse
*High abstraction with a sudden loss of coherence: the decoupling of meaning from organisation.*

**Rule.** Draft: max node A ≥ 8 AND some node 2-yr ΔC ≤ −2. Frozen: a node with A ≥ P90 = 7.8 AND that node's 2-yr ΔC ≤ P5 = -1.2.

**Counts.** Frozen: 104 firing years in 10 societies. Draft: 114 firing years in 11 societies.

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Brazil | 1945, 1954, 1961–1962, 1964, 2015–2016 | 1945, 1954, 1961–1962, 1964, 2015–2016 |
| Chile | 1972, 2019–2020 | 1971, 2019 |
| China | 1829, 1955–1957, 1967, 1969, 2011 | 1917–1918, 1956, 1966 |
| Finland | 1909, 1930, 1974, 1992, 2008–2010, 2020, 2024 | 1902, 1908–1910, 1915, 1930, 1939–1940, 1945, 1958, 1974, 1991–1992, 2009, 2020 |
| Germany | 1935, 1941–1943, 1968, 1992 | 1942, 2024 |
| Greece | — | 1974 |
| Japan | 1906–1907, 1990 | 1990 |
| Russia | 1936, 1947, 1951, 1989, 2023 | 1929, 1937–1938, 1990 |
| Singapore | 2009, 2020–2021 | 2009, 2020–2021 |
| South Africa ⚑ constant firer | 1895–1896, 1907, 1913–1915, 1922, 1929–1931, 1939–1940, 1947, 1953, 1973, 1976, 1981, 1993, 2001, 2006–2008 | 1890–1892, 1895–1896, 1907, 1913–1915, 1921–1922, 1929–1931, 1939–1940, 1946–1947, 1953, 1973, 1975–1976, 1981, 1985–1986, 1993, 2001, 2006–2008, 2016, 2020–2021, 2024 |
| Sweden ⚑ constant firer | 1905, 1909, 1914–1918, 1922–1923, 1931–1933, 1939–1941, 1946, 1967–1970, 1973–1974, 1976–1977, 1986–1987, 1990–1992, 2002, 2009–2011, 2014–2015, 2018, 2020, 2022–2023 | 1886–1887, 1905, 1909, 1914–1918, 1922–1923, 1931–1933, 1939–1941, 1946, 1967–1970, 1973–1974, 1976–1977, 1986–1987, 1990–1992, 2002, 2009–2011, 2014–2015, 2018, 2020, 2022–2023 |

### Narrative–material gap
*The story outruns the material base: Lore/Archive abstraction high while productive capacity drops below trend.*

**Rule.** Draft: mean A(Lore,Archive) ≥ 7 AND mean K(Flow,Craft,Hands) ≤ trailing 10-yr mean − 1.5. Frozen: mean A ≥ P90 = 8.0 AND K gap ≤ P10 = -0.9.

**Counts.** Frozen: 52 firing years in 6 societies. Draft: 27 firing years in 6 societies.

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Brazil ⚑ constant firer | 1963–1964, 1980–1983, 2015–2018 | 1964, 2015–2016 |
| Finland ⚑ constant firer | 1931, 1940, 1944–1945, 1990–1993, 2009–2014 | 1918, 1991–1993, 2009–2013 |
| Japan | 1973 | 1974–1975, 1991 |
| Singapore | 2020 | — |
| South Africa | 1914, 1916, 1922, 1931–1932, 2007, 2012 | 1895, 1915, 1932 |
| Sweden ⚑ constant firer | 1909, 1916–1918, 1932, 1970–1971, 1973–1979, 1991–1993, 2009, 2020 | 1917, 1932, 1973, 1975, 1977, 1991–1993 |
| Thailand | — | 2020 |

### Lore–Helm divergence
*Meaning-makers and rulers pull apart (either sign).*

**Rule.** Draft only: |Δ5(V Lore − V Helm)| ≥ 2. Not in the frozen run (fires in ~30% of years at this threshold; treat as a background rate, not a signature hit).

**Counts.** Frozen: 0 firing years in 0 societies. Draft: 1485 firing years in 36 societies.

| Nation | Frozen hits (primary) | Draft hits (comparison) |
|---|---|---|
| Argentina | — | 1955–1957, 1962–1963, 1967, 1976–1980, 1982–1987, 1989, 1991–1996, 1999–2002, 2004–2009, 2012, 2015, 2022, 2026 |
| Australia | — | 1896–1898, 1900, 1916–1917, 1920, 1925, 1929–1931, 1934–1936, 1942–1945, 1947–1950, 1970, 1973–1976, 1979–1981, 1983–1988, 1990–1994, 2001–2003, 2005–2006, 2008–2012, 2018–2023, 2025–2026 |
| Brazil | — | 1885–1892, 1895–1903, 1905–1906, 1910, 1916–1919, 1924, 1926, 1930, 1934–1935, 1937, 1940–1941, 1943–1946, 1950–1951, 1954–1955, 1959–1964, 1966–1970, 1974–1976, 1982–1984, 2013, 2015–2017, 2020–2022, 2024–2025 |
| Cambodia | — | 1954–1958, 1975–1978, 1993–1996, 2000–2004, 2025–2026 |
| Chile | — | 1883–1886, 1889–1893, 1924, 1927–1934, 1936–1937, 1948–1950, 1952–1956, 1959, 1961–1962, 1974–1978, 1982–1990, 1993–1994, 2016–2017, 2019, 2023 |
| China | — | 1811, 1813–1815, 1830, 1833, 1842, 1855, 1862–1863, 1867, 1869, 1881, 1884–1887, 1890–1892, 1900–1901, 1905–1906, 1912, 1915, 1917–1921, 1924–1933, 1935, 1938, 1940, 1943–1944, 1946, 1950, 1953–1954, 1957, 1960–1962, 1965–1970, 1974, 1976–1981, 1989, 1994, 1997, 2002, 2008, 2016–2017, 2020 |
| Colombia | — | 1880, 1899–1903, 1906–1908, 1910, 1948–1950, 1953–1955, 2014–2015, 2020–2021, 2026 |
| Finland | — | 1905–1907, 1909–1914, 1917–1922, 1930, 1934–1937, 1940, 1958–1959, 1961, 1966, 1968–1969, 1974, 1981, 1989–1993, 1995–1999, 2009–2012, 2015–2019, 2025 |
| France | — | 1855–1856, 1865–1874, 1876, 1894, 1904, 1907, 1915, 1917–1918, 1923–1926, 1929, 1933–1935, 1937, 1943, 1945–1947, 1950–1957, 1959–1963, 1965, 1968, 2003–2004, 2006–2007, 2011–2012, 2016, 2018, 2024, 2026 |
| Germany | — | 1888–1892, 1894, 1906, 1911, 1914, 1916, 1918–1923, 1928, 1930–1939, 1942–1943, 1945–1953, 1956–1957, 1973, 1980, 1999, 2015–2016, 2020 |
| Greece | — | 1881, 1891, 1900, 1902, 1908, 1912–1916, 1922–1926, 1931, 1936–1944, 1946–1952, 1954, 1967–1971, 1974–1978, 1981, 1983–1984, 1986, 1988, 1990, 1995, 2008, 2011, 2013, 2016, 2018–2019, 2021–2026 |
| Hong Kong | — | 1913, 1916–1917, 1924, 1926, 1932–1933, 1936–1937, 1939–1942, 1945–1947, 1950, 1955–1956, 1959, 1968, 1971, 1983, 1986–1987, 1990–1992, 1994, 1996, 1998, 2000, 2020–2026 |
| Indonesia (STEP) | — | 1947–1949, 1980–1984 |
| Iran | — | 1889, 1891, 1893–1895, 1901, 1906–1909, 1911–1912, 1918, 1922–1929, 1932, 1936, 1939, 1941–1945, 1947, 1949, 1951–1956, 1960, 1964–1968, 1977–1982, 1989, 1991, 2006–2007, 2009, 2014, 2018 |
| Iraq (STEP) | — | 1914–1918, 1921–1925, 1941–1944, 1946–1949, 1958, 1974, 1977–1978 |
| Italy (STEP) | — | 1906–1908 |
| Japan | — | 1880–1881, 1886, 1891–1894, 1896–1900, 1904–1905, 1909–1910, 1918–1922, 1947–1948, 1950–1951, 1954, 1956, 1974, 1992–1995, 2002–2006, 2012–2016, 2024 |
| Laos | — | 1887, 1892–1893, 1897–1898, 1900, 1907, 1940–1945, 1947–1948, 1950, 1955, 1958–1961, 1965–1966, 1976–1977, 1980, 1998–2000, 2003–2005, 2018–2022, 2025–2026 |
| Lebanon (STEP) | — | 1948 |
| Mongolia | — | 1919, 1922–1926, 1930–1932, 1934–1941, 1943–1944, 1953, 1985–1988, 1994–1995, 1999, 2008, 2014–2016, 2018–2021, 2025–2026 |
| Netherlands | — | 1856–1857, 1860, 1862–1868, 1871–1872, 1877–1878, 1882–1883, 1887, 1892, 1898–1900, 1917–1921, 1931–1933, 1943–1944, 1962, 1967–1969, 1972–1977, 1982–1986, 1991, 1993–1994, 1998–1999, 2002, 2005, 2007–2009, 2012–2013, 2015, 2018–2019, 2021, 2023, 2025 |
| New Zealand | — | 1881–1884, 1893–1898, 1919–1920, 1924–1925, 1929–1931, 1933–1939, 1944, 1946, 1973–1977, 1993–1994, 2001, 2006, 2009–2011, 2014–2018, 2023–2024 |
| Norway | — | 1889, 1905–1910, 1914–1918, 1925, 1934–1938, 1940–1950, 1965, 1972–1973, 1977–1978, 1994, 2001–2005, 2012, 2017 |
| Pakistan (STEP) | — | 1952–1955, 1958–1962, 1968–1972 |
| Palestine (STEP) | — | 1950–1952 |
| Poland | — | 1915–1921, 1925, 1932–1933, 1943–1944, 1947–1951, 1954, 1956–1960, 1979–1981, 1983–1986, 1988–1991, 1995–1996, 1998–1999 |
| Russia | — | 1807, 1811–1813, 1817–1818, 1825–1826, 1830, 1838, 1841, 1843, 1854–1857, 1860–1862, 1877, 1879, 1882–1886, 1895–1899, 1904–1905, 1907, 1909–1910, 1917, 1920–1925, 1928–1932, 1934, 1938, 1941–1944, 1946, 1952–1958, 1968–1969, 1972, 1975, 1981, 1985–1991, 1994–1997, 2000–2001, 2003–2007, 2009–2011, 2014, 2017, 2019, 2024 |
| Saudi Arabia | — | 1905–1909, 1921–1930, 1953–1962, 1964–1969, 1985–1989, 1993, 1996–1997, 2003–2004, 2008, 2015–2019, 2021, 2024–2025 |
| Singapore | — | 1936–1937, 1941, 1943–1944, 1946–1949, 1956, 1959–1962, 1968–1970, 1985, 1987, 1992, 1998, 2003, 2006, 2011, 2016, 2025 |
| South Africa | — | 1885–1886, 1889–1894, 1897, 1904–1905, 1907, 1911–1912, 1914–1919, 1922–1923, 1925–1927, 1929–1932, 1934–1937, 1939–1941, 1944–1945, 1948, 1953–1954, 1968, 1970–1972, 1977–1981, 1990, 1993, 1996–1998, 2001, 2009, 2017–2020, 2022–2025 |
| Sweden | — | 1886–1887, 1891–1893, 1902–1907, 1910–1911, 1913–1917, 1919–1926, 1933–1938, 1941, 1943, 1946, 1949, 1951, 1957, 1960–1962, 1966–1967, 1971–1973, 1977, 1979–1980, 1982–1986, 1988, 1990–1997, 2001, 2003, 2006–2007, 2009–2012, 2015–2018, 2020–2025 |
| Syria | — | 1902–1903, 1928–1932, 1947–1951, 1954, 1961–1965, 1967, 1971–1975, 1979–1983, 1987, 2015–2019 |
| UK | — | 1885–1886, 1890–1893, 1897–1898, 1910–1912, 1915–1917, 1919–1923, 1925, 1927–1928, 1930–1931, 1935–1936, 1941, 1943, 1949, 1954, 1958, 1966–1968, 1971, 1974–1976, 1979–1983, 1986, 1990–1994, 1997–1998, 2006, 2009, 2011–2012, 2015–2021, 2023–2024, 2026 |
| USA | — | 1931–1933 |
| Ukraine | — | 1957–1958, 1968–1969, 1973–1977, 1987, 1989–1993, 1997–1998, 2002–2004, 2010–2012, 2015–2017, 2019–2021, 2024–2026 |
| Venezuela | — | 1977–1982, 1984–1988, 1990–2004, 2006–2007, 2009–2015, 2018, 2020, 2023 |

## Constant-firer flags (frozen run: rate > 3× pooled AND ≥ 10 firing years)

- Helm isolation: Poland (31 of 152 years, rate 0.2039 vs pooled 0.0457)
- Helm isolation: Russia (50 of 227 years, rate 0.2203 vs pooled 0.0457)
- Helm isolation: Saudi Arabia (24 of 127 years, rate 0.189 vs pooled 0.0457)
- Helm isolation: Sweden (29 of 145 years, rate 0.2 vs pooled 0.0457)
- Helm isolation: Venezuela (27 of 56 years, rate 0.4821 vs pooled 0.0457)
- Shield inversion: Argentina (16 of 77 years, rate 0.2078 vs pooled 0.0544)
- Shield inversion: Japan (29 of 152 years, rate 0.1908 vs pooled 0.0544)
- Shield inversion: Russia (100 of 227 years, rate 0.4405 vs pooled 0.0544)
- Shield inversion: Venezuela (14 of 56 years, rate 0.25 vs pooled 0.0544)
- Late abstraction collapse: South Africa (22 of 145 years, rate 0.1517 vs pooled 0.0211)
- Late abstraction collapse: Sweden (39 of 145 years, rate 0.269 vs pooled 0.0211)
- Narrative–material gap: Brazil (10 of 146 years, rate 0.0685 vs pooled 0.0106)
- Narrative–material gap: Finland (14 of 126 years, rate 0.1111 vs pooled 0.0106)
- Narrative–material gap: Sweden (19 of 145 years, rate 0.131 vs pooled 0.0106)

Sweden is the known case. It fires 'Helm Incompetence' repeatedly under both threshold sets (draft 13 years; frozen 1924, 1967–68, 1986–87, 2010–11, 2014), and also dominates Late abstraction collapse and the narrative–material gap. Read this as a property of how Sweden's series moves (scale resolution, oscillation), not as a history of repeated helm failure.

## By-nation index (frozen hits; draft-only signatures in brackets)
Read each nation's history against this.

### Argentina
- Shield inversion: 1952–1954, 1962, 1980–1982, 1989, 2000–2002, 2018–2019, 2022–2024
- [draft only] Lore–Helm divergence: 1955–1957, 1962–1963, 1967, 1976–1980, 1982–1987, 1989, 1991–1996, 1999–2002, 2004–2009, 2012, 2015, 2022, 2026
- Archive amnesia: 1978
- [draft only] Helm Incompetence: 2009

### Australia
- [draft only] Lore–Helm divergence: 1896–1898, 1900, 1916–1917, 1920, 1925, 1929–1931, 1934–1936, 1942–1945, 1947–1950, 1970, 1973–1976, 1979–1981, 1983–1988, 1990–1994, 2001–2003, 2005–2006, 2008–2012, 2018–2023, 2025–2026
- Helm Incompetence: 1948
- [draft only] Archive amnesia: 1976–1980, 2003
- Shield inversion: 1990
- Helm isolation: 2010, 2015, 2018

### Brazil
- [draft only] Lore–Helm divergence: 1885–1892, 1895–1903, 1905–1906, 1910, 1916–1919, 1924, 1926, 1930, 1934–1935, 1937, 1940–1941, 1943–1946, 1950–1951, 1954–1955, 1959–1964, 1966–1970, 1974–1976, 1982–1984, 2013, 2015–2017, 2020–2022, 2024–2025
- Helm Incompetence: 1887–1888
- Helm isolation: 1893
- Late abstraction collapse: 1945, 1954, 1961–1962, 1964, 2015–2016
- Narrative–material gap: 1963–1964, 1980–1983, 2015–2018
- [draft only] Shield inversion: 1981

### Cambodia
- [draft only] Lore–Helm divergence: 1954–1958, 1975–1978, 1993–1996, 2000–2004, 2025–2026
- Archive amnesia: 1975–1978
- Shield inversion: 1975–1978

### Chile
- [draft only] Lore–Helm divergence: 1883–1886, 1889–1893, 1924, 1927–1934, 1936–1937, 1948–1950, 1952–1956, 1959, 1961–1962, 1974–1978, 1982–1990, 1993–1994, 2016–2017, 2019, 2023
- Shield inversion: 1919–1920, 1930–1931, 1982–1983
- Late abstraction collapse: 1972, 2019–2020
- Archive amnesia: 1975–1976
- Helm isolation: 1976–1978

### China
- [draft only] Lore–Helm divergence: 1811, 1813–1815, 1830, 1833, 1842, 1855, 1862–1863, 1867, 1869, 1881, 1884–1887, 1890–1892, 1900–1901, 1905–1906, 1912, 1915, 1917–1921, 1924–1933, 1935, 1938, 1940, 1943–1944, 1946, 1950, 1953–1954, 1957, 1960–1962, 1965–1970, 1974, 1976–1981, 1989, 1994, 1997, 2002, 2008, 2016–2017, 2020
- Late abstraction collapse: 1829, 1955–1957, 1967, 1969, 2011
- [draft only] Helm Incompetence: 1883, 1910, 1931
- Archive amnesia: 1899, 1961–1962
- Helm isolation: 1917–1919, 1923, 1939, 1949–1950, 1960, 1966–1968, 1970
- Shield inversion: 1958–1963, 1966, 1971, 1974

### Colombia
- [draft only] Lore–Helm divergence: 1880, 1899–1903, 1906–1908, 1910, 1948–1950, 1953–1955, 2014–2015, 2020–2021, 2026

### Finland
- Helm isolation: 1901–1902, 1904–1905, 1910–1916
- [draft only] Lore–Helm divergence: 1905–1907, 1909–1914, 1917–1922, 1930, 1934–1937, 1940, 1958–1959, 1961, 1966, 1968–1969, 1974, 1981, 1989–1993, 1995–1999, 2009–2012, 2015–2019, 2025
- Late abstraction collapse: 1909, 1930, 1974, 1992, 2008–2010, 2020, 2024
- [draft only] Archive amnesia: 1913–1914
- Narrative–material gap: 1931, 1940, 1944–1945, 1990–1993, 2009–2014
- Shield inversion: 2023–2025

### France
- [draft only] Lore–Helm divergence: 1855–1856, 1865–1874, 1876, 1894, 1904, 1907, 1915, 1917–1918, 1923–1926, 1929, 1933–1935, 1937, 1943, 1945–1947, 1950–1957, 1959–1963, 1965, 1968, 2003–2004, 2006–2007, 2011–2012, 2016, 2018, 2024, 2026
- [draft only] Helm Incompetence: 1865–1867

### Germany
- [draft only] Lore–Helm divergence: 1888–1892, 1894, 1906, 1911, 1914, 1916, 1918–1923, 1928, 1930–1939, 1942–1943, 1945–1953, 1956–1957, 1973, 1980, 1999, 2015–2016, 2020
- Helm isolation: 1890–1891, 1902, 1907–1908, 1919, 1935
- Shield inversion: 1914, 1916
- [draft only] Archive amnesia: 1931
- Late abstraction collapse: 1935, 1941–1943, 1968, 1992
- [draft only] Helm Incompetence: 1939

### Greece
- [draft only] Lore–Helm divergence: 1881, 1891, 1900, 1902, 1908, 1912–1916, 1922–1926, 1931, 1936–1944, 1946–1952, 1954, 1967–1971, 1974–1978, 1981, 1983–1984, 1986, 1988, 1990, 1995, 2008, 2011, 2013, 2016, 2018–2019, 2021–2026
- Archive amnesia: 1967–1969, 1971
- [draft only] Late abstraction collapse: 1974

### Hong Kong
- [draft only] Lore–Helm divergence: 1913, 1916–1917, 1924, 1926, 1932–1933, 1936–1937, 1939–1942, 1945–1947, 1950, 1955–1956, 1959, 1968, 1971, 1983, 1986–1987, 1990–1992, 1994, 1996, 1998, 2000, 2020–2026
- Shield inversion: 1943
- Helm Incompetence: 1969
- Archive amnesia: 2020–2023

### Indonesia (STEP)
- [draft only] Lore–Helm divergence: 1947–1949, 1980–1984

### Iran
- [draft only] Lore–Helm divergence: 1889, 1891, 1893–1895, 1901, 1906–1909, 1911–1912, 1918, 1922–1929, 1932, 1936, 1939, 1941–1945, 1947, 1949, 1951–1956, 1960, 1964–1968, 1977–1982, 1989, 1991, 2006–2007, 2009, 2014, 2018
- Archive amnesia: 1911
- Helm isolation: 1917, 1931–1933, 1942–1943, 1978, 2001
- Shield inversion: 1922–1925, 1938, 1959, 1977, 2010–2011, 2013, 2018, 2020–2026

### Iraq (STEP)
- [draft only] Lore–Helm divergence: 1914–1918, 1921–1925, 1941–1944, 1946–1949, 1958, 1974, 1977–1978
- [draft only] Shield inversion: 1981–1983, 1985–1994, 1996–2001, 2010–2025

### Israel (STEP)
- Helm isolation: 2020
- Shield inversion: 2024

### Italy (STEP)
- [draft only] Lore–Helm divergence: 1906–1908

### Japan
- [draft only] Lore–Helm divergence: 1880–1881, 1886, 1891–1894, 1896–1900, 1904–1905, 1909–1910, 1918–1922, 1947–1948, 1950–1951, 1954, 1956, 1974, 1992–1995, 2002–2006, 2012–2016, 2024
- Late abstraction collapse: 1906–1907, 1990
- Shield inversion: 1930–1939, 1941–1943, 1992–2001, 2008–2011, 2020–2021
- [draft only] Archive amnesia: 1932, 2024
- Narrative–material gap: 1973

### Laos
- [draft only] Lore–Helm divergence: 1887, 1892–1893, 1897–1898, 1900, 1907, 1940–1945, 1947–1948, 1950, 1955, 1958–1961, 1965–1966, 1976–1977, 1980, 1998–2000, 2003–2005, 2018–2022, 2025–2026
- Shield inversion: 1975, 1979, 1997–2000, 2020–2023
- [draft only] Archive amnesia: 1975
- Helm isolation: 1976–1978

### Lebanon (STEP)
- [draft only] Lore–Helm divergence: 1948

### Mongolia
- [draft only] Lore–Helm divergence: 1919, 1922–1926, 1930–1932, 1934–1941, 1943–1944, 1953, 1985–1988, 1994–1995, 1999, 2008, 2014–2016, 2018–2021, 2025–2026
- Shield inversion: 1930–1931, 1990, 2013–2016, 2020–2021
- Archive amnesia: 1939

### Netherlands
- [draft only] Lore–Helm divergence: 1856–1857, 1860, 1862–1868, 1871–1872, 1877–1878, 1882–1883, 1887, 1892, 1898–1900, 1917–1921, 1931–1933, 1943–1944, 1962, 1967–1969, 1972–1977, 1982–1986, 1991, 1993–1994, 1998–1999, 2002, 2005, 2007–2009, 2012–2013, 2015, 2018–2019, 2021, 2023, 2025
- Shield inversion: 1944
- [draft only] Archive amnesia: 1945, 2022
- Helm isolation: 1977, 2002, 2017, 2021, 2023–2025

### New Zealand
- Shield inversion: 1880–1881, 1883–1889
- [draft only] Lore–Helm divergence: 1881–1884, 1893–1898, 1919–1920, 1924–1925, 1929–1931, 1933–1939, 1944, 1946, 1973–1977, 1993–1994, 2001, 2006, 2009–2011, 2014–2018, 2023–2024
- Helm isolation: 1894, 1896, 1900, 1909, 1945
- [draft only] Helm Incompetence: 2017–2018

### Norway
- [draft only] Lore–Helm divergence: 1889, 1905–1910, 1914–1918, 1925, 1934–1938, 1940–1950, 1965, 1972–1973, 1977–1978, 1994, 2001–2005, 2012, 2017
- [draft only] Helm Incompetence: 1965–1967

### Pakistan (STEP)
- [draft only] Lore–Helm divergence: 1952–1955, 1958–1962, 1968–1972
- Shield inversion: 1958–1967
- [draft only] Helm Incompetence: 1968–1970

### Palestine (STEP)
- [draft only] Lore–Helm divergence: 1950–1952

### Poland
- Helm isolation: 1875–1904, 1920
- [draft only] Archive amnesia: 1915–1916
- [draft only] Lore–Helm divergence: 1915–1921, 1925, 1932–1933, 1943–1944, 1947–1951, 1954, 1956–1960, 1979–1981, 1983–1986, 1988–1991, 1995–1996, 1998–1999
- Shield inversion: 1943–1944, 1982–1983, 1985

### Russia
- Shield inversion: 1800–1804, 1808–1815, 1817, 1819, 1821–1822, 1824–1835, 1837–1840, 1842–1844, 1846, 1848–1849, 1851–1852, 1854, 1865, 1877, 1885, 1912, 1914, 1916, 1920–1921, 1929–1932, 1935, 1943–1954, 1959–1961, 1963–1982, 1984, 1994, 1998, 2008–2009, 2014–2016, 2020, 2022
- [draft only] Lore–Helm divergence: 1807, 1811–1813, 1817–1818, 1825–1826, 1830, 1838, 1841, 1843, 1854–1857, 1860–1862, 1877, 1879, 1882–1886, 1895–1899, 1904–1905, 1907, 1909–1910, 1917, 1920–1925, 1928–1932, 1934, 1938, 1941–1944, 1946, 1952–1958, 1968–1969, 1972, 1975, 1981, 1985–1991, 1994–1997, 2000–2001, 2003–2007, 2009–2011, 2014, 2017, 2019, 2024
- Helm isolation: 1827–1828, 1830, 1832–1845, 1847, 1849–1852, 1860, 1883–1884, 1889–1891, 1893, 1919, 1928–1940, 1944, 1946, 1948–1949, 1952, 1990, 2016
- [draft only] Helm Incompetence: 1856, 1895, 1948
- Archive amnesia: 1920
- Late abstraction collapse: 1936, 1947, 1951, 1989, 2023

### Saudi Arabia
- [draft only] Lore–Helm divergence: 1905–1909, 1921–1930, 1953–1962, 1964–1969, 1985–1989, 1993, 1996–1997, 2003–2004, 2008, 2015–2019, 2021, 2024–2025
- Helm isolation: 1921–1923, 1925–1927, 1929–1944, 1962–1963
- Shield inversion: 1921–1923, 1926, 1985–1986
- Helm Incompetence: 1928, 1958

### Singapore
- Shield inversion: 1931–1932, 1942–1944
- [draft only] Lore–Helm divergence: 1936–1937, 1941, 1943–1944, 1946–1949, 1956, 1959–1962, 1968–1970, 1985, 1987, 1992, 1998, 2003, 2006, 2011, 2016, 2025
- [draft only] Archive amnesia: 1946
- [draft only] Helm Incompetence: 1956
- Late abstraction collapse: 2009, 2020–2021
- Narrative–material gap: 2020

### South Africa
- [draft only] Lore–Helm divergence: 1885–1886, 1889–1894, 1897, 1904–1905, 1907, 1911–1912, 1914–1919, 1922–1923, 1925–1927, 1929–1932, 1934–1937, 1939–1941, 1944–1945, 1948, 1953–1954, 1968, 1970–1972, 1977–1981, 1990, 1993, 1996–1998, 2001, 2009, 2017–2020, 2022–2025
- Late abstraction collapse: 1895–1896, 1907, 1913–1915, 1922, 1929–1931, 1939–1940, 1947, 1953, 1973, 1976, 1981, 1993, 2001, 2006–2008
- Helm Incompetence: 1907–1908, 1941, 1973, 1993
- Narrative–material gap: 1914, 1916, 1922, 1931–1932, 2007, 2012

### Sweden
- [draft only] Lore–Helm divergence: 1886–1887, 1891–1893, 1902–1907, 1910–1911, 1913–1917, 1919–1926, 1933–1938, 1941, 1943, 1946, 1949, 1951, 1957, 1960–1962, 1966–1967, 1971–1973, 1977, 1979–1980, 1982–1986, 1988, 1990–1997, 2001, 2003, 2006–2007, 2009–2012, 2015–2018, 2020–2025
- Helm isolation: 1887, 1905, 1914, 1922–1931, 1973, 1976–1981, 1986, 2009, 2014–2019, 2021
- Late abstraction collapse: 1905, 1909, 1914–1918, 1922–1923, 1931–1933, 1939–1941, 1946, 1967–1970, 1973–1974, 1976–1977, 1986–1987, 1990–1992, 2002, 2009–2011, 2014–2015, 2018, 2020, 2022–2023
- Narrative–material gap: 1909, 1916–1918, 1932, 1970–1971, 1973–1979, 1991–1993, 2009, 2020
- Helm Incompetence: 1924, 1967–1968, 1986–1987, 2010–2011, 2014
- Shield inversion: 1973–1980
- Archive amnesia: 2015

### Syria
- Helm isolation: 1893–1894
- [draft only] Archive amnesia: 1902–1903, 1966
- [draft only] Lore–Helm divergence: 1902–1903, 1928–1932, 1947–1951, 1954, 1961–1965, 1967, 1971–1975, 1979–1983, 1987, 2015–2019
- [draft only] Helm Incompetence: 1979–1982
- Shield inversion: 2025

### Thailand
- [draft only] Shield inversion: 1935, 1950, 1955, 1975, 2020, 2025
- [draft only] Narrative–material gap: 2020

### Turkey
- [draft only] Shield inversion: 1895, 1915, 1960, 1980, 2020
- [draft only] Helm isolation: 1925, 1930

### UK
- [draft only] Lore–Helm divergence: 1885–1886, 1890–1893, 1897–1898, 1910–1912, 1915–1917, 1919–1923, 1925, 1927–1928, 1930–1931, 1935–1936, 1941, 1943, 1949, 1954, 1958, 1966–1968, 1971, 1974–1976, 1979–1983, 1986, 1990–1994, 1997–1998, 2006, 2009, 2011–2012, 2015–2021, 2023–2024, 2026
- [draft only] Archive amnesia: 1941, 1943
- Helm Incompetence: 2016–2019
- Helm isolation: 2018

### USA
- [draft only] Lore–Helm divergence: 1931–1933
- [draft only] Shield inversion: 1932

### Ukraine
- [draft only] Lore–Helm divergence: 1957–1958, 1968–1969, 1973–1977, 1987, 1989–1993, 1997–1998, 2002–2004, 2010–2012, 2015–2017, 2019–2021, 2024–2026

### Venezuela
- Helm Incompetence: 1977, 2008
- [draft only] Lore–Helm divergence: 1977–1982, 1984–1988, 1990–2004, 2006–2007, 2009–2015, 2018, 2020, 2023
- Shield inversion: 1983–1984, 1987–1988, 1994–1995, 2015–2020, 2024–2025
- Helm isolation: 1985, 1987–1988, 1998–2001, 2003–2006, 2008–2012, 2014, 2016–2025
- [draft only] Archive amnesia: 1987, 1995, 2012

## Possible connections (frozen hits, same nation: signature B begins within 5 years after a firing of A)

| A → B | Nations | First instance in each |
|---|---|---|
| Helm isolation → Shield inversion | 9 | Chile 1977→1982; China 1960→1961; Iran 1917→1922; Israel 2020→2024; Laos 1976→1979; Russia 1827→1828; Saudi Arabia 1921→1922; Sweden 1973→1974; Venezuela 1985→1987 |
| Shield inversion → Helm isolation | 9 | China 1958→1960; Germany 1914→1919; Iran 1938→1942; Laos 1975→1976; New Zealand 1889→1894; Russia 1822→1827; Saudi Arabia 1921→1922; Sweden 1973→1976; Venezuela 1983→1985 |
| Helm Incompetence → Helm isolation | 5 | Brazil 1888→1893; Saudi Arabia 1928→1929; Sweden 1924→1925; UK 2016→2018; Venezuela 2008→2009 |
| Narrative–material gap → Late abstraction collapse | 5 | Brazil 1963→1964; Finland 1990→1992; Singapore 2020→2021; South Africa 1914→1915; Sweden 1909→1914 |
| Late abstraction collapse → Helm isolation | 5 | Chile 1972→1976; China 1955→1960; Finland 1909→1910; Russia 1936→1937; Sweden 1909→1914 |
| Late abstraction collapse → Shield inversion | 5 | China 1955→1958; Finland 2020→2023; Japan 1990→1992; Russia 1947→1948; Sweden 1968→1973 |
| Archive amnesia → Shield inversion | 4 | Argentina 1978→1980; Cambodia 1975→1976; China 1961→1962; Russia 1920→1921 |
| Late abstraction collapse → Narrative–material gap | 4 | Brazil 1961→1963; Finland 1930→1931; South Africa 1913→1914; Sweden 1905→1909 |
| Helm isolation → Late abstraction collapse | 4 | China 1950→1955; Finland 1904→1909; Russia 1931→1936; Sweden 1905→1909 |
| Shield inversion → Late abstraction collapse | 4 | China 1962→1967; Finland 2023→2024; Russia 1931→1936; Sweden 1973→1974 |
| Helm isolation → Helm Incompetence | 4 | Saudi Arabia 1923→1928; Sweden 1922→1924; UK 2018→2019; Venezuela 2003→2008 |
| Shield inversion → Archive amnesia | 3 | Cambodia 1975→1976; China 1958→1961; Russia 1916→1920 |
| Archive amnesia → Helm isolation | 3 | Chile 1975→1976; China 1961→1966; Sweden 2015→2016 |
| Late abstraction collapse → Archive amnesia | 3 | Chile 1972→1975; China 1956→1961; Sweden 2010→2015 |
| Helm isolation → Archive amnesia | 3 | China 1960→1961; Russia 1919→1920; Sweden 2014→2015 |
| Archive amnesia → Late abstraction collapse | 2 | China 1962→1967; Sweden 2015→2018 |
| Helm Incompetence → Late abstraction collapse | 2 | South Africa 1908→1913; Sweden 1967→1968 |
| Late abstraction collapse → Helm Incompetence | 2 | South Africa 1907→1908; Sweden 1922→1924 |

These are co-occurrences to read in the histories, not established sequences. A year can satisfy several rules at once because the rules share nodes (Helm in Helm isolation and Helm Incompetence; Lore/Archive A in Late collapse and Narrative gap).

## Data caveats
- Denmark: Stress = 0 in all 870 rows. Excluded; queued for blind rescore.
- Israel before 2015: flat series. Excluded. Israel from 2015 is quarterly-mean and step-coded.
- India and Turkey: 26–31 sampled years (India Hands = 0 anomaly). Excluded from the frozen run, and not annual in the draft run.
- Thailand: 5-yearly ensemble; Canada: 33 years. Excluded from the frozen run.
- Step-coded series (Iraq, Italy, Lebanon, Indonesia, Pakistan, Palestine): hits are labelled STEP; each rests on few changing years.
- UAE removed pending rescore.
- Lore–Helm divergence is draft only and fires in about 30% of years: a background rate.
