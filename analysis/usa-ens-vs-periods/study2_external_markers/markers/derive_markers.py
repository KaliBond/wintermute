#!/usr/bin/env python3
"""Derives marker CSVs from hand-transcribed source tables (no outcome data involved).
Sources are fetched 29 Oct 2026 (Sydney) and cited in PREREG.md section 4. Run: python3 derive_markers.py"""
import csv
HERE = '/workspace/analysis/usa_external_marker_lead/markers/'

# ---- NBER (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions, table updated 14 Mar 2023)
NBER = [('1899-06','1900-12'),('1902-09','1904-08'),('1907-05','1908-06'),('1910-01','1912-01'),('1913-01','1914-12'),
        ('1918-08','1919-03'),('1920-01','1921-07'),('1923-05','1924-07'),('1926-10','1927-11'),('1929-08','1933-03'),
        ('1937-05','1938-06'),('1945-02','1945-10'),('1948-11','1949-10'),('1953-07','1954-05'),('1957-08','1958-04'),
        ('1960-04','1961-02'),('1969-12','1970-11'),('1973-11','1975-03'),('1980-01','1980-07'),('1981-07','1982-11'),
        ('1990-07','1991-03'),('2001-03','2001-11'),('2007-12','2009-06'),('2020-02','2020-04')]

# ---- Party control. End-of-calendar-year control state of each body, from
# House: history.house.gov Party Divisions + Speakers list (organising party; 65th: Clark D elected 2 Apr 1917; 72nd: Garner D 7 Dec 1931)
# Senate: senate.gov/history/partydiv.htm (majority party lines; ties by VP; 107th stated by date)
# President: party at start of term (Wikipedia List of presidents, citing LOC and whitehouse.gov)
def span(d, a, b, p):
    for y in range(a, b + 1): d[y] = p
H = {}; S = {}; P = {}
# House: new House takes effect in the odd start year
span(H,1899,1910,'R'); span(H,1911,1918,'D'); span(H,1919,1930,'R'); span(H,1931,1946,'D'); span(H,1947,1948,'R'); span(H,1949,1952,'D')
span(H,1953,1954,'R'); span(H,1955,1994,'D'); span(H,1995,2006,'R'); span(H,2007,2010,'D'); span(H,2011,2018,'R'); span(H,2019,2022,'D'); span(H,2023,2025,'R')
# Senate
span(S,1899,1912,'R'); span(S,1913,1918,'D'); span(S,1919,1932,'R'); span(S,1933,1946,'D'); span(S,1947,1948,'R'); span(S,1949,1952,'D')
span(S,1953,1954,'R'); span(S,1955,1980,'D'); span(S,1981,1986,'R'); span(S,1987,1994,'D'); span(S,1995,2000,'R'); S[2001]='D'; span(S,2002,2006,'R')
span(S,2007,2014,'D'); span(S,2015,2020,'R'); span(S,2021,2024,'D'); S[2025]='R'
# President (party in office at year end)
span(P,1899,1912,'R'); span(P,1913,1920,'D'); span(P,1921,1932,'R'); span(P,1933,1952,'D'); span(P,1953,1960,'R'); span(P,1961,1968,'D')
span(P,1969,1976,'R'); span(P,1977,1980,'D'); span(P,1981,1992,'R'); span(P,1993,2000,'D'); span(P,2001,2008,'R'); span(P,2009,2016,'D')
span(P,2017,2020,'R'); span(P,2021,2024,'D'); span(P,2025,2025,'R')

with open(HERE + 'party_control_states.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['year','house','senate','president'])
    for y in range(1899, 2026): w.writerow([y, H[y], S[y], P[y]])
def chg(D): return [y for y in range(1900, 2026) if D[y] != D[y-1]]
ch = {'house': chg(H), 'senate': chg(S), 'president': chg(P)}
union = sorted(set(sum(ch.values(), [])))
with open(HERE + 'party_control_switches.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['year','bodies_switched','notes'])
    notes = {1917:'', 1931:'House: Garner (D) elected Speaker 7 Dec 1931; special elections after the 1930 election changed control (House Party Divisions footnote 2)',
             1911:'House: Champ Clark (D) Speaker 4 Apr 1911', 1919:'House: Gillett (R) Speaker 19 May 1919; Senate R from 66th Congress',
             2001:'Senate: D from 6 Jun 2001 (Jeffords); R 20 Jan to 6 Jun 2001; D 3-20 Jan (Senate Historical Office). End-of-year state D. Decision D1.',
             2002:'Senate: R from 12 Nov 2002 per Senate Historical Office line, no reorganisation until 3 Jan 2003. Decision D1: keep 2002 or drop.',
             2021:'Senate: D from 20 Jan 2021 (50-50, VP Harris)'}
    for y in union:
        w.writerow([y, '+'.join(k for k in ch if y in ch[k]), notes.get(y, '')])
with open(HERE + 'nber_peaks.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['peak_month','peak_year','eligible_T_ge_1911'])
    for p, t in NBER: w.writerow([p, int(p[:4]), int(p[:4]) >= 1911])
with open(HERE + 'nber_troughs.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['trough_month','trough_year'])
    for p, t in NBER: w.writerow([t, int(t[:4])])
AM = [('XVI','1913-02-03'),('XVII','1913-04-08'),('XVIII','1919-01-16'),('XIX','1920-08-18'),('XX','1933-01-23'),('XXI','1933-12-05'),
      ('XXII','1951-02-27'),('XXIII','1961-03-29'),('XXIV','1964-01-23'),('XXV','1967-02-10'),('XXVI','1971-07-01'),('XXVII','1992-05-07')]
with open(HERE + 'amendment_ratifications.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['amendment','ratified','year'])
    for a, d in AM: w.writerow([a, d, int(d[:4])])
census = list(range(1920, 2021, 10))
with open(HERE + 'census_years_negative_control.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['year']); [w.writerow([y]) for y in census]
elig = lambda ys: sorted({y for y in ys if 1911 <= y <= 2025})
print('NBER peaks all', [int(p[:4]) for p, _ in NBER]); print('eligible', elig([int(p[:4]) for p, _ in NBER]), len(elig([int(p[:4]) for p, _ in NBER])))
for k, v in ch.items(): print(k, v)
print('party union', union, len(union), 'eligible', len(elig(union)))
ay = elig([int(d[:4]) for _, d in AM]); print('amendment years', ay, len(ay))
print('census', census, len(census))
