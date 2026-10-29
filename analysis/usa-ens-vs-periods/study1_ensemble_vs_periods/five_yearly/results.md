# USA_ENS canonical run (frozen PREREG, sha256 matched FROZEN.txt)
file md5 2f51ff07a507324e6cfcc272e00e1fa4; rows 208; years 26: [1900, 1905, 1910, 1915, 1920, 1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2026]; NaN cells 0; Society values ['United States']; no # header lines
columns identical to hi-fi file:  True
sampled years falling in no frozen period (gap years): [1900, 1920, 1990, 2026]

## Sampled years per frozen period
                       label  n_sampled_years               years
CASE-18      Progressive era                3  [1905, 1910, 1915]
CASE-01                  WWI                0                  []
CASE-16           1920s boom                1              [1925]
CASE-14     Great Depression                1              [1930]
CASE-05             New Deal                2        [1935, 1940]
CASE-09                 WWII                1              [1945]
CASE-04              Postwar                3  [1950, 1955, 1960]
CASE-21                1960s                2        [1965, 1970]
CASE-17                1970s                2        [1975, 1980]
CASE-06         Reagan 1980s                1              [1985]
CASE-23            1991-2001                2        [1995, 2000]
CASE-10            Post-9/11                1              [2005]
CASE-08        GFC aftermath                2        [2010, 2015]
CASE-03  Realignment 2016-20                0                  []
CASE-24              2020-25                1              [2020]

## H1 STRICT (frozen: period needs >=2 sampled years; section 6)
H1 strict: n periods=7 rho=0.607 p_a=0.0854; median r=0.493 p_b=0.0027; thresholds met=True; dropped from H1b=[]

## H1 RELAXED (POST HOC, labelled; periods with >=1 sampled year; not confirmatory)
H1 relaxed: n periods=13 rho=0.610 p_a=0.0162; median r=0.606 p_b=0.0001; thresholds met=True; dropped from H1b=[]

## H2 (annual drift X needs 5 consecutive annual years before t)
drift_X values computable on USA_ENS: 0 (every year has no 5 preceding annual years; spacing is 5 years)
=> H2 NOT EVALUABLE. No threshold, upcrossing, lead or lag count, or null p can be computed without imputing years. Counted as not supported.
years with >=10 observed baseline years in the trailing 20 (needed by D, acute z): 0 => secondaries D, acute z, coupling (10 annual obs in window) NOT EVALUABLE

## H3 (frozen needs both periods retained, i.e. >=2 sampled years; plus bootstrap CI)
                           transition  years_a  years_b  strict_testable  delta_ens  delta_rater                                                    per_rater  same_sign nodes_matching                                                                               CI
0  N-a 1920s boom -> Great Depression        1        1            False    -16.650      -24.083  {'FROZEN_GPT': -24.625, 'CLAUDE': -23.167, 'GROK': -26.375}       True            8/8  degenerate (a period has 1 sampled year, no within-period resampling variation)
1                N-b New Deal -> WWII        2        1            False     16.828       22.417        {'FROZEN_GPT': 22.75, 'CLAUDE': 24.958, 'GROK': 20.5}       True            8/8  degenerate (a period has 1 sampled year, no within-period resampling variation)
2            N-c 1991-2001 -> 2020-25        2        1            False    -21.050      -22.229       {'FROZEN_GPT': -19.0, 'CLAUDE': -26.0, 'GROK': -22.75}       True            8/8  degenerate (a period has 1 sampled year, no within-period resampling variation)
=> H3 STRICT: NOT EVALUABLE for any of the three transitions (each involves a 1-year period). Not supported. Signs above are descriptive only (relaxed, post hoc).

## Holm (convention for non-evaluable tests: p set to 1.0, i.e. cannot pass)
p used: {'H1': 0.08539146085391461, 'H2': 1.0, 'H3': 1.0}
Holm: {'H1': False, 'H2': False, 'H3': False}
H1 strict supported=False  H2 supported=False (not evaluable)  H3 supported=False (not evaluable)

## Control (POST HOC, descriptive): hi-fi annual series subsampled to the USA_ENS sampled years
hi-fi subsampled H1 strict: n periods=7 rho=0.786 p_a=0.0228; median r=0.634 p_b=0.0004; thresholds met=True; dropped from H1b=[]
shared year-node cells 200 (years 1900-2020, 25 years): Pearson r hi-fi vs ENS Q = 0.730; mean Q hi-fi 38.62, ENS 31.23
  N-a 1920s boom -> Great Depression: delta hi-fi subsampled -21.925
  N-b New Deal -> WWII: delta hi-fi subsampled 20.432
  N-c 1991-2001 -> 2020-25: delta hi-fi subsampled -29.323
hi-fi annual, same 7 periods as ENS strict: n periods=7 rho=0.714 p_a=0.0448; median r=0.664 p_b=0.0004; thresholds met=True; dropped from H1b=[]
