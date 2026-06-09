"""CAMS-CORP v0.2 locked formulation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import (
    CF1_GROWTH_BOUNDARY,
    CF2_DISCORDANCE_THRESHOLD,
    CRISIS_MODERATE,
    CRISIS_SEVERE,
    D_HC_P90,
    D_HC_P95,
    EDEWS_MU_RAW,
    EDEWS_SIGMA_RAW,
    EDEWS_THRESHOLDS,
    ETA_EPSILON,
    ETA_THRESHOLDS,
    MU_P75,
    MU_P90,
    SLOW_MEMORY_NODES,
)


def _wide(panel: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return panel.pivot_table(
        index=["Entity", "Year"], columns="Node", values=value_column, aggfunc="first"
    ).sort_index()


def _threshold_label(value: float, thresholds: dict[str, float]) -> str:
    if pd.isna(value):
        return "NONE"
    label = "NONE"
    for name, threshold in sorted(thresholds.items(), key=lambda item: item[1]):
        if abs(value) > threshold:
            label = name
    return label


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        normed = pd.Series(0.0, index=series.index)
    else:
        normed = (values - values.min()) / span
    return 1.0 - normed if invert else normed


def _bounded_phi(current: pd.Series, previous: pd.Series) -> float:
    pairs = pd.concat([current.rename("current"), previous.rename("previous")], axis=1).dropna()
    if len(pairs) < 3:
        return 0.99
    corr = pairs["current"].corr(pairs["previous"])
    prev_std = pairs["previous"].std(ddof=0)
    curr_std = pairs["current"].std(ddof=0)
    if pd.isna(corr) or prev_std == 0 or pd.isna(prev_std):
        return 0.99
    phi = corr * (curr_std / prev_std)
    return min(0.99, max(0.01, float(phi)))


def compute_edews(panel: pd.DataFrame) -> pd.DataFrame:
    abstraction = _wide(panel, "Abstraction")
    coherence = _wide(panel, "Coherence")
    capacity = _wide(panel, "Capacity")
    node_value = _wide(panel, "Node Value")

    rows: list[dict[str, float | str | int]] = []
    for entity in abstraction.index.get_level_values("Entity").unique():
        a_entity = abstraction.loc[entity]
        c_entity = coherence.loc[entity]
        k_entity = capacity.loc[entity]
        v_entity = node_value.loc[entity]

        ratio = np.log(a_entity["Helm"] / c_entity["Craft"])
        delta_log_ac = ratio.diff()
        delta_k_hands = k_entity["Hands"].diff()
        m_slow = v_entity[SLOW_MEMORY_NODES].mean(axis=1)
        m_star = float(m_slow.median()) if float(m_slow.median()) != 0 else 1.0

        raw = (delta_log_ac * (1.0 + 0.5 * (-delta_k_hands).clip(lower=0))) / (1.0 + (m_slow / m_star))
        edews = (raw - EDEWS_MU_RAW) / EDEWS_SIGMA_RAW

        for year in a_entity.index:
            value = edews.loc[year]
            rows.append(
                {
                    "Entity": entity,
                    "Year": int(year),
                    "delta_log_a_helm_over_c_craft": delta_log_ac.loc[year],
                    "delta_k_hands": delta_k_hands.loc[year],
                    "m_slow": m_slow.loc[year],
                    "raw_edews": raw.loc[year],
                    "edews_v0_2": value,
                    "edews_level": _threshold_label(value, EDEWS_THRESHOLDS),
                }
            )
    return pd.DataFrame(rows)


def compute_mu(panel: pd.DataFrame) -> pd.DataFrame:
    node_value = _wide(panel, "Node Value")
    rows: list[dict[str, float | str]] = []
    for entity in node_value.index.get_level_values("Entity").unique():
        v_entity = node_value.loc[entity]
        mus: dict[str, float] = {}
        for node in SLOW_MEMORY_NODES:
            series = v_entity[node].dropna()
            phi = _bounded_phi(series.iloc[1:].reset_index(drop=True), series.iloc[:-1].reset_index(drop=True))
            mus[node] = -math.log(phi)
        entity_mu = float(np.median(list(mus.values())))
        if entity_mu < 0.15:
            classification = "Low decay"
        elif entity_mu <= 0.40:
            classification = "Moderate decay"
        else:
            classification = "High decay"
        rows.append(
            {
                "Entity": entity,
                "mu_archive": mus["Archive"],
                "mu_lore": mus["Lore"],
                "mu_stewards": mus["Stewards"],
                "mu_entity": entity_mu,
                "mu_classification": classification,
                "mu_gt_p75": entity_mu > MU_P75,
                "mu_gt_p90": entity_mu > MU_P90,
            }
        )
    return pd.DataFrame(rows).round(4)


def compute_kappa(panel: pd.DataFrame) -> pd.DataFrame:
    stress = _wide(panel, "Stress")
    bond = panel.groupby(["Entity", "Year"], as_index=False)["Bond Strength"].mean()
    rows: list[dict[str, float | str | int]] = []
    for entity in stress.index.get_level_values("Entity").unique():
        s_entity = stress.loc[entity]
        delta_s = s_entity.diff()
        omega = delta_s.std(axis=1, ddof=0)
        b_entity = bond.loc[bond["Entity"].eq(entity)].set_index("Year")["Bond Strength"]
        kappa = b_entity / omega.clip(lower=0.1)
        percentiles = kappa.dropna().quantile([0.50, 0.75, 0.90]).to_dict()
        for year, value in kappa.items():
            rows.append(
                {
                    "Entity": entity,
                    "Year": int(year),
                    "omega_delta_stress": omega.loc[year],
                    "b_system": b_entity.loc[year],
                    "kappa": value,
                    "kappa_p50_entity": percentiles.get(0.50),
                    "kappa_p75_entity": percentiles.get(0.75),
                    "kappa_p90_entity": percentiles.get(0.90),
                    "kappa_gt_p90_entity": bool(value > percentiles.get(0.90, np.inf)) if not pd.isna(value) else False,
                }
            )
    return pd.DataFrame(rows).round(4)


def compute_cf1(panel: pd.DataFrame) -> pd.DataFrame:
    summary = panel.groupby(["Entity", "Year"], as_index=False)["Node Value"].mean()
    rows: list[dict[str, float | str | bool]] = []
    for entity, entity_panel in panel.groupby("Entity"):
        rho = entity_panel["Stress"].corr(entity_panel["Capacity"])
        yearly = summary.loc[summary["Entity"].eq(entity)].sort_values("Year")
        slope = np.polyfit(yearly["Year"], yearly["Node Value"], 1)[0]
        phase = "growth" if slope > CF1_GROWTH_BOUNDARY else "mature"
        passed = bool(rho >= -0.3) if phase == "growth" else bool(rho < -0.3)
        rows.append(
            {
                "Entity": entity,
                "rho_stress_capacity": rho,
                "v_mean_slope_per_year": slope,
                "phase": phase,
                "cf1_pass": passed,
            }
        )
    return pd.DataFrame(rows).round(4)


def compute_hc_divergence(panel: pd.DataFrame) -> pd.DataFrame:
    values = _wide(panel, "Node Value")
    rows: list[dict[str, float | str | int | bool]] = []
    for entity in values.index.get_level_values("Entity").unique():
        v_entity = values.loc[entity]
        d_hc = v_entity["Helm"] - v_entity["Craft"]
        above = d_hc > D_HC_P90
        consecutive = above.groupby((above != above.shift()).cumsum()).cumcount() + 1
        consecutive = consecutive.where(above, 0)
        for year, value in d_hc.items():
            rows.append(
                {
                    "Entity": entity,
                    "Year": int(year),
                    "d_hc": value,
                    "d_hc_gt_p90": bool(value > D_HC_P90),
                    "d_hc_gt_p95": bool(value > D_HC_P95),
                    "d_hc_consecutive_years_gt_p90": int(consecutive.loc[year]),
                    "hc_divergence_alert": bool(consecutive.loc[year] >= 2),
                }
            )
    return pd.DataFrame(rows).round(4)


def compute_eta_loop(panel: pd.DataFrame) -> pd.DataFrame:
    stress = _wide(panel, "Stress")
    bond = _wide(panel, "Bond Strength")
    eta = (bond["Lore"] * bond["Archive"]) / (stress["Hands"] + ETA_EPSILON)
    rows = []
    for (entity, year), value in eta.items():
        rows.append(
            {
                "Entity": entity,
                "Year": int(year),
                "eta_loop_v0_2": value,
                "eta_below_poetry_threshold": bool(value < ETA_THRESHOLDS["P10"]),
                "eta_gt_p75": bool(value > ETA_THRESHOLDS["P75"]),
                "eta_gt_p90": bool(value > ETA_THRESHOLDS["P90"]),
            }
        )
    return pd.DataFrame(rows).round(4)


def compute_discordance(panel: pd.DataFrame) -> pd.DataFrame:
    slow = panel.loc[panel["Node"].isin(SLOW_MEMORY_NODES)].copy()
    slow["discordance_flag"] = slow["V_range"] > CF2_DISCORDANCE_THRESHOLD
    return slow[
        ["Entity", "Year", "Node", "V_range", "V_min", "V_max", "discordance_flag"]
    ].sort_values(["Entity", "Year", "Node"]).reset_index(drop=True)


def compute_crisis_score(panel: pd.DataFrame, edews: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame:
    yearly = panel.groupby(["Entity", "Year"], as_index=False).agg(
        v_mean=("Node Value", "mean"),
        v_std=("Node Value", "std"),
    )
    m_slow = (
        panel.loc[panel["Node"].isin(SLOW_MEMORY_NODES)]
        .groupby(["Entity", "Year"], as_index=False)["Node Value"]
        .mean()
        .rename(columns={"Node Value": "m_slow"})
    )
    result = yearly.merge(m_slow, on=["Entity", "Year"], validate="one_to_one")
    result = result.merge(edews[["Entity", "Year", "edews_v0_2"]], on=["Entity", "Year"], validate="one_to_one")
    result = result.merge(hc[["Entity", "Year", "d_hc"]], on=["Entity", "Year"], validate="one_to_one")

    result["s_1_low_v_mean"] = _minmax(result["v_mean"], invert=True)
    result["s_2_v_dispersion"] = _minmax(result["v_std"])
    result["s_3_low_m_slow"] = _minmax(result["m_slow"], invert=True)
    max_abs_edews = result["edews_v0_2"].abs().max()
    result["s_4_edews_shear"] = result["edews_v0_2"].abs() / max_abs_edews if max_abs_edews else 0.0
    result["s_5_hc_divergence"] = _minmax(result["d_hc"])
    result["crisis_score_v0_2"] = (
        0.25 * result["s_1_low_v_mean"]
        + 0.20 * result["s_2_v_dispersion"]
        + 0.20 * result["s_3_low_m_slow"]
        + 0.20 * result["s_4_edews_shear"].fillna(0.0)
        + 0.15 * result["s_5_hc_divergence"]
    )
    result["crisis_level"] = "GREEN"
    result.loc[result["crisis_score_v0_2"] > CRISIS_MODERATE, "crisis_level"] = "MODERATE"
    result.loc[result["crisis_score_v0_2"] > CRISIS_SEVERE, "crisis_level"] = "SEVERE"
    return result.round(4)


def compute_alarm_protocol(
    crisis: pd.DataFrame, edews: pd.DataFrame, mu: pd.DataFrame, hc: pd.DataFrame, eta: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    helm = _wide(panel, "Node Value")["Helm"].rename("v_helm").reset_index()
    monitor = crisis[["Entity", "Year", "crisis_score_v0_2"]].merge(
        edews[["Entity", "Year", "edews_v0_2"]], on=["Entity", "Year"], validate="one_to_one"
    )
    monitor = monitor.merge(hc[["Entity", "Year", "d_hc", "hc_divergence_alert"]], on=["Entity", "Year"], validate="one_to_one")
    monitor = monitor.merge(eta[["Entity", "Year", "eta_loop_v0_2"]], on=["Entity", "Year"], validate="one_to_one")
    monitor = monitor.merge(helm, on=["Entity", "Year"], validate="one_to_one")
    monitor = monitor.merge(mu[["Entity", "mu_entity"]], on="Entity", validate="many_to_one")

    def alarm(row: pd.Series) -> str:
        yellow = abs(row["edews_v0_2"]) > EDEWS_THRESHOLDS["WARNING"] or row["mu_entity"] > MU_P75
        orange = abs(row["edews_v0_2"]) > EDEWS_THRESHOLDS["CRITICAL"] or row["hc_divergence_alert"]
        red = row["crisis_score_v0_2"] > CRISIS_SEVERE
        black = red and row["eta_loop_v0_2"] < ETA_THRESHOLDS["P10"] and row["v_helm"] < 6
        if black:
            return "BLACK"
        if red:
            return "RED"
        if orange:
            return "ORANGE"
        if yellow:
            return "YELLOW"
        return "GREEN"

    monitor["alarm_state"] = monitor.apply(alarm, axis=1)
    return monitor.round(4)


def compute_v02_outputs(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    edews = compute_edews(panel)
    mu = compute_mu(panel)
    kappa = compute_kappa(panel)
    cf1 = compute_cf1(panel)
    hc = compute_hc_divergence(panel)
    eta = compute_eta_loop(panel)
    crisis = compute_crisis_score(panel, edews, hc)
    alarms = compute_alarm_protocol(crisis, edews, mu, hc, eta, panel)
    discordance = compute_discordance(panel)
    return {
        "corp_v02_edews": edews.round(4),
        "corp_v02_mu": mu,
        "corp_v02_kappa": kappa,
        "corp_v02_cf1": cf1,
        "corp_v02_hc_divergence": hc,
        "corp_v02_eta_loop": eta,
        "corp_v02_crisis_score": crisis,
        "corp_v02_alarm_protocol": alarms,
        "corp_v02_discordance": discordance,
    }
