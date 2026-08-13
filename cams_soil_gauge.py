import numpy as np
import sys
from pathlib import Path

def robson_gauge(cams_data, scaling_factor=1000.0, epsilon=2.0):
    nodes = ['Helm', 'Shield', 'Flow', 'Hands', 'Craft', 'Stewards', 'Archive', 'Lore']
    V = {}
    for node in nodes:
        d = cams_data.get(node, {})
        V[node] = (d.get('C', 5.0) + d.get('K', 5.0) - d.get('S', 5.0) + 0.5 * d.get('A', 5.0))
    V_array = np.array(list(V.values()))
    sigma_V = np.std(V_array, ddof=0)

    def q(node):
        d = cams_data.get(node, {})
        return (0.6 * d.get('C', 5.0) + 0.4 * d.get('A', 5.0)) / 10.0

    def W(i, j):
        si = cams_data.get(i, {}).get('S', 5.0)
        sj = cams_data.get(j, {}).get('S', 5.0)
        return np.sqrt(q(i) * q(j)) * (2.0 ** (-(si + sj) / 10.0))

    def BS(node):
        others = [n for n in nodes if n != node]
        return np.mean([W(node, other) for other in others])

    numerator = BS('Lore') * BS('Archive') * cams_data.get('Hands', {}).get('C', 5.0)
    denominator = cams_data.get('Hands', {}).get('S', 5.0) * max(sigma_V, 1e-6) + epsilon
    eta = (numerator / denominator) * scaling_factor

    return round(eta, 2), {
        'V_mean': round(float(np.mean(V_array)), 2),
        'sigma_V': round(float(sigma_V), 2),
        'BS_Lore': round(float(BS('Lore')), 3),
        'BS_Archive': round(float(BS('Archive')), 3),
        'C_Hands': cams_data.get('Hands', {}).get('C', 5.0),
        'S_Hands': cams_data.get('Hands', {}).get('S', 5.0),
        'scaling': scaling_factor
    }

# === Example archetypes (run these by default) ===
examples = {
    "Singapore_Artificial": {
        'Helm': {'C':9.0,'K':8.5,'S':1.5,'A':8.5},
        'Shield': {'C':9.5,'K':7.5,'S':1.0,'A':7.5},
        'Flow': {'C':9.0,'K':8.0,'S':2.0,'A':8.5},
        'Hands': {'C':8.5,'K':7.5,'S':1.0,'A':8.0},
        'Craft': {'C':8.0,'K':8.5,'S':1.5,'A':8.0},
        'Stewards': {'C':9.0,'K':8.5,'S':1.0,'A':9.0},
        'Archive': {'C':9.5,'K':9.0,'S':0.5,'A':9.5},
        'Lore': {'C':8.5,'K':8.5,'S':1.5,'A':8.5}
    },
    "France_Organic": {
        'Helm': {'C':6.5,'K':7.0,'S':4.5,'A':6.0},
        'Shield': {'C':7.5,'K':6.5,'S':4.0,'A':6.5},
        'Flow': {'C':6.0,'K':6.5,'S':5.0,'A':6.5},
        'Hands': {'C':5.5,'K':5.5,'S':6.0,'A':5.5},
        'Craft': {'C':6.5,'K':7.0,'S':4.0,'A':6.5},
        'Stewards': {'C':7.0,'K':6.5,'S':4.5,'A':7.5},
        'Archive': {'C':8.0,'K':8.0,'S':3.5,'A':8.0},
        'Lore': {'C':7.0,'K':7.5,'S':4.0,'A':7.5}
    },
    "USA_Strained": {
        'Helm': {'C':4.0,'K':5.0,'S':7.0,'A':4.5},
        'Shield': {'C':8.5,'K':6.0,'S':3.0,'A':5.5},
        'Flow': {'C':5.5,'K':5.5,'S':6.5,'A':5.0},
        'Hands': {'C':4.5,'K':4.0,'S':7.5,'A':4.0},
        'Craft': {'C':5.0,'K':6.0,'S':5.5,'A':5.5},
        'Stewards': {'C':4.0,'K':4.5,'S':6.0,'A':4.5},
        'Archive': {'C':5.5,'K':6.0,'S':5.0,'A':6.0},
        'Lore': {'C':4.5,'K':5.0,'S':6.5,'A':4.5}
    }
}

if __name__ == "__main__":
    print("CAMS Robson Gauge (η_soil) — v3.2-R canonical implementation\n")
    for name, data in examples.items():
        eta, details = robson_gauge(data, scaling_factor=1000.0)
        print(f"{name}:")
        print(f"  η_soil     = {eta}")
        print(f"  V_mean     = {details['V_mean']}")
        print(f"  sigma_V    = {details['sigma_V']}")
        print(f"  BS_Lore    = {details['BS_Lore']}")
        print(f"  BS_Archive = {details['BS_Archive']}")
        print(f"  Hands C/S  = {details['C_Hands']}/{details['S_Hands']}")
        print("  Interpretation: " + ("Artificial high anchoring (Singapore pattern)" if eta > 800 else
                                     "Organic / WATCH range" if eta < 100 else
                                     "Moderate Maritime"))
        print("-" * 60)

    print("\nScript ready. Paste the full output back to me.")
