import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

DATA = Path('juno/JUNO_Unified_Dataset.csv')
OUT = Path('node')
NODES = ['Lore', 'Shield', 'Archive', 'Hands', 'Craft', 'Stewards', 'Helm', 'Flow']

df = pd.read_csv(DATA)

beta_rows = []
dynamic_rows = []
common_ss = 0.0
total_ss = 0.0

for society, g in df.groupby('Society'):
    p = (g.pivot_table(index='Year', columns='Node', values='Node_Value_Calc', aggfunc='mean')
           .sort_index().dropna(subset=NODES))
    if len(p) < 3:
        continue

    years = p.index.to_numpy()
    dt = np.diff(years).astype(float)
    d = p[NODES].diff().iloc[1:].div(dt, axis=0)

    arr = d.to_numpy()
    system_change = arr.mean(axis=1, keepdims=True)
    common = np.repeat(system_change, len(NODES), axis=1)
    common_ss += np.sum(common ** 2)
    total_ss += np.sum(arr ** 2)

    rel = p[NODES].sub(p[NODES].mean(axis=1), axis=0)

    for node in NODES:
        others = [n for n in NODES if n != node]
        x = d[others].mean(axis=1).to_numpy()
        y = d[node].to_numpy()
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x) if len(x) > 2 and np.var(x) > 0 else np.nan
        beta_rows.append({'Society': society, 'Node': node, 'Beta': beta})

        rv = rel[node].to_numpy()
        if len(rv) > 2 and np.std(rv[:-1]) > 0 and np.std(rv[1:]) > 0:
            persistence = np.corrcoef(rv[:-1], rv[1:])[0, 1]
        else:
            persistence = np.nan

        dynamic_rows.append({
            'Society': society,
            'Node': node,
            'Persistence': persistence,
            'MeanAbsChange': d[node].abs().mean(),
            'ZeroShare': np.isclose(d[node].dropna(), 0).mean(),
        })

beta_df = pd.DataFrame(beta_rows)
dyn_df = pd.DataFrame(dynamic_rows)
node_beta = beta_df.groupby('Node')['Beta'].median().reindex(NODES)
node_dyn = dyn_df.groupby('Node')[['Persistence', 'MeanAbsChange', 'ZeroShare']].median().reindex(NODES)
common_fraction = common_ss / total_ss
relational_fraction = 1 - common_fraction

B = beta_df.pivot(index='Society', columns='Node', values='Beta')[NODES]
loos_rows = []
for held_out in B.index:
    prediction = B.drop(index=held_out).median(axis=0)
    observed = B.loc[held_out]
    rho = spearmanr(prediction.to_numpy(), observed.to_numpy()).statistic

    correct = 0
    total = 0
    for i in range(len(NODES)):
        for j in range(i + 1, len(NODES)):
            correct += int(np.sign(prediction.iloc[i] - prediction.iloc[j]) ==
                           np.sign(observed.iloc[i] - observed.iloc[j]))
            total += 1

    loos_rows.append({
        'Society': held_out,
        'Spearman': rho,
        'PairAccuracy': correct / total,
        'UnitSideAccuracy': np.mean(np.sign(prediction.to_numpy() - 1) ==
                                    np.sign(observed.to_numpy() - 1)),
    })

loos = pd.DataFrame(loos_rows).sort_values('PairAccuracy')
mean_rho = loos['Spearman'].mean()
mean_pair = loos['PairAccuracy'].mean()
mean_unit = loos['UnitSideAccuracy'].mean()

summary = pd.DataFrame({
    'Node': NODES,
    'Median_Beta': [node_beta[n] for n in NODES],
    'Median_Relative_Persistence': [node_dyn.loc[n, 'Persistence'] for n in NODES],
    'Median_Mean_Abs_Annualised_Change': [node_dyn.loc[n, 'MeanAbsChange'] for n in NODES],
    'Median_Zero_Change_Share': [node_dyn.loc[n, 'ZeroShare'] for n in NODES],
})
summary.to_csv(OUT / 'cams_node_dynamics_metrics.csv', index=False)
loos.to_csv(OUT / 'cams_node_dynamics_LOOS.csv', index=False)

# Chart 1
fig, ax = plt.subplots(figsize=(8, 5))
ordered_beta = node_beta.sort_values()
ax.barh(ordered_beta.index, ordered_beta.values)
ax.axvline(1.0, linestyle='--', linewidth=1.5)
ax.set_xlabel('Median response coefficient β')
ax.set_title('Node response architecture')
for i, v in enumerate(ordered_beta.values):
    ax.text(v, i, f'  {v:.3f}', va='center')
fig.tight_layout()
p1 = OUT / 'cams_plot_1_elasticity.png'
fig.savefig(p1, dpi=180, bbox_inches='tight')
plt.close(fig)

# Chart 2
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(loos['Society'], loos['PairAccuracy'] * 100)
ax.axvline(50, linestyle='--', linewidth=1.5)
ax.axvline(mean_pair * 100, linestyle=':', linewidth=1.5)
ax.set_xlabel('Held-out pair-order accuracy (%)')
ax.set_title('Leave-one-society-out prediction')
fig.tight_layout()
p2 = OUT / 'cams_plot_2_loos_accuracy.png'
fig.savefig(p2, dpi=180, bbox_inches='tight')
plt.close(fig)

# Chart 3
fig, ax = plt.subplots(figsize=(7, 5))
parts = pd.Series({'Common-mode': common_fraction * 100, 'Relational': relational_fraction * 100})
ax.bar(parts.index, parts.values)
ax.set_ylabel('Share of squared annualised movement (%)')
ax.set_ylim(0, 100)
ax.set_title('Where node movement lives')
for i, v in enumerate(parts.values):
    ax.text(i, v + 2, f'{v:.1f}%', ha='center')
fig.tight_layout()
p3 = OUT / 'cams_plot_3_common_relational.png'
fig.savefig(p3, dpi=180, bbox_inches='tight')
plt.close(fig)

# Chart 4
fig, ax = plt.subplots(figsize=(8, 5))
x = node_dyn['Persistence']
y = node_dyn['MeanAbsChange']
ax.scatter(x, y, s=80)
for node in NODES:
    ax.annotate(node, (x.loc[node], y.loc[node]), xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Median persistence of relative node position')
ax.set_ylabel('Median absolute annualised change')
ax.set_title('Anchor–amplifier dynamics')
fig.tight_layout()
p4 = OUT / 'cams_plot_4_persistence_movement.png'
fig.savefig(p4, dpi=180, bbox_inches='tight')
plt.close(fig)

# Compose dashboard (charts remain separate matplotlib figures; PIL only assembles them).
images = [Image.open(p).convert('RGB') for p in [p1, p2, p3, p4]]
cell_w = max(im.width for im in images)
cell_h = max(im.height for im in images)
header_h = 190
canvas = Image.new('RGB', (cell_w * 2, header_h + cell_h * 2), 'white')
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype('DejaVuSans-Bold.ttf', 38)
    font_metric = ImageFont.truetype('DejaVuSans-Bold.ttf', 25)
    font_small = ImageFont.truetype('DejaVuSans.ttf', 20)
except OSError:
    font_title = font_metric = font_small = ImageFont.load_default()

draw.text((35, 22), 'CAMS Node Dynamics — Evidentiary Floor Dashboard', fill='black', font=font_title)
draw.text((35, 88), f'36 societies    Common mode {common_fraction*100:.1f}%    Relational {relational_fraction*100:.1f}%    LOOS pair accuracy {mean_pair*100:.1f}%', fill='black', font=font_metric)
draw.text((35, 128), f'Mean held-out rank agreement ρ={mean_rho:.3f}    Above/below-unit classification {mean_unit*100:.1f}%', fill='black', font=font_small)
draw.text((35, 157), 'β = response to the mean annualised change of the other seven nodes. Prior hard-null validation: 0/5,000 relabelled datasets matched the observed held-out architecture.', fill='black', font=font_small)

positions = [(0, header_h), (cell_w, header_h), (0, header_h + cell_h), (cell_w, header_h + cell_h)]
for im, pos in zip(images, positions):
    canvas.paste(im, pos)

canvas.save(OUT / 'cams_node_dynamics_dashboard.png')

print(summary.round(3))
print('\nCommon mode:', round(common_fraction, 4))
print('Relational:', round(relational_fraction, 4))
print('Mean LOOS pair accuracy:', round(mean_pair, 4))
print('Mean LOOS Spearman:', round(mean_rho, 4))
