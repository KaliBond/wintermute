import csv, sys
from collections import defaultdict

def load(path):
    eras = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            eras[(row['entity'], row['era_id'])].append({
                'node': row['node'], 'Cog': float(row['Cog']),
                'Aff': float(row['Aff']), 'sigma': float(row['sigma']),
                'rank': float(row['rank']),
                'ys': int(row['year_start']), 'ye': int(row['year_end'])})
    return eras

def show(path):
    eras = load(path)
    def key(e):
        return (e[0][0], e[0][1])
    # sort: entity in given order, era by roman numeral value
    def roman(r):
        vals = {'I':1,'V':5,'X':10,'L':50}
        n = 0
        for i,c in enumerate(r):
            v = vals[c]
            n += -v if i+1 < len(r) and vals[r[i+1]] > v else v
        return n
    for (ent, eid), rows in sorted(eras.items(), key=lambda e: (e[0][0], roman(e[0][1]))):
        rows_s = sorted(rows, key=lambda r: -r['sigma'])
        ys, ye = rows[0]['ys'], rows[0]['ye']
        allneg = all(r['sigma'] < 0 for r in rows)
        print(f"== {ent} {eid} ({ys}-{ye})" + ("  [ALL-NEGATIVE]" if allneg else ""))
        for r in rows_s:
            print(f"  {r['node']:<9} sigma={r['sigma']:>9.3f}  Cog={r['Cog']:>7.2f}  Aff={r['Aff']:>7.2f}  rank={r['rank']:.2f}")
        print()

show(sys.argv[1])
