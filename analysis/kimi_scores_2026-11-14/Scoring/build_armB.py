# Build Arm B output JSONL: validate choices against the shortlist, write, hash, append ledger line.
import json, hashlib, csv, sys
from datetime import datetime

MODEL = "Kimi K2.5 (Kimi Work desktop runtime, Moonshot AI)"
ZIP_SHA = "dceee12e89c1b07b9f63e4ce26836f51fb231089b621afc4f9d8f819ab73aae9"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# shortlist lookup
short = {}
with open("armB/era_shortlist.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        short[(row["entity"], row["era_id"])] = (int(row["year_start"]), int(row["year_end"]))

# blind ids present
blind = {}
with open("armB/pockets_blind.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        blind[r["blind_id"]] = r["entity"]

import armB_choices
G = armB_choices.G
problems = []
seen = set()
for bid, ent, eid, ys, ye, conf, reason in G:
    if bid not in blind: problems.append(f"{bid}: not in blind set")
    elif blind[bid] != ent: problems.append(f"{bid}: entity {ent} != {blind[bid]}")
    if (ent, eid) not in short: problems.append(f"{bid}: {ent} {eid} not in shortlist")
    elif short[(ent, eid)] != (ys, ye): problems.append(f"{bid}: years {ys}-{ye} mismatch shortlist {short[(ent,eid)]}")
    if conf not in {"immediate","recognisable","broad","miss"}: problems.append(f"{bid}: bad confidence")
    if bid in seen: problems.append(f"{bid}: duplicate")
    seen.add(bid)
missing = set(blind) - seen
if missing: problems.append(f"missing blind ids: {sorted(missing)}")
if problems:
    print("VALIDATION FAIL"); [print(" ", p) for p in problems]; sys.exit(1)

ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M") + " AEDT"
with open("armB_output.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for bid, ent, eid, ys, ye, conf, reason in G:
        rec = {"blind_id": bid, "entity": ent, "chosen_era_id": eid,
               "chosen_year_start": ys, "chosen_year_end": ye,
               "confidence": conf, "brief_reason": reason}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

out_sha = sha256("armB_output.jsonl")
print(f"records: {len(G)}")
print(f"output sha256: {out_sha}")
print(f"timestamp: {ts}")

with open("ledger_armB.csv", "w", encoding="utf-8", newline="\n") as f:
    f.write("model,arm,entities,input_sha,output_sha,timestamp_aedt,operator_notes\n")
    f.write(f'"{MODEL}",B,CORE+LIMIT+EXTENSION,{ZIP_SHA},{out_sha},{ts},'
            '"Blind identification from pocket topology and closed shortlist only; no web; confidence marked honestly; choices unedited after hash"\n')
print("ledger_armB.csv written")
