# Build Arm A JSONL outputs, validate against packet rules, hash, and write ledger.
import csv, json, hashlib, re, sys
from datetime import datetime

MODEL = "Kimi K2.5 (Kimi Work desktop runtime, Moonshot AI)"
NODES = {"Helm","Lore","Craft","Hands","Flow","Stewards","Shield","Archive"}
BANNED = ["wikipedia","britannica","–","—"]  # no em/en dashes, no encyclopaedia paraphrase markers

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_csv(path):
    eras = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["entity"], row["era_id"])
            if key not in eras:
                eras[key] = (int(row["year_start"]), int(row["year_end"]))
    return eras

def ts_aedt():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M") + " AEDT"

def build(pockets_mod, csv_path, out_path):
    spec = __import__(pockets_mod)
    expected = load_csv(csv_path)
    got = {(p[0], p[1]) for p in spec.P}
    missing = set(expected) - got
    extra = got - set(expected)
    if missing or extra:
        print("COVERAGE FAIL"); print("missing:", sorted(missing)); print("extra:", sorted(extra)); sys.exit(1)
    in_sha = sha256(csv_path)
    ts = ts_aedt()
    lines, problems = [], []
    for ent, eid, ys, ye, leaders, constraints, s1, s2, notes in spec.P:
        eys, eye = expected[(ent, eid)]
        if (ys, ye) != (eys, eye):
            problems.append(f"{ent} {eid}: years {ys}-{ye} != csv {eys}-{eye}")
        if not set(leaders) <= NODES or not set(constraints) <= NODES:
            problems.append(f"{ent} {eid}: unknown node name")
        pocket = s1 + " " + s2
        if len([s for s in re.split(r"(?<=[.!?]) ", pocket) if s.strip()]) != 2:
            problems.append(f"{ent} {eid}: not exactly two sentences")
        for b in BANNED:
            if b in pocket.lower():
                problems.append(f"{ent} {eid}: banned token {b!r}")
        rec = {
            "model": MODEL, "arm": "A", "entity": ent, "era_id": eid,
            "year_start": ys, "year_end": ye,
            "leaders": leaders, "constraints": constraints,
            "pocket": pocket, "notes": notes,
            "sha256_input_csv": in_sha,
            "sha256_this_output": "",
            "timestamp_aedt": ts,
        }
        lines.append(json.dumps(rec, ensure_ascii=False))
    if problems:
        print("VALIDATION FAIL"); [print(" ", p) for p in problems]; sys.exit(1)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    out_sha = sha256(out_path)
    print(f"{out_path}: {len(lines)} records")
    print(f"input  sha256: {in_sha}")
    print(f"output sha256: {out_sha}")
    print(f"timestamp: {ts}")
    return out_sha, in_sha, ts

sha_core, in_core, ts_core = build("pockets_core", "packet/era_means.csv", "armA_core.jsonl")
sha_dk, in_dk, ts_dk = build("pockets_denmark", "packet/era_means_denmark.csv", "armA_denmark.jsonl")

with open("ledger.csv", "w", encoding="utf-8", newline="\n") as f:
    f.write("model,arm,entity_set,sha256_input,sha256_output,timestamp_aedt,operator_notes\n")
    f.write(f'"{MODEL}",A,CORE,{in_core},{sha_core},{ts_core},"Arm A blind pockets from era means only; no web; no V or Bond; pockets unedited after hash"\n')
    f.write(f'"{MODEL}",A,LIMIT(Denmark),{in_dk},{sha_dk},{ts_dk},"Denmark sparse-input stress test; several identical tables carried identical pocket text"\n')
print("ledger.csv written")
