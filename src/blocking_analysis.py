"""
blocking_analysis.py — analyse noise patterns to decide blocking strategies.
Runs on a small reproducible sample — safe on i3 laptop.

Memory usage: loads ~100k rows from GT + cherry-picked S2/S3 rows.
Peak RAM: < 300 MB.
"""
import pandas as pd
import json

DATA = "dataset/train/"
OUT  = "output/"

# ── imports from Stage 2 ──────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, "src")
from normalization import normalize_name, normalize_address

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load GT sample and cherry-pick true-match row IDs
# ─────────────────────────────────────────────────────────────────────────────
print("Loading ground truth sample...")
gt = pd.read_csv(DATA + "train_ground_truth.tsv", sep="\t", dtype=str,
                 keep_default_na=False)
gt["matched_list"] = gt["matched_entity_ids"].apply(
    lambda x: [i.strip() for i in x.split(",") if i.strip()] if x.strip() else []
)
gt["match_count"] = gt["matched_list"].apply(len)

# Keep only GT rows that have at least one match
gt_with_match = gt[gt["match_count"] > 0].sample(n=10_000, random_state=42)

# Collect specific S1, S2, S3 IDs needed
s1_ids = set(gt_with_match["source1_entity_id"].tolist())
s2_ids, s3_ids = set(), set()
for row in gt_with_match["matched_list"]:
    for mid in row:
        if mid.startswith("S2-"): s2_ids.add(mid)
        elif mid.startswith("S3-"): s3_ids.add(mid)

print(f"Sample GT pairs: {len(gt_with_match):,}")
print(f"Unique S1 IDs: {len(s1_ids):,}  |  S2 IDs: {len(s2_ids):,}  |  S3 IDs: {len(s3_ids):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load only those specific rows + 20k background rows for context
# ──────────────────────────────────────────────────
# ───────────────────────────
def load_with_norm(path, target_ids, bg_size, cols=None):
    """Load specific rows + background rows; apply normalization."""
    rows = []
    bg  = []
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         chunksize=100_000)
    for chunk in reader:
        mask = chunk["entity_id"].isin(target_ids)
        rows.append(chunk[mask])
        if len(bg) < bg_size:
            bg.append(chunk[~mask].head(bg_size - sum(len(b) for b in bg)))
        if len(rows) > 0 and all(chunk["entity_id"].isin(target_ids).sum() == 0
                                  for chunk in [chunk]):
            pass
    df = pd.concat(rows + bg, ignore_index=True).drop_duplicates("entity_id")
    df["business_name_norm"]    = df["business_name"].apply(normalize_name)
    df["business_address_norm"] = df["business_address"].apply(normalize_address)
    return df

print("\nLoading source files (target rows + 20k background each)...")
s1_df = load_with_norm(DATA + "train_source1.tsv", s1_ids, bg_size=20_000)
s2_df = load_with_norm(DATA + "train_source2.tsv", s2_ids, bg_size=20_000)
s3_df = load_with_norm(DATA + "train_source3.tsv", s3_ids, bg_size=20_000)

print(f"S1 loaded: {len(s1_df):,} rows   S2: {len(s2_df):,}   S3: {len(s3_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Analyse what signals are available in true-match pairs
# ─────────────────────────────────────────────────────────────────────────────
s1_idx = s1_df.set_index("entity_id")
s2_idx = s2_df.set_index("entity_id")
s3_idx = s3_df.set_index("entity_id")

def get_row(eid):
    if eid.startswith("S2-"): idx = s2_idx
    elif eid.startswith("S3-"): idx = s3_idx
    else: return None
    return idx.loc[eid] if eid in idx.index else None

# Analyse signal availability on 500 random true pairs
sample_pairs = gt_with_match.sample(500, random_state=7)
results = {"same_country": 0, "exact_name_norm": 0, "name_token_overlap": 0,
           "addr_token_overlap": 0, "addr_num_overlap": 0,
           "name_prefix4_match": 0, "no_signal": 0, "total": 0}

for _, row in sample_pairs.iterrows():
    s1id = row["source1_entity_id"]
    if s1id not in s1_idx.index: continue
    s1r = s1_idx.loc[s1id]

    for mid in row["matched_list"][:2]:                 # check first 2 matches
        mr = get_row(mid)
        if mr is None: continue
        results["total"] += 1

        sc = str(s1r.get("country","")).lower()
        mc = str(mr.get("country","")).lower()
        sn = s1r.get("business_name_norm") or ""
        mn = mr.get("business_name_norm") or ""
        sa = s1r.get("business_address_norm") or ""
        ma = mr.get("business_address_norm") or ""

        same_c = (sc == mc)
        if same_c: results["same_country"] += 1

        # exact normalized name
        if same_c and sn and mn and sn == mn:
            results["exact_name_norm"] += 1; continue

        # name token overlap (ignoring suffix tokens)
        SUFFIX = {"llc","ltd","pvt","inc","corp","co","and"}
        stoks = {t for t in sn.split() if len(t) >= 4 and t not in SUFFIX}
        mtoks = {t for t in mn.split() if len(t) >= 4 and t not in SUFFIX}
        if same_c and stoks & mtoks:
            results["name_token_overlap"] += 1; continue

        # name prefix-4
        if same_c and sn[:4] and sn[:4] == mn[:4]:
            results["name_prefix4_match"] += 1; continue

        # address numeric token overlap
        snums = {t for t in sa.split() if t.isdigit() and len(t) >= 2}
        mnums = {t for t in ma.split() if t.isdigit() and len(t) >= 2}
        if same_c and snums & mnums:
            results["addr_num_overlap"] += 1; continue

        # address text token overlap
        ADDR_STOP = {"st","rd","ave","ln","ct","dr","apt","hwy","bldg"}
        satoks = {t for t in sa.split() if len(t) >= 5 and t not in ADDR_STOP}
        matoks = {t for t in ma.split() if len(t) >= 5 and t not in ADDR_STOP}
        if same_c and satoks & matoks:
            results["addr_token_overlap"] += 1; continue

        results["no_signal"] += 1

with open(OUT + "blocking_signal_analysis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n=== SIGNAL ANALYSIS on 500 sampled true pairs ===")
total = results["total"] or 1
for k, v in results.items():
    if k != "total":
        print(f"  {k:<25}: {v:>5}  ({100*v/total:.1f}%)")
print(f"  {'total':<25}: {results['total']}")
