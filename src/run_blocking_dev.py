"""
run_blocking_dev.py
-------------------
Stage 3 development runner — safe on i3 laptop.

Loads a cherry-picked sample that guarantees true-match rows are present,
builds blocking indexes, generates candidates, and evaluates recall.

Peak RAM: ~300-400 MB (sample only).
Full-dataset run: see blocking.py generate_candidates() — use SageMaker.
"""
import sys, os, json, statistics
import pandas as pd

DATA = "dataset/train/"
OUT  = "output/"
sys.path.insert(0, "src")

from normalization import normalize_name, normalize_address
from blocking import generate_candidates_from_dfs, evaluate_recall

# ── 1. Load GT and cherry-pick a recall-testable sample ──────────────────────
print("Loading ground truth...")
gt = pd.read_csv(DATA + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt["matched_list"] = gt["matched_entity_ids"].apply(
    lambda x: [i.strip() for i in x.split(",") if i.strip()] if x.strip() else []
)

# Take 5000 S1 entities that have matches (for recall eval)
gt_with = gt[gt["matched_list"].apply(len) > 0].sample(n=5_000, random_state=42)
# Also take 500 singletons (for specificity check)
gt_zero = gt[gt["matched_list"].apply(len) == 0].sample(n=500, random_state=42)
gt_sample = pd.concat([gt_with, gt_zero], ignore_index=True)

# Collect exact IDs needed
s1_ids_needed = set(gt_sample["source1_entity_id"])
s2_ids_needed, s3_ids_needed = set(), set()
for ml in gt_with["matched_list"]:
    for mid in ml:
        if mid.startswith("S2-"): s2_ids_needed.add(mid)
        elif mid.startswith("S3-"): s3_ids_needed.add(mid)

print(f"  S1 IDs needed: {len(s1_ids_needed):,}")
print(f"  S2 IDs needed: {len(s2_ids_needed):,}")
print(f"  S3 IDs needed: {len(s3_ids_needed):,}")

# ── 2. Load cherry-picked rows + 10k background rows per source ───────────────
def smart_load(path, needed_ids, bg_size=10_000):
    target_rows, bg_rows = [], []
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         chunksize=200_000)
    for chunk in reader:
        hit  = chunk[chunk["entity_id"].isin(needed_ids)]
        miss = chunk[~chunk["entity_id"].isin(needed_ids)]
        target_rows.append(hit)
        remaining = bg_size - sum(len(b) for b in bg_rows)
        if remaining > 0:
            bg_rows.append(miss.head(remaining))
    df = pd.concat(target_rows + bg_rows, ignore_index=True)
    df = df.drop_duplicates("entity_id").reset_index(drop=True)
    df["business_name_norm"]    = df["business_name"].apply(normalize_name)
    df["business_address_norm"] = df["business_address"].apply(normalize_address)
    return df

print("\nLoading source files (cherry-picked + 10k background each)...")
s1_df = smart_load(DATA + "train_source1.tsv", s1_ids_needed, bg_size=10_000)
s2_df = smart_load(DATA + "train_source2.tsv", s2_ids_needed, bg_size=10_000)
s3_df = smart_load(DATA + "train_source3.tsv", s3_ids_needed, bg_size=10_000)
print(f"  S1: {len(s1_df):,} rows  |  S2: {len(s2_df):,}  |  S3: {len(s3_df):,}")

# ── 3. Generate candidates ────────────────────────────────────────────────────
print("\nGenerating candidates (sample mode)...")
cands_df = generate_candidates_from_dfs(s1_df, s2_df, s3_df, verbose=True)

# Candidate statistics
cand_counts = cands_df["candidates"].apply(len)
print("\n=== CANDIDATE STATISTICS (sample) ===")
print(f"  S1 entities             : {len(cands_df):,}")
print(f"  Total candidate pairs   : {cand_counts.sum():,}")
print(f"  Avg candidates/S1       : {cand_counts.mean():.1f}")
print(f"  Median candidates/S1    : {cand_counts.median():.0f}")
print(f"  Max candidates for a S1 : {cand_counts.max()}")
print(f"  Zero-candidate S1       : {(cand_counts==0).sum()} ({100*(cand_counts==0).mean():.1f}%)")

# ── 4. Evaluate recall ────────────────────────────────────────────────────────
recall_stats = evaluate_recall(cands_df, gt_sample, verbose=True)

# ── 5. Inspect missed true matches ───────────────────────────────────────────
print("\n=== MISSED TRUE MATCH EXAMPLES ===")
s1_idx = s1_df.set_index("entity_id")
s2_idx = s2_df.set_index("entity_id")
s3_idx = s3_df.set_index("entity_id")

def get_record(eid):
    if eid in s1_idx.index: return s1_idx.loc[eid]
    if eid.startswith("S2-") and eid in s2_idx.index: return s2_idx.loc[eid]
    if eid.startswith("S3-") and eid in s3_idx.index: return s3_idx.loc[eid]
    return None

shown = 0
with open(OUT + "blocking_missed_examples.txt", "w", encoding="utf-8") as fw:
    for s1id, mid, src in recall_stats["missed_examples"]:
        if shown >= 10: break
        r1 = get_record(s1id)
        rm = get_record(mid)
        if r1 is None or rm is None: continue
        fw.write(f"\n  S1 [{s1id}]: '{r1['business_name']}' | '{r1['business_address']}' | {r1['country']}\n")
        fw.write(f"  {src} [{mid}]: '{rm['business_name']}' | '{rm['business_address']}' | {rm['country']}\n")
        # diagnose
        nn1 = r1.get("business_name_norm")
        if pd.isna(nn1): nn1 = ""
        nn2 = rm.get("business_name_norm")
        if pd.isna(nn2): nn2 = ""
        na1 = r1.get("business_address_norm")
        if pd.isna(na1): na1 = ""
        na2 = rm.get("business_address_norm")
        if pd.isna(na2): na2 = ""
        tok1 = set(t for t in nn1.split() if len(t)>=4 and t not in {"llc","ltd","pvt","inc","corp","co","and"})
        tok2 = set(t for t in nn2.split() if len(t)>=4 and t not in {"llc","ltd","pvt","inc","corp","co","and"})
        num1 = set(t for t in na1.split() if t.isdigit() and len(t)>=2)
        num2 = set(t for t in na2.split() if t.isdigit() and len(t)>=2)
        fw.write(f"    Name tokens overlap  : {tok1 & tok2} (S1:{tok1}, Match:{tok2})\n")
        fw.write(f"    Addr num overlap     : {num1 & num2} (S1:{num1}, Match:{num2})\n")
        fw.write(f"    Prefix match         : {'YES' if nn1[:4]==nn2[:4] and nn1[:4] else 'NO'} ('{nn1[:4]}' vs '{nn2[:4]}')\n")
        shown += 1

# ── 6. Save results ───────────────────────────────────────────────────────────
summary = {
    "candidate_stats": {
        "total_s1": int(len(cands_df)),
        "total_candidate_pairs": int(cand_counts.sum()),
        "avg_candidates_per_s1": float(round(cand_counts.mean(), 1)),
        "median_candidates_per_s1": float(cand_counts.median()),
        "max_candidates_for_s1": int(cand_counts.max()),
        "zero_candidate_s1": int((cand_counts == 0).sum()),
        "pct_zero_candidate": float(round(100 * (cand_counts == 0).mean(), 1)),
    },
    "recall_stats": {k: v for k, v in recall_stats.items() if k != "missed_examples"},
}
with open(OUT + "blocking_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)
print(f"\nResults saved to {OUT}blocking_results.json")

