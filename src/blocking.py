"""
blocking.py
-----------
Stage 3: Blocking / Candidate Generation

Generates high-recall candidate pairs (S1 -> S2, S1 -> S3) using an inverted-
index approach.  NO brute-force S1×S2 comparisons are performed.

Strategy design is based on signal analysis of 962 sampled true-match pairs:
  exact_name_norm   : 267 / 962  (27.8%)  — exact normalized name match
  name_token_overlap: 512 / 962  (53.2%)  — at least one meaningful name token shared
  name_prefix4_match:  58 / 962  (6.0%)   — first 4 chars match (covers word-order variants)
  addr_num_overlap  : 100 / 962  (10.4%)  — shared numeric address token
  addr_token_overlap:  25 / 962  (2.6%)   — shared long address text token
  no_signal         :   0 / 962  (0.0%)   — ***every sampled true pair hit at least one signal***

Union of ALL five strategies maximises recall with manageable candidate sets.

Blocking rules (applied within the same country):
  B1 — exact normalized name
  B2 — name token overlap (tokens ≥ 4 chars, excluding suffix stopwords)
  B3 — name prefix-4 chars (handles word-order differences)
  B4 — numeric address token overlap (street numbers, PIN codes — very distinctive)
  B5 — long address text token overlap (tokens ≥ 5 chars, excluding type words)

Country matching:
  Always required across all rules.  Country is open-set — no hard-coding.

Memory design:
  - Build ONE inverted index per rule per source (S2, S3).  Indexes held in RAM.
  - Stream S1 in chunks to look up candidates — never hold all S1 in RAM at once.
  - i3 dev mode: use sample data (~50k S1 + cherry-picked S2/S3 rows).
  - SageMaker full mode: full 2.2M/5.0M/5.3M rows.

Estimated RAM (full mode, both sources combined):
  Name token index (S2+S3):  ~10M entries × ~30 bytes  ≈ 300 MB
  Addr num  index (S2+S3):   ~  5M entries × ~30 bytes  ≈ 150 MB
  Addr text index (S2+S3):   ~  5M entries × ~30 bytes  ≈ 150 MB
  Prefix-4  index (S2+S3):   ~  5M entries × ~30 bytes  ≈ 150 MB
  Total estimate: ~750 MB — safe on SageMaker; borderline on i3 for full data.
  For i3 dev: use sample mode (see generate_candidates_sample()).
"""

import os
import sys
import csv
import json
import pandas as pd
from collections import defaultdict

# ── Normalization (Stage 2) ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from normalization import normalize_name, normalize_address

# ── Constants ─────────────────────────────────────────────────────────────────

# Tokens to exclude from name-token blocking — they are ubiquitous and
# would create massive, low-signal candidate explosions.
NAME_STOPWORDS = frozenset([
    'llc', 'ltd', 'pvt', 'inc', 'corp', 'co', 'and', 'the', 'of', 'for',
    'a', 'an', 'at', 'by', 'in', 'to', 'or', 'on', 'is', 'as',
    'plc', 'lp', 'llp', 'pc', 'pa', 'na', 'nv',
    'प्राइवेट', 'लिमिटेड', 'services', 'center', 'group', 'care', 'health', 
    'holdings', 'associates', 'partners', 'india', 'solutions', 'enterprises',
    'international', 'technologies', 'private', 'limited', 'company'
])

# Tokens to exclude from address-text-token blocking
ADDR_STOPWORDS = frozenset([
    'st', 'rd', 'ave', 'ln', 'ct', 'dr', 'apt', 'hwy', 'bldg',
    'near', 'main', 'road', 'post', 'block', 'sector', 'nagar',
    'floor', 'no', 'plot', 'shop', 'unit', 'suite', 'complex',
    'house', 'tower', 'vihar', 'colony', 'marg', 'society',
    'maharashtra', 'delhi', 'pradesh', 'mumbai', 'uttar', 'karnataka',
    'bangalore', 'महाराष्ट्र', 'tamil', 'bengal', 'gujarat', 'kolkata',
    'telangana', 'hyderabad', 'दिल्ली', 'county', 'south', 'haryana', 
    'chennai', 'saint', 'rajasthan', 'andhra', 'प्रदेश', 'north', 'ground',
    'kerala', 'village', 'township'
])

MIN_NAME_TOKEN_LEN  = 4   # ignore name tokens shorter than this
MIN_ADDR_TEXT_LEN   = 5   # ignore address text tokens shorter than this
MIN_ADDR_NUM_LEN    = 2   # ignore numeric address tokens shorter than this (e.g. "1")
NAME_PREFIX_LEN     = 4   # length of name prefix for B3 rule


# ── Key extractors — pure functions, testable in isolation ────────────────────

def name_token_keys(country: str, norm_name: str) -> set:
    """B2: (country, token) for each meaningful name token."""
    if not country or not norm_name:
        return set()
    return {
        (country, t)
        for t in norm_name.split()
        if len(t) >= MIN_NAME_TOKEN_LEN and t not in NAME_STOPWORDS
    }

def name_prefix_keys(country: str, norm_name: str) -> set:
    """B3: (country, 'pfx:' + first N chars of normalized name)."""
    if not country or not norm_name or len(norm_name) < NAME_PREFIX_LEN:
        return set()
    return {(country, 'pfx:' + norm_name[:NAME_PREFIX_LEN])}

def addr_num_keys(country: str, norm_addr: str) -> set:
    """B4: (country, 'num:' + digit_token) for each numeric address token."""
    if not country or not norm_addr:
        return set()
    return {
        (country, 'num:' + t)
        for t in norm_addr.split()
        if t.isdigit() and len(t) >= MIN_ADDR_NUM_LEN
    }

def addr_text_keys(country: str, norm_addr: str) -> set:
    """B5: (country, 'atxt:' + token) for distinctive address text tokens."""
    if not country or not norm_addr:
        return set()
    return {
        (country, 'atxt:' + t)
        for t in norm_addr.split()
        if not t.isdigit() and len(t) >= MIN_ADDR_TEXT_LEN and t not in ADDR_STOPWORDS
    }

def all_keys_for_row(country: str, norm_name: str, norm_addr: str) -> set:
    """Union of all blocking keys for a single entity row."""
    keys = set()
    # B2 name tokens
    keys |= name_token_keys(country, norm_name)
    # B3 name prefix
    keys |= name_prefix_keys(country, norm_name)
    # B4 addr numeric tokens
    keys |= addr_num_keys(country, norm_addr)
    # B5 addr text tokens
    keys |= addr_text_keys(country, norm_addr)
    return keys


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(df: pd.DataFrame, verbose: bool = True, max_key_freq: int = 15_000) -> dict:
    """
    Build blocking index from a source DataFrame (S2 or S3).

    Returns dict: blocking_key -> frozenset of entity_ids
    Memory: O(unique_keys × avg_ids_per_key).

    NOTE: B1 (exact norm name) is handled separately to avoid polluting the
    general token index with full-string keys.
    max_key_freq dynamically prunes high-frequency broad keys (e.g. state names, common words).
    """
    key_to_ids = defaultdict(set)
    exact_name_index = defaultdict(set)  # B1: (country, norm_name) -> ids

    for i, (_, row) in enumerate(df.iterrows()):
        eid     = row['entity_id']
        country = str(row.get('country', '')).lower().strip()
        nn      = row.get('business_name_norm')
        if pd.isna(nn): nn = ''
        na      = row.get('business_address_norm')
        if pd.isna(na): na = ''

        # B1: exact normalized name
        if country and nn:
            exact_name_index[(country, nn)].add(eid)

        # B2–B5
        for key in all_keys_for_row(country, nn, na):
            key_to_ids[key].add(eid)

        if verbose and (i + 1) % 200_000 == 0:
            print(f"  indexed {i+1:,} rows, {len(key_to_ids):,} keys so far")

    # Dynamic pruning: drop tokens linked to > max_key_freq entities.
    pruned_token_idx = {k: frozenset(v) for k, v in key_to_ids.items() if len(v) <= max_key_freq}

    if verbose:
        dropped = len(key_to_ids) - len(pruned_token_idx)
        print(f"  pruned {dropped:,} high-frequency keys (> {max_key_freq} items)")

    # Convert to frozensets to save a little memory
    return (
        {k: frozenset(v) for k, v in exact_name_index.items()},
        pruned_token_idx,
    )


# ── Candidate lookup for a single S1 row ─────────────────────────────────────

def get_candidates_for_row(row, exact_idx: dict, token_idx: dict) -> set:
    """Return set of candidate entity_ids for one S1 row."""
    country = str(row.get('country', '')).lower().strip()
    nn      = row.get('business_name_norm')
    if pd.isna(nn): nn = ''
    na      = row.get('business_address_norm')
    if pd.isna(na): na = ''

    candidates = set()

    # B1: exact name match
    key_b1 = (country, nn)
    if key_b1 in exact_idx:
        candidates |= exact_idx[key_b1]

    # B2–B5 via unified token index
    for key in all_keys_for_row(country, nn, na):
        if key in token_idx:
            candidates |= token_idx[key]

    return candidates


# ── Full candidate generation (chunked S1 stream) ────────────────────────────

def generate_candidates(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_candidates_path: str,
    chunk_size: int = 100_000,
    verbose: bool = True,
) -> dict:
    """
    Full candidate generation pipeline (for SageMaker / full dataset).

    Memory profile:
      - Loads full S2 then full S3 to build indexes (one at a time).
        Peak during index build: ~1.5–2 GB (S2 or S3 + index).
        i3 WARNING: Running this on full S2/S3 may push RAM to 3+ GB.
        Recommended: run on SageMaker for full dataset.
      - Streams S1 in chunks — adds only ~150 MB per chunk.

    Returns dict with summary statistics.
    """
    stats = {}

    # ----- Build S2 index -----
    if verbose: print("Building S2 index (full load)...")
    s2_df = pd.read_csv(s2_path, sep='\t', dtype=str, keep_default_na=False)
    s2_df['business_name_norm']    = s2_df['business_name'].apply(normalize_name)
    s2_df['business_address_norm'] = s2_df['business_address'].apply(normalize_address)
    s2_exact_idx, s2_token_idx = build_index(s2_df, verbose)
    del s2_df
    if verbose: print(f"  S2 index: {len(s2_exact_idx):,} exact keys, {len(s2_token_idx):,} token keys")

    # ----- Build S3 index -----
    if verbose: print("Building S3 index (full load)...")
    s3_df = pd.read_csv(s3_path, sep='\t', dtype=str, keep_default_na=False)
    s3_df['business_name_norm']    = s3_df['business_name'].apply(normalize_name)
    s3_df['business_address_norm'] = s3_df['business_address'].apply(normalize_address)
    s3_exact_idx, s3_token_idx = build_index(s3_df, verbose)
    del s3_df
    if verbose: print(f"  S3 index: {len(s3_exact_idx):,} exact keys, {len(s3_token_idx):,} token keys")

    # ----- Stream S1 and write candidates -----
    total_s1 = 0
    total_cands = 0
    zero_cand_s1 = 0
    cand_counts = []

    os.makedirs(os.path.dirname(out_candidates_path), exist_ok=True)

    with open(out_candidates_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])

        reader = pd.read_csv(s1_path, sep='\t', dtype=str,
                             keep_default_na=False, chunksize=chunk_size)
        for s1_chunk in reader:
            s1_chunk['business_name_norm']    = s1_chunk['business_name'].apply(normalize_name)
            s1_chunk['business_address_norm'] = s1_chunk['business_address'].apply(normalize_address)

            for _, row in s1_chunk.iterrows():
                s1id = row['entity_id']
                cands = (
                    get_candidates_for_row(row, s2_exact_idx, s2_token_idx) |
                    get_candidates_for_row(row, s3_exact_idx, s3_token_idx)
                )
                cands.discard(s1id)   # safety: remove self if present

                cand_list = sorted(cands)
                writer.writerow([s1id, ','.join(cand_list)])

                n = len(cand_list)
                total_s1     += 1
                total_cands  += n
                cand_counts.append(n)
                if n == 0: zero_cand_s1 += 1

            if verbose:
                print(f"  S1 processed: {total_s1:,}", end='\r')

    if verbose: print(f"\nDone. Total S1: {total_s1:,}")

    import statistics
    stats = {
        "total_s1": total_s1,
        "total_candidate_pairs": total_cands,
        "zero_candidate_s1": zero_cand_s1,
        "pct_zero_candidate": round(100 * zero_cand_s1 / max(total_s1, 1), 2),
        "avg_candidates_per_s1": round(total_cands / max(total_s1, 1), 2),
        "median_candidates_per_s1": statistics.median(cand_counts) if cand_counts else 0,
        "max_candidates_for_s1": max(cand_counts) if cand_counts else 0,
    }
    return stats


# ── Sample mode — safe on i3 laptop ──────────────────────────────────────────

def generate_candidates_from_dfs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate candidates from pre-loaded DataFrames (used in sample/dev mode).
    DataFrames must already have business_name_norm and business_address_norm columns.

    Returns DataFrame with columns: source1_entity_id, candidate_entity_ids (list)
    """
    if verbose: print("Building S2 index...")
    s2_exact, s2_tok = build_index(s2_df, verbose=False)
    if verbose: print(f"  S2: {len(s2_exact):,} exact, {len(s2_tok):,} token keys")

    if verbose: print("Building S3 index...")
    s3_exact, s3_tok = build_index(s3_df, verbose=False)
    if verbose: print(f"  S3: {len(s3_exact):,} exact, {len(s3_tok):,} token keys")

    if verbose: print("Looking up S1 candidates...")
    records = []
    for _, row in s1_df.iterrows():
        s1id = row['entity_id']
        cands = (
            get_candidates_for_row(row, s2_exact, s2_tok) |
            get_candidates_for_row(row, s3_exact, s3_tok)
        )
        cands.discard(s1id)
        records.append({'source1_entity_id': s1id, 'candidates': sorted(cands)})

    return pd.DataFrame(records)


# ── Recall evaluator ──────────────────────────────────────────────────────────

def evaluate_recall(
    candidates_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Evaluate blocking recall against ground truth.

      candidates_df: columns [source1_entity_id, candidates (list)]
      gt_df: columns [source1_entity_id, matched_list (list)]

    Returns dict with per-source and overall recall.
    """
    cand_map = candidates_df.set_index('source1_entity_id')['candidates'].to_dict()
    gt_map   = gt_df.set_index('source1_entity_id')['matched_list'].to_dict()

    # Only evaluate S1 entities present in both
    common = set(cand_map) & set(gt_map)

    hits_s2 = miss_s2 = hits_s3 = miss_s3 = 0
    missed_examples = []

    for s1id in common:
        true_matches = set(gt_map[s1id])
        cands        = set(cand_map[s1id])
        for mid in true_matches:
            is_hit = mid in cands
            if mid.startswith('S2-'):
                if is_hit: hits_s2 += 1
                else:
                    miss_s2 += 1
                    if len(missed_examples) < 20:
                        missed_examples.append((s1id, mid, 'S2'))
            elif mid.startswith('S3-'):
                if is_hit: hits_s3 += 1
                else:
                    miss_s3 += 1
                    if len(missed_examples) < 20:
                        missed_examples.append((s1id, mid, 'S3'))

    total_s2 = hits_s2 + miss_s2
    total_s3 = hits_s3 + miss_s3
    total    = total_s2 + total_s3

    recall_s2  = hits_s2 / max(total_s2, 1)
    recall_s3  = hits_s3 / max(total_s3, 1)
    recall_all = (hits_s2 + hits_s3) / max(total, 1)

    stats = {
        "s1_entities_evaluated": len(common),
        "s2_true_matches": total_s2, "s2_hits": hits_s2, "s2_recall": round(recall_s2, 4),
        "s3_true_matches": total_s3, "s3_hits": hits_s3, "s3_recall": round(recall_s3, 4),
        "total_true_matches": total, "total_hits": hits_s2 + hits_s3,
        "overall_recall": round(recall_all, 4),
        "missed_examples": missed_examples,
    }

    if verbose:
        print(f"\n=== RECALL REPORT ===")
        print(f"  S1 entities evaluated : {len(common):,}")
        print(f"  S2 recall             : {hits_s2}/{total_s2} = {recall_s2:.4f}")
        print(f"  S3 recall             : {hits_s3}/{total_s3} = {recall_s3:.4f}")
        print(f"  Overall recall        : {(hits_s2+hits_s3)}/{total} = {recall_all:.4f}")
        print(f"  Missed true matches   : {miss_s2 + miss_s3}")

    return stats
