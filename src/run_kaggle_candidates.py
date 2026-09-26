"""
run_kaggle_candidates.py
------------------------
This script runs the Stage 3 blocking pipeline across the FULL dataset
using Kaggle's memory. It saves exactly what is needed for the ML Model:
1. output/train_candidate_pairs.tsv
2. output/test_candidate_pairs.tsv
"""
import sys, os, time, gc
import pandas as pd
from collections import defaultdict, Counter

sys.path.insert(0, "src")
from normalization import normalize_name, normalize_address
from blocking import all_keys_for_row

def count_freq(path):
    freq = Counter()
    reader = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=500_000)
    for chunk in reader:
        for eid, ctr, nn, na in zip(chunk['entity_id'], chunk['country'], 
                                    chunk['business_name'].apply(normalize_name), 
                                    chunk['business_address'].apply(normalize_address)):
            for k_type, k_val in all_keys_for_row(ctr, nn, na):
                freq[k_val] += 1
    return freq

def build_index(path, ok_keys):
    exact_idx, tok_idx = defaultdict(set), defaultdict(set)
    reader = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=500_000)
    for chunk in reader:
        for eid, ctr, nn, na in zip(chunk['entity_id'], chunk['country'], 
                                    chunk['business_name'].apply(normalize_name), 
                                    chunk['business_address'].apply(normalize_address)):
            for k_type, k_val in all_keys_for_row(ctr, nn, na):
                if k_val in ok_keys:
                    if k_type == 'exact': exact_idx[k_val].add(eid)
                    else: tok_idx[k_val].add(eid)
    return exact_idx, tok_idx

def generate_candidates(s1_path, exact_idx, tok_idx, outfile):
    out_lines = []
    reader = pd.read_csv(s1_path, sep='\t', dtype=str, keep_default_na=False, chunksize=200_000)
    
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for chunk in reader:
            for eid, ctr, nn, na in zip(chunk['entity_id'], chunk['country'], 
                                        chunk['business_name'].apply(normalize_name), 
                                        chunk['business_address'].apply(normalize_address)):
                cands = set()
                ctr_str = str(ctr).lower().strip() if str(ctr) != 'nan' else ''
                if ctr_str:
                    for k_type, k_val in all_keys_for_row(ctr_str, nn, na):
                        if k_type == 'exact' and k_val in exact_idx: cands |= exact_idx[k_val]
                        elif k_val in tok_idx: cands |= tok_idx[k_val]
                
                cands.discard(eid)
                if len(cands) > 150:
                    cands = set(list(cands)[:150])
                cand_str = ",".join(sorted(cands))
                f.write(f"{eid}\t{cand_str}\n")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    THRESHOLD = 1000
    
    print("=== TRAIN SET BLOCKING ===")
    print("Pass 1: S2 Frequencies...")
    s2_freq = count_freq("dataset/train/train_source2.tsv")
    print("Pass 1: S3 Frequencies...")
    s3_freq = count_freq("dataset/train/train_source3.tsv")
    
    ok_keys = {k for k,v in (s2_freq + s3_freq).items() if v <= THRESHOLD}
    del s2_freq, s3_freq; gc.collect()
    
    print("Pass 2: Building Index...")
    exact_s2, tok_s2 = build_index("dataset/train/train_source2.tsv", ok_keys)
    exact_s3, tok_s3 = build_index("dataset/train/train_source3.tsv", ok_keys)
    
    exact_idx = {**exact_s2, **exact_s3}
    tok_idx = {**tok_s2, **tok_s3}
    del exact_s2, tok_s2, exact_s3, tok_s3; gc.collect()
    
    print("Generating train_candidate_pairs.tsv...")
    generate_candidates("dataset/train/train_source1.tsv", exact_idx, tok_idx, "output/train_candidate_pairs.tsv")
    print("Done generating train pairs!\n")
    
    # Optional: Once we have test set, we do the exact same (or build test index and sweep test S1)
