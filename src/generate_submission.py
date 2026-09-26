import sys
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_engineering import engineer_features
from run_kaggle_candidates import count_freq, build_index, generate_candidates
import os

def generate_test_candidates():
    print("=== TEST SET BLOCKING ===")
    print("Extracting Token Frequencies from Test S2 and S3...")
    s2_freq = count_freq("dataset/test/test_source2.tsv")
    s3_freq = count_freq("dataset/test/test_source3.tsv")
    
    # Use strict freq cutoff to protect memory
    ok_keys = {k for k, v in (s2_freq + s3_freq).items() if v <= 1000}
    del s2_freq, s3_freq; gc.collect()
    
    print("Building Test Indexes...")
    exact_s2, tok_s2 = build_index("dataset/test/test_source2.tsv", ok_keys)
    exact_s3, tok_s3 = build_index("dataset/test/test_source3.tsv", ok_keys)
    
    exact_idx = {**exact_s2, **exact_s3}
    tok_idx = {**tok_s2, **tok_s3}
    del exact_s2, tok_s2, exact_s3, tok_s3; gc.collect()
    
    print("Outputting Candidate Pairs...")
    # Generate final raw candidates (Needed for submission packet)
    generate_candidates("dataset/test/test_source1.tsv", exact_idx, tok_idx, "output/candidate_pairs.tsv")
    print("Test candidates generated!\n")

def infer_test_matches():
    print("=== MODEL INFERENCE ===")
    
    with open('output/model_metadata.txt', 'r') as f:
        best_thresh = float(f.read().strip())
    print(f"Loaded Optimal Threshold: {best_thresh}")
    
    model = lgb.Booster(model_file='output/lgbm_model.txt')

    print("Loading Test Datasets into RAM for fast lookup (10GB RAM peak)...")
    # Load once, keep purely strings to minimize memory
    s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')
    print("S1 Loaded.")
    s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')
    print("S2 Loaded.")
    s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')
    print("S3 Loaded.")

    # Open output file
    out_f = open('output/matching_results.tsv', 'w', encoding='utf-8')
    out_f.write("source1_entity_id\tmatched_entity_ids\n")
    
    print("Processing candidate pairs in chunks...")
    reader = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=100_000)
    
    for i, chunk in enumerate(reader):
        all_s1_ids = chunk['source1_entity_id'].unique()
        chunk['cand_list'] = chunk['candidate_entity_ids'].apply(
            lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
        )
        pairs_df = chunk.explode('cand_list').rename(columns={'cand_list': 'target_id'}).dropna(subset=['target_id'])
        
        if len(pairs_df) == 0:
            for s1_id in all_s1_ids: out_f.write(f"{s1_id}\t\n")
            continue
            
        pairs_df = pairs_df[['source1_entity_id', 'target_id']]
        
        # Build features for this chunk dataframe
        df = pairs_df.copy()
        
        # Map S1 data
        df['s1_name'] = df['source1_entity_id'].map(s1['business_name']).fillna('')
        df['s1_addr'] = df['source1_entity_id'].map(s1['business_address']).fillna('')
        df['s1_country'] = df['source1_entity_id'].map(s1['country']).fillna('')
        
        # Map target data (checking if it starts with S2 or S3)
        def get_t_name(tid): return s2.at[tid, 'business_name'] if tid.startswith('S2-') and tid in s2.index else (s3.at[tid, 'business_name'] if tid in s3.index else '')
        def get_t_addr(tid): return s2.at[tid, 'business_address'] if tid.startswith('S2-') and tid in s2.index else (s3.at[tid, 'business_address'] if tid in s3.index else '')
        def get_t_ctry(tid): return s2.at[tid, 'country'] if tid.startswith('S2-') and tid in s2.index else (s3.at[tid, 'country'] if tid in s3.index else '')
        
        df['business_name'] = df['target_id'].map(get_t_name)
        df['business_address'] = df['target_id'].map(get_t_addr)
        df['country'] = df['target_id'].map(get_t_ctry)
        
        s1_df = df[['s1_name', 's1_addr', 's1_country']].rename(columns={'s1_name':'business_name', 's1_addr':'business_address', 's1_country':'country'})
        t_df = df[['business_name', 'business_address', 'country']]
        
        X = engineer_features(s1_df, t_df)
        probs = model.predict(X)
        df['is_match'] = (probs >= best_thresh).astype(int)
        
        matches = df[df['is_match'] == 1]
        results_map = matches.groupby('source1_entity_id')['target_id'].apply(lambda x: ','.join(sorted(set(x)))).to_dict()
        
        for s1_id in all_s1_ids:
            match_str = results_map.get(s1_id, "")
            out_f.write(f"{s1_id}\t{match_str}\n")
            
        print(f"  Processed chunk {i+1} (~{min((i+1)*100000, 5300000)} S1 items)...")
        del df, s1_df, t_df, X, matches, pairs_df, chunk; gc.collect()
        
    out_f.close()
    print("\nSUCCESS! Files are ready in the output/ folder.")
    print("- output/candidate_pairs.tsv")
    print("- output/matching_results.tsv")

if __name__ == "__main__":
    generate_test_candidates()
    infer_test_matches()
    
    print("\nValidating Submission format before you upload...")
    import subprocess
    result = subprocess.run(["python", "utils/validate_submission.py", 
                             "--matching", "output/matching_results.tsv", 
                             "--candidate", "output/candidate_pairs.tsv", 
                             "--test-dir", "dataset/test"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
