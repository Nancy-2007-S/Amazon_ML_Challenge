import sys
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_engineering import engineer_features
from run_kaggle_candidates import count_freq, build_index, generate_candidates

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
    
    # 1. Load Candidates and create pairing target df
    cands = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # Store the S1 IDs to guarantee every S1 exists in the final submission
    all_s1_ids = cands['source1_entity_id'].unique()
    
    cands['cand_list'] = cands['candidate_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    
    # Only evaluate pairs that actually exist (ignore ones with empty candidate sets)
    pairs_df = cands.explode('cand_list').rename(columns={'cand_list': 'target_id'}).dropna(subset=['target_id'])
    pairs_df = pairs_df[['source1_entity_id', 'target_id']]
    del cands; gc.collect()
    
    if len(pairs_df) > 0:
        print(f"Total Test Pairs to Evaluate: {len(pairs_df)}")
        
        # 2. Sequential Merging of Data to prevent OOM
        print("Merging Test S1 Data...")
        s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
        s1 = s1.rename(columns={'business_name': 's1_name', 'business_address': 's1_addr', 'country': 's1_country'})
        df = pairs_df.merge(s1, left_on='source1_entity_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
        del s1; gc.collect()

        print("Merging Test S2 Data...")
        s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
        s2 = s2.rename(columns={'business_name': 't_name', 'business_address': 't_addr', 'country': 't_country'})
        df = df.merge(s2, left_on='target_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
        del s2; gc.collect()

        print("Merging Test S3 Data...")
        s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
        s3 = s3.rename(columns={'business_name': 't_name_3', 'business_address': 't_addr_3', 'country': 't_country_3'})
        df = df.merge(s3, left_on='target_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
        del s3; gc.collect()

        # Consolidate S2/S3
        df['business_name'] = df['t_name'].fillna('') + df['t_name_3'].fillna('')
        df['business_address'] = df['t_addr'].fillna('') + df['t_addr_3'].fillna('')
        df['country'] = df['t_country'].fillna('') + df['t_country_3'].fillna('')
        df.drop(columns=['t_name', 't_addr', 't_country', 't_name_3', 't_addr_3', 't_country_3'], inplace=True)
        
        s1_df = df[['s1_name', 's1_addr', 's1_country']].rename(columns={'s1_name':'business_name', 's1_addr':'business_address', 's1_country':'country'})
        t_df = df[['business_name', 'business_address', 'country']]
        
        print("Scoring Features...")
        X = engineer_features(s1_df, t_df)
        
        # 3. Model Prediction
        print("Predicting matches...")
        probs = model.predict(X)
        df['is_match'] = (probs >= best_thresh).astype(int)
        
        # Keep only the rows predicted as true matches
        matches = df[df['is_match'] == 1]
    else:
        # Failsafe if literally no candidates triggered (unlikely)
        matches = pd.DataFrame(columns=['source1_entity_id', 'target_id'])

    # 4. Format the final output
    print("Formatting Final Output...")
    results_map = matches.groupby('source1_entity_id')['target_id'].apply(lambda x: ','.join(sorted(set(x)))).to_dict()
    
    with open('output/matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        
        for s1_id in all_s1_ids:
            match_str = results_map.get(s1_id, "")
            f.write(f"{s1_id}\t{match_str}\n")
            
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
