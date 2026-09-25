import pandas as pd
from fuzzywuzzy import fuzz
from normalization import normalize_name, normalize_address
import gc

def compute_jaccard(str1, str2):
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0.0
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_num_jaccard(str1, str2):
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0.0
    set1 = set(t for t in str1.split() if t.isdigit())
    set2 = set(t for t in str2.split() if t.isdigit())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def engineer_features(s1_df, cand_df):
    """
    Given a dataframe of S1 records and a dataframe of Candidate records (S2/S3),
    compute similarity features row-by-row.
    """
    # 1. Normalize
    s1_df['nn'] = s1_df['business_name'].apply(normalize_name)
    s1_df['na'] = s1_df['business_address'].apply(normalize_address)
    cand_df['nn'] = cand_df['business_name'].apply(normalize_name)
    cand_df['na'] = cand_df['business_address'].apply(normalize_address)

    # 2. Extract country
    s1_df['country'] = s1_df['country'].fillna('').astype(str).str.lower().str.strip()
    cand_df['country'] = cand_df['country'].fillna('').astype(str).str.lower().str.strip()

    # 3. Compute Features (Vectorized using pandas lists/zips where possible)
    features = []
    
    # We assume s1_df and cand_df are perfectly aligned (row i of s1_df matched with row i of cand_df)
    for s1_n, s1_a, s1_c, c_n, c_a, c_c in zip(
        s1_df['nn'], s1_df['na'], s1_df['country'],
        cand_df['nn'], cand_df['na'], cand_df['country']
    ):
        f = {}
        
        # Country Feature
        f['country_match'] = 1.0 if s1_c and c_c and s1_c == c_c else 0.0
        
        # Name Features
        s1_n = s1_n or ""
        c_n = c_n or ""
        f['name_fuzz_ratio'] = fuzz.ratio(s1_n, c_n) / 100.0
        f['name_fuzz_token_sort'] = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
        f['name_jaccard'] = compute_jaccard(s1_n, c_n)
        
        # Address Features
        s1_a = s1_a or ""
        c_a = c_a or ""
        f['addr_fuzz_token_set'] = fuzz.token_set_ratio(s1_a, c_a) / 100.0
        f['addr_jaccard'] = compute_jaccard(s1_a, c_a)
        f['addr_num_jaccard'] = compute_num_jaccard(s1_a, c_a)
        
        features.append(f)
        
    feature_df = pd.DataFrame(features)
    return feature_df

def create_training_pairs(s1_path, s2_path, s3_path, gt_path, candidate_path):
    """
    Master function to merge blocking output with raw dataset to create training pairs.
    """
    pass # To be implemented perfectly scaled for Kaggle in the next step
