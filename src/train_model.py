import sys
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_engineering import engineer_features
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir="dataset/train/"):
    print("Loading GT...")
    gt = pd.read_csv(data_dir + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt['matched_list'] = gt['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    positive_pairs = set()
    for s1_id, matches in zip(gt['source1_entity_id'], gt['matched_list']):
        for match_id in matches:
            positive_pairs.add((s1_id, match_id))
    del gt; gc.collect()
    return positive_pairs

def build_training_dataset(positive_pairs, data_dir="dataset/train/", candidates_path="output/train_candidate_pairs.tsv"):
    print("Loading Candidates...")
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    cands['cand_list'] = cands['candidate_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    
    # Cap to protect memory limit (30GB Kaggle)
    cands = cands.sample(n=min(len(cands), 40_000), random_state=42)
    
    pairs_df = cands.explode('cand_list').rename(columns={'cand_list': 'target_id'}).dropna(subset=['target_id'])
    pairs_df = pairs_df[['source1_entity_id', 'target_id']]
    del cands; gc.collect()
    
    print(f"Total modeling pairs: {len(pairs_df)}")
    
    print("Merging S1 Data...")
    s1 = pd.read_csv(data_dir + "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
    s1 = s1.rename(columns={'business_name': 's1_name', 'business_address': 's1_addr', 'country': 's1_country'})
    df = pairs_df.merge(s1, left_on='source1_entity_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
    del s1; gc.collect()

    print("Merging S2 Data...")
    s2 = pd.read_csv(data_dir + "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
    s2 = s2.rename(columns={'business_name': 't_name', 'business_address': 't_addr', 'country': 't_country'})
    df = df.merge(s2, left_on='target_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
    del s2; gc.collect()

    print("Merging S3 Data...")
    s3 = pd.read_csv(data_dir + "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=['entity_id', 'business_name', 'business_address', 'country'])
    s3 = s3.rename(columns={'business_name': 't_name_3', 'business_address': 't_addr_3', 'country': 't_country_3'})
    df = df.merge(s3, left_on='target_id', right_on='entity_id', how='left').drop(columns=['entity_id'])
    del s3; gc.collect()

    # Consolidate S2 and S3 targets natively
    df['business_name'] = df['t_name'].fillna('') + df['t_name_3'].fillna('')
    df['business_address'] = df['t_addr'].fillna('') + df['t_addr_3'].fillna('')
    df['country'] = df['t_country'].fillna('') + df['t_country_3'].fillna('')
    
    df.drop(columns=['t_name', 't_addr', 't_country', 't_name_3', 't_addr_3', 't_country_3'], inplace=True)
    
    s1_df = df[['s1_name', 's1_addr', 's1_country']].rename(columns={'s1_name':'business_name', 's1_addr':'business_address', 's1_country':'country'})
    t_df = df[['business_name', 'business_address', 'country']]
    
    print("Starting NLP computation...")
    X = engineer_features(s1_df, t_df)
    
    def is_match(row):
        return 1 if (row['source1_entity_id'], row['target_id']) in positive_pairs else 0
        
    y = df.apply(is_match, axis=1).values
    del s1_df; del t_df; del df; gc.collect()
    return X, y

def train_and_evaluate(X, y):
    print("Training LightGBM classifier...")
    # Split into train/validation (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss', callbacks=[lgb.early_stopping(stopping_rounds=30)])
    
    importances = list(zip(X.columns, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<25}: {imp}")
        
    print("\nEvaluating Probability Thresholds for F0.5 Score...")
    val_probs = model.predict_proba(X_val)[:, 1]
    best_f05 = 0; best_thresh = 0
    from sklearn.metrics import precision_score, recall_score, fbeta_score
    
    for thresh in np.arange(0.5, 0.99, 0.05):
        y_pred = (val_probs >= thresh).astype(int)
        if sum(y_pred) == 0: continue
        p = precision_score(y_val, y_pred)
        r = recall_score(y_val, y_pred)
        f = fbeta_score(y_val, y_pred, beta=0.5)
        print(f"  Threshold {thresh:.2f} -> Prec: {p:.4f} | Rec: {r:.4f} | F0.5: {f:.4f}")
        if f > best_f05:
            best_f05, best_thresh = f, thresh
            
    print(f"\n=> BEST THRESHOLD: {best_thresh:.2f} (F0.5 = {best_f05:.4f})")
    
    model.booster_.save_model('output/lgbm_model.txt')
    with open('output/model_metadata.txt', 'w') as f:
        f.write(str(best_thresh))
    print("Optimization Complete! Saved to output/")

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    pos_pairs = load_data()
    X, y = build_training_dataset(pos_pairs)
    train_and_evaluate(X, y)
