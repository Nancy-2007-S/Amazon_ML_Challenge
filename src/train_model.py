import sys
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_engineering import engineer_features
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir="dataset/train/"):
    print("Loading raw data...")
    s1 = pd.read_csv(data_dir + "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    s2 = pd.read_csv(data_dir + "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    s3 = pd.read_csv(data_dir + "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("entity_id")
    
    gt = pd.read_csv(data_dir + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # Process ground truth into a set of positive pair tuples: (s1_id, match_id)
    print("Processing Ground Truth...")
    gt['matched_list'] = gt['matched_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    positive_pairs = set()
    for s1_id, matches in zip(gt['source1_entity_id'], gt['matched_list']):
        for match_id in matches:
            positive_pairs.add((s1_id, match_id))
            
    return s1, s2, s3, positive_pairs

def build_training_dataset(s1, s2, s3, positive_pairs, candidates_path="output/train_candidate_pairs.tsv"):
    print(f"Loading candidates from {candidates_path}...")
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    cands['cand_list'] = cands['candidate_entity_ids'].apply(
        lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
    )
    
    # To save Kaggle RAM, only take a sample of S1s for training (e.g., 200,000 entities)
    print("Sampling candidates to prevent OOM on Kaggle...")
    cands = cands.sample(n=min(len(cands), 200_000), random_state=42)
    
    print("Exploding candidates into pairs...")
    pairs_df = cands.explode('cand_list').rename(columns={'cand_list': 'target_id'}).dropna(subset=['target_id'])
    
    pair_count = len(pairs_df)
    print(f"Total training pairs created: {pair_count:,}")
    
    print("Building S1 vs Target dataframes...")
    s1_df = s1.loc[pairs_df['source1_entity_id']].reset_index()
    
    # We must retrieve the corresponding records from S2 and S3 for target_id
    def get_target_record(eid):
        if eid.startswith("S2-") and eid in s2.index: return s2.loc[eid]
        if eid.startswith("S3-") and eid in s3.index: return s3.loc[eid]
        return pd.Series(dtype=str)
    
    # Using a fast list comprehension since map on large DataFrames can be slow
    print("Fetching target records (S2/S3)...")
    target_records = [get_target_record(tid) for tid in pairs_df['target_id']]
    target_df = pd.DataFrame(target_records).reset_index(drop=True)
    
    # Engineer Features
    print("Computing NLP features (Jaccard, Levenshtein, etc.). This may take exactly a few minutes...")
    X = engineer_features(s1_df, target_df)
    
    # Apply ground truth labels
    print("Applying target labels...")
    def is_match(row):
        return 1 if (row['source1_entity_id'], row['target_id']) in positive_pairs else 0
        
    y = pairs_df.apply(is_match, axis=1).values
    
    print(f"Dataset Built! Positives: {sum(y)}, Negatives: {len(y) - sum(y)}")
    return X, y

def train_and_evaluate(X, y):
    print("Training LightGBM classifier...")
    
    # Split into train/validation (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        class_weight='balanced', # Crucial: heavy class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    print("\nFeature Importances:")
    importances = list(zip(X.columns, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        print(f"  {feat:<25}: {imp}")
        
    # Evaluate thresholds for F0.5
    print("\nEvaluating Probability Thresholds for F0.5 Score...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    best_f05 = 0
    best_thresh = 0
    from sklearn.metrics import precision_score, recall_score, fbeta_score
    
    for thresh in np.arange(0.5, 0.99, 0.05):
        y_pred = (val_probs >= thresh).astype(int)
        p = precision_score(y_val, y_pred)
        r = recall_score(y_val, y_pred)
        f = fbeta_score(y_val, y_pred, beta=0.5)
        print(f"  Threshold {thresh:.2f} -> Prec: {p:.4f} | Rec: {r:.4f} | F0.5: {f:.4f}")
        if f > best_f05:
            best_f05 = f
            best_thresh = thresh
            
    print(f"\n=> BEST THRESHOLD: {best_thresh:.2f} (F0.5 = {best_f05:.4f})")
    
    # Save the model
    model.booster_.save_model('output/lgbm_model.txt')
    print("Model saved to output/lgbm_model.txt")
    
    # Save the threshold to a metadata file for inference
    with open('output/model_metadata.txt', 'w') as f:
        f.write(str(best_thresh))

if __name__ == "__main__":
    s1, s2, s3, positive_pairs = load_data()
    X, y = build_training_dataset(s1, s2, s3, positive_pairs)
    train_and_evaluate(X, y)
    print("Stage 5 Complete. Ready for Inference!")
