import pandas as pd
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def basic_stats(df, name):
    stats = {
        "name": name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "types": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
        "missing_values": {str(k): v for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_entity_ids": df["entity_id"].duplicated().sum() if "entity_id" in df.columns else 0,
        "duplicate_records": df.duplicated().sum()
    }
    
    if "country" in df.columns:
        stats["country_distribution"] = {str(k): v for k, v in df["country"].value_counts(dropna=False).to_dict().items()}
        
    if "business_name" in df.columns:
        df["business_name_len"] = df["business_name"].astype(str).str.len()
        stats["name_len_dist"] = df["business_name_len"].describe().to_dict()
        stats["empty_names"] = (df["business_name"].astype(str).str.strip() == "").sum()
        
    if "business_address" in df.columns:
        df["business_address_len"] = df["business_address"].astype(str).str.len()
        stats["addr_len_dist"] = df["business_address_len"].describe().to_dict()
        stats["empty_addresses"] = (df["business_address"].astype(str).str.strip() == "").sum()
        
    return stats

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

report = {}
report["Source 1"] = basic_stats(s1, "Source 1")
report["Source 2"] = basic_stats(s2, "Source 2")
report["Source 3"] = basic_stats(s3, "Source 3")

# Ground truth stats
matches = gt["matched_entity_ids"].fillna("").astype(str).str.split(",").apply(lambda x: [i.strip() for i in x if i.strip()])
gt["match_count"] = matches.apply(len)
s2_counts = matches.apply(lambda x: sum(1 for i in x if i.startswith("S2-")))
s3_counts = matches.apply(lambda x: sum(1 for i in x if i.startswith("S3-")))

report["Ground Truth"] = {
    "num_source1_entities": len(gt),
    "singletons_no_match": (gt["match_count"] == 0).sum(),
    "has_match": (gt["match_count"] > 0).sum(),
    "match_count_distribution": {str(k): v for k, v in gt["match_count"].value_counts().to_dict().items()},
    "total_s2_matches": s2_counts.sum(),
    "total_s3_matches": s3_counts.sum(),
    "multiple_matches": (gt["match_count"] > 1).sum()
}

with open("eda_report.json", "w") as f:
    json.dump(report, f, indent=4, cls=NpEncoder)
