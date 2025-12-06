# main.py
import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# optional tree lib
try:
    from ete3 import Tree
    ETE3_AVAILABLE = True
except Exception:
    ETE3_AVAILABLE = False

# -------------------------
# User files (edit paths)
# -------------------------
GENUS_CSV = "dataset/genus_data.csv"      # contains Genus,Family,Order,...,HybProp
FAMILY_CSV = "dataset/family_data.csv"    # optional
GENUS_TREE = "dataset/genus.tre"          # optional Newick (prefer genus tree)
KNOWN_PAIRS_CSV = "dataset/known_pairs.csv"  # optional labeled pairs: A_name,B_name,CanHybridize (1/0)

# -------------------------
# Helper: phylogenetic distance
# -------------------------
def load_tree(path):
    if not os.path.exists(path):
        return None
    if ETE3_AVAILABLE:
        try:
            return Tree(path, format=1)  # try format=1, common for Newick with names
        except Exception:
            try:
                return Tree(path, format=0)
            except Exception:
                return None
    else:
        return None

def phylo_distance(tree, name1, name2):
    """
    Returns branch-length distance between name1 and name2 if tree available and nodes found.
    If not available, returns np.nan so caller can fallback to taxonomic distance.
    """
    if tree is None or not ETE3_AVAILABLE:
        return np.nan
    # try to find nodes by name
    n1 = tree.search_nodes(name=name1)
    n2 = tree.search_nodes(name=name2)
    if not n1 or not n2:
        return np.nan
    try:
        return tree.get_distance(n1[0], n2[0])
    except Exception:
        return np.nan

# -------------------------
# Taxonomic fallback distance
# -------------------------
def taxonomic_distance(rowA, rowB):
    """
    simple integer distance: 0 = same genus, 1 = same family, 2 = same order, 3 = different
    normalized to [0,1]
    """
    if rowA.get("Genus") == rowB.get("Genus"):
        d = 0
    elif rowA.get("Family") == rowB.get("Family"):
        d = 1
    elif rowA.get("Order") == rowB.get("Order"):
        d = 2
    else:
        d = 3
    return d / 3.0

# -------------------------
# Load and clean datasets
# -------------------------
def load_and_clean_genus(path):
    df = pd.read_csv(path)
    # standard cleaning
    df.replace('.', pd.NA, inplace=True)
    # numeric cols we expect
    numeric_cols = ["HybProp","Hyb_Ratio","C_value","CV_C_value","perc_per","perc_wood","perc_ag","tavg"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows with no genus name or no HybProp info at all (user choice)
    df = df.dropna(subset=["Genus"]).reset_index(drop=True)
    # fill categorical NA with 'Unknown'
    cat_cols = ["Family","Order","floral_symm","mating_system","repro_syndrome","pollination_syndrome","RedList"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
    # ensure Genus is string
    df["Genus"] = df["Genus"].astype(str)
    return df

df_genus = load_and_clean_genus(GENUS_CSV)
tree = load_tree(GENUS_TREE)  # may be None

# -------------------------
# Pair feature builder
# -------------------------
NUMERIC_FEATURES = [c for c in ["Hyb_Ratio","C_value","CV_C_value","perc_per","perc_wood","perc_ag","tavg"] if c in df_genus.columns]
CAT_FEATURES = [c for c in ["Family","Order","floral_symm","mating_system","repro_syndrome","pollination_syndrome","RedList"] if c in df_genus.columns]

def make_pair_features(rowA, rowB):
    feat = {}
    # ids
    feat["A_Genus"] = rowA["Genus"]
    feat["B_Genus"] = rowB["Genus"]
    # taxonomic same indicators
    feat["same_genus"] = int(rowA.get("Genus") == rowB.get("Genus"))
    feat["same_family"] = int(rowA.get("Family") == rowB.get("Family")) if "Family" in rowA else 0
    feat["same_order"] = int(rowA.get("Order") == rowB.get("Order")) if "Order" in rowA else 0
    # phylogenetic distance (branch-length) if possible
    pdist = phylo_distance(tree, rowA["Genus"], rowB["Genus"])
    if np.isnan(pdist):
        feat["phylo_dist"] = taxonomic_distance(rowA, rowB)  # normalized fallback
        feat["phylo_exact"] = 0
    else:
        feat["phylo_dist"] = pdist
        feat["phylo_exact"] = 1
    # numeric differences and means
    for c in NUMERIC_FEATURES:
        a = rowA.get(c, np.nan)
        b = rowB.get(c, np.nan)
        feat[f"absdiff_{c}"] = np.nan if pd.isna(a) or pd.isna(b) else abs(a - b)
        feat[f"mean_{c}"] = np.nan if pd.isna(a) or pd.isna(b) else (a + b) / 2.0
    # categorical similarity counts (1 if equal, 0 else)
    for c in CAT_FEATURES:
        feat[f"same_{c}"] = int(rowA.get(c) == rowB.get(c))
    return feat

# -------------------------
# Supervised training if labeled pairs exist
# -------------------------
def train_pairwise_classifier(df_genus, known_pairs_csv=None):
    """
    If known_pairs_csv exists, it should have columns: A_name,B_name,CanHybridize (1/0)
    A_name and B_name should match the Genus names in df_genus.
    """
    if known_pairs_csv is None or not os.path.exists(known_pairs_csv):
        print("No labeled pairs file found -> cannot train supervised pairwise classifier.")
        return None, None, None

    pairs = pd.read_csv(known_pairs_csv)
    # keep only rows where both taxa exist in df_genus
    pairs = pairs[pairs["A_name"].isin(df_genus["Genus"]) & pairs["B_name"].isin(df_genus["Genus"])].reset_index(drop=True)
    feats = []
    labels = []
    for _, r in pairs.iterrows():
        rowA = df_genus[df_genus["Genus"] == r["A_name"]].iloc[0].to_dict()
        rowB = df_genus[df_genus["Genus"] == r["B_name"]].iloc[0].to_dict()
        f = make_pair_features(rowA, rowB)
        feats.append(f)
        labels.append(int(r["CanHybridize"]))
    X = pd.DataFrame(feats).fillna(0)
    y = np.array(labels)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    # oversample numeric matrix for minority class using SMOTE (works on numeric arrays)
    # So first scale numeric columns; SMOTE doesn't like extremely high dims but should work
    numeric_cols_for_smote = [c for c in X_train.columns if X_train[c].dtype.kind in "fi"]
    sm = SMOTE(random_state=42)
    X_train_num = X_train[numeric_cols_for_smote].values
    X_train_res_num, y_train_res = sm.fit_resample(X_train_num, y_train)
    # replace numeric columns with resampled ones and replicate categorical columns
    # NOTE: this is a simple approach; better approach is to encode categories before SMOTE
    # Here we will re-create a new X_train_res by sampling from X_train rows by indices from sm
    # To keep it simple, use RandomForest without SMOTE if encoding complexity is an issue
    try:
        # Try encoding categories properly and resample using SMOTE on full numeric array of encoded features
        X_train_enc = pd.get_dummies(X_train, drop_first=False)
        X_test_enc = pd.get_dummies(X_test, drop_first=False)
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_enc.values, y_train)
        # train classifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_res, y_train_res)
        preds = clf.predict(X_test_enc.values)
        print("Supervised pairwise classifier results (held-out test):")
        print("Accuracy:", accuracy_score(y_test, preds))
        print("F1:", f1_score(y_test, preds))
        return clf, X_train_enc.columns.tolist(), ("supervised", X_test_enc, y_test)
    except Exception as e:
        # fallback: no SMOTE encoding path; train without SMOTE on one-hot encoded data
        X_enc = pd.get_dummies(X, drop_first=False)
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_e, y_train_e)
        preds = clf.predict(X_test_e)
        print("Supervised pairwise classifier (fallback, no SMOTE) results:")
        print("Accuracy:", accuracy_score(y_test_e, preds))
        print("F1:", f1_score(y_test_e, preds))
        return clf, X_enc.columns.tolist(), ("supervised_fallback", X_test_e, y_test_e)

# -------------------------
# Heuristic scorer (if no labels)
# -------------------------
def heuristic_score_pair(rowA, rowB):
    """
    Returns a score in [0,1] estimating hybridization likelihood:
    - low phylo distance (closer) increases score
    - similar numeric traits increase score (lower absdiff)
    - same family/genus increases score
    This is a heuristic — not learned from data.
    """
    p = make_pair_features(rowA, rowB)
    # phylo score: invert distance (if phylo_exact==1 and dist>0, use 1/(1+dist)); if using normalized taxonomy, use 1-dist
    if p["phylo_exact"]:
        phy_score = 1.0 / (1.0 + p["phylo_dist"])
    else:
        phy_score = 1.0 - p["phylo_dist"]  # taxonomy distance normalized [0,1], invert
    # numeric similarity: average of (1 - normalized absdiff)
    numeric_scores = []
    for c in NUMERIC_FEATURES:
        ad = p.get(f"absdiff_{c}", np.nan)
        meanv = p.get(f"mean_{c}", np.nan)
        if pd.isna(ad) or pd.isna(meanv) or meanv == 0:
            continue
        # relative diff
        rel = ad / (meanv + 1e-6)
        numeric_scores.append(max(0.0, 1 - rel))  # clipped
    num_score = np.mean(numeric_scores) if numeric_scores else 0.0
    # categorical bonus
    cat_bonus = 0
    for c in CAT_FEATURES:
        cat_bonus += p.get(f"same_{c}", 0)
    cat_bonus = cat_bonus / max(1, len(CAT_FEATURES))
    # combine
    score = 0.5 * phy_score + 0.35 * num_score + 0.15 * cat_bonus
    return float(np.clip(score, 0.0, 1.0))

# -------------------------
# Convenience: predict_pair (uses supervised model if present, else heuristic)
# -------------------------
def build_named_index(df):
    return {r["Genus"]: r for _, r in df.iterrows()}

named_idx = build_named_index(df_genus)

# try supervised train if known pairs available
clf, feature_names, debug_info = None, None, None
if os.path.exists(KNOWN_PAIRS_CSV):
    clf, feature_names, debug_info = train_pairwise_classifier(df_genus, KNOWN_PAIRS_CSV)
    print("Trained supervised pairwise classifier.")
else:
    print("No labeled pairs found. Using heuristic scorer only.")

def predict_pair(genusA, genusB, return_score_only=False):
    if genusA not in named_idx or genusB not in named_idx:
        raise ValueError("One or both genus names not found in genus dataset.")
    rowA = named_idx[genusA]
    rowB = named_idx[genusB]
    # supervised path
    if clf is not None:
        # build features consistent with training encoding
        feat = make_pair_features(rowA, rowB)
        Xf = pd.DataFrame([feat])
        # if feature_names provided (one-hot encoded columns), we must one-hot encode like training
        Xf_enc = pd.get_dummies(Xf)
        # align to training columns
        train_cols = feature_names
        Xf_enc = Xf_enc.reindex(columns=train_cols, fill_value=0)
        prob = clf.predict_proba(Xf_enc.values)[0, 1] if hasattr(clf, "predict_proba") else clf.predict(Xf_enc.values)[0]
        if return_score_only:
            return prob
        return {"probability": float(prob), "method": "supervised"}
    # heuristic path
    score = heuristic_score_pair(rowA, rowB)
    if return_score_only:
        return score
    return {"score": score, "method": "heuristic"}

# -------------------------
# Example usage (uncomment to run)
# -------------------------
if __name__ == "__main__":
    # If you have a supervised model trained (known_pairs.csv present), you can call:
    # print(predict_pair("Abies", "Acacia"))
    # If not, you still can use heuristic:
    print("Example heuristic score ( vs ):", predict_pair("Abies", "Abutilon"))
