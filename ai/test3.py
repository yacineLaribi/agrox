import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from Bio import Phylo
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

class PhylogeneticPredictor:
    """
    Hybridization predictor that uses Phylogeny (.tre) and Hierarchical Data (Family+Genus).
    Prioritizes 'Family First' logic via Gradient Boosting.
    """
    
    def __init__(self):
        self.gb_model = None
        self.scaler = StandardScaler()
        self.df_genus = None
        self.df_family = None
        self.genus_tree = None
        self.family_tree = None
        self.names = None
        self.feature_cols = None
        self.hyb_prop = None
        
    def load_data(self, genus_csv, family_csv, genus_tre, family_tre):
        """Loads CSV data and Newick Tree files."""
        print("="*70)
        print("1. LOADING DATA & PHYLOGENY")
        print("="*70)
        
        # 1. Load CSVs
        self.df_genus = pd.read_csv(genus_csv, na_values=['.', 'NA', 'na', '', 'nan'])
        self.df_family = pd.read_csv(family_csv, na_values=['.', 'NA', 'na', '', 'nan'])
        
        # 2. Load Trees
        try:
            self.genus_tree = Phylo.read(genus_tre, "newick")
            self.family_tree = Phylo.read(family_tre, "newick")
            print("✓ Phylogenetic trees loaded.")
        except Exception as e:
            print(f"! Warning: Could not load trees ({e}). Distance features will be estimated.")
            self.genus_tree = None

        # 3. Hierarchical Merge: Inject Family Data into Genus Data
        # This forces the model to "see" the family stats when looking at a genus
        print("✓ Merging Family-level stats into Genus data...")
        
        fam_subset = self.df_family[['Family', 'HybProp', 'C_value', 'repro_syndrome']].copy()
        fam_subset.columns = ['Family', 'Fam_HybProp', 'Fam_C_value', 'Fam_Repro']
        
        # Left join on Family column
        self.df_genus = pd.merge(self.df_genus, fam_subset, on='Family', how='left')
        
        # 4. Clean Data
        # Define features: Genus specific + Family Context
        self.feature_cols = [
            'perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 
            'mating_system', 'repro_syndrome', 'pollination_syndrome', 
            'RedList', 'C_value', 'CV_C_value', 'tavg',
            'Fam_HybProp', 'Fam_C_value' # <--- Hierarchical features
        ]
        
        # Handle missing values (Median imputation)
        for col in self.feature_cols:
            if col in self.df_genus.columns:
                self.df_genus[col] = pd.to_numeric(self.df_genus[col], errors='coerce')
                med = self.df_genus[col].median()
                self.df_genus[col] = self.df_genus[col].fillna(med if not pd.isna(med) else 0)

        # Finalize matrices
        self.feature_cols = [c for c in self.feature_cols if c in self.df_genus.columns]
        self.names = self.df_genus['Genus'].values
        self.hyb_prop = pd.to_numeric(self.df_genus['HybProp'], errors='coerce').fillna(0).values
        
        # Scale the raw features
        X_raw = self.df_genus[self.feature_cols].values
        self.X_scaled = self.scaler.fit_transform(X_raw)
        
        print(f"✓ Data loaded. Total Genera: {len(self.df_genus)}")
        print(f"✓ Features utilized: {len(self.feature_cols)}")
        return self

    def get_tree_distance(self, name1, name2):
        """Calculates evolutionary distance between two names using the loaded tree."""
        if self.genus_tree is None: return 1.0
        try:
            # Find the terminal nodes
            n1 = next(self.genus_tree.find_elements(target=lambda t: t.name == name1))
            n2 = next(self.genus_tree.find_elements(target=lambda t: t.name == name2))
            return self.genus_tree.distance(n1, n2)
        except:
            return 2.0 # Penalty for not finding them (implies distant or missing)

    def create_pairs(self, sample_size=20000):
        """Generates pairs with Hierarchical Logic."""
        print("\n" + "="*70)
        print("2. CREATING TRAINING PAIRS (HIERARCHY AWARE)")
        print("="*70)
        
        pairs_X = []
        pairs_y = []
        
        n = len(self.names)
        families = self.df_genus['Family'].values
        
        count = 0
        print("Generating pairs...")
        
        while count < sample_size:
            # STRATEGY: 60% Same-Family pairs, 40% Random
            # This teaches the model to distinguish WITHIN families
            if np.random.random() < 0.6:
                idx1 = np.random.randint(n)
                fam = families[idx1]
                same_fam_idxs = np.where(families == fam)[0]
                if len(same_fam_idxs) > 1:
                    idx2 = np.random.choice(same_fam_idxs)
                else:
                    idx2 = np.random.randint(n)
            else:
                idx1, idx2 = np.random.choice(n, 2, replace=False)
            
            if idx1 == idx2: continue

            # Feature Engineering
            f1 = self.X_scaled[idx1]
            f2 = self.X_scaled[idx2]
            
            # 1. Standard Comparison
            diff = np.abs(f1 - f2)
            prod = f1 * f2
            
            # 2. Hierarchical / Taxonomic Features
            same_fam = 1 if families[idx1] == families[idx2] else 0
            
            # 3. Family Compatibility (The "Check Family First" logic)
            # If both plants belong to families that NEVER hybridize, this value is 0
            fam_hyb_potential = (self.df_genus.iloc[idx1]['Fam_HybProp'] + self.df_genus.iloc[idx2]['Fam_HybProp']) / 2
            
            # Combine all features
            # [Diffs, Products, Same_Family_Bool, Family_Potential]
            pair_feats = np.concatenate([diff, prod, [same_fam, fam_hyb_potential]])
            
            # Labeling
            # Strict labeling: Must be capable of hybridization AND usually requires being in same family
            # Unless you want to allow inter-familial, we strictly penalize different families here
            if same_fam == 1 and (self.hyb_prop[idx1] > 0.05 or self.hyb_prop[idx2] > 0.05):
                label = 1
            else:
                label = 0
                
            pairs_X.append(pair_feats)
            pairs_y.append(label)
            count += 1
            
            if count % 5000 == 0: print(f"  {count} pairs generated...")

        return np.array(pairs_X), np.array(pairs_y)

    def train(self, X, y):
        print("\n" + "="*70)
        print("3. TRAINING GRADIENT BOOSTING MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        # Gradient Boosting matches your request for "Decision Tree" logic
        # It creates a series of trees. The first splits usually latch onto "Same Family" and "C-Value"
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.gb_model.fit(X_train, y_train)
        
        # Metrics
        preds = self.gb_model.predict(X_test)
        probs = self.gb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        print(f"Results:")
        print(f"  AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=['No Hyb', 'Hyb']))
        
        return self

    def predict_pair(self, plant1, plant2):
        """Predicts hybridization with real phylogenetic distance."""
        # Check existence
        if plant1 not in self.names or plant2 not in self.names:
            print(f"Error: Plants not found in dataset.")
            return

        idx1 = np.where(self.names == plant1)[0][0]
        idx2 = np.where(self.names == plant2)[0][0]
        
        # Get Features
        f1 = self.X_scaled[idx1]
        f2 = self.X_scaled[idx2]
        
        # 1. Base Math
        diff = np.abs(f1 - f2)
        prod = f1 * f2
        
        # 2. Taxonomy & Hierarchy
        fam1 = self.df_genus.iloc[idx1]['Family']
        fam2 = self.df_genus.iloc[idx2]['Family']
        same_fam = 1 if fam1 == fam2 else 0
        fam_pot = (self.df_genus.iloc[idx1]['Fam_HybProp'] + self.df_genus.iloc[idx2]['Fam_HybProp']) / 2
        
        # 3. Real Phylogenetic Distance (using the .tre file)
        # We calculate this live for the prediction to be precise
        phylo_dist = self.get_tree_distance(plant1, plant2)
        
        # Note: The training loop used a proxy for distance (same_fam) for speed.
        # If we trained with same_fam, we predict with same_fam. 
        # The phylo_dist is useful for YOUR verification, but if we didn't train on it 
        # (because it's slow to generate 20k times), we rely on the family boolean.
        
        # Construct vector
        features = np.concatenate([diff, prod, [same_fam, fam_pot]]).reshape(1, -1)
        
        # Predict
        prob = self.gb_model.predict_proba(features)[0][1]
        
        print("-" * 50)
        print(f"Prediction: {plant1} x {plant2}")
        print(f"  Families: {fam1} / {fam2}")
        print(f"  Evolutionary Distance: {phylo_dist:.4f}")
        print(f"  Family Hybridization Potential: {fam_pot:.3f}")
        print(f"  Probability: {prob:.4f}")
        print(f"  Outcome: {'✓ Compatible' if prob > 0.5 else '✗ Incompatible'}")
        print("-" * 50)


# --- EXECUTION ---
if __name__ == "__main__":
    # Initialize
    bot = PhylogeneticPredictor()
    
    # 1. Load your specific files
    # Ensure these files are in the same folder or provide full paths
    bot.load_data(
        genus_csv='genus_data.csv',
        family_csv='family_data.csv',
        genus_tre='genus.tre',
        family_tre='family.tre'
    )
    
    # 2. Create Pairs
    X, y = bot.create_pairs(sample_size=20000)
    
    # 3. Train
    bot.train(X, y)
    
    # 4. Test Example (Replace with actual names from your CSV)
    # Get random examples from the dataset to test
    if len(bot.names) > 2:
        p1 = bot.names[0]
        p2 = bot.names[333]
        bot.predict_pair(p1, p2)