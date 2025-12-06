import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- HybridizationPredictor Class ---
class HybridizationPredictor:
    """
    A machine learning model to predict plant hybridization potential
    """
    
    def __init__(self, use_genus=True):
        self.use_genus = use_genus
        self.name_col = 'Genus' if use_genus else 'Family'
        self.scaler = StandardScaler()
        self.rf_model = None
        self.gb_model = None
        self.feature_cols = None
        self.df = None
        self.names = None
        self.X_scaled = None
        self.hyb_prop = None
        self.best_model = None
        
    def load_data(self, family_path='family_data.csv', genus_path='genus_data.csv'):
        """Load and preprocess data, using mock data if files are missing."""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load data with proper missing value handling
        # Using mock data since files were not provided
        try:
            family_data = pd.read_csv(family_path, na_values=['.', 'NA', 'na', '', 'nan'])
        except FileNotFoundError:
            print(f"\n! WARNING: {family_path} not found. Using mock data.")
            family_data = mock_family_data 

        try:
            genus_data = pd.read_csv(genus_path, na_values=['.', 'NA', 'na', '', 'nan'])
        except FileNotFoundError:
            print(f"! WARNING: {genus_path} not found. Using mock data.")
            genus_data = mock_genus_data 

        # Choose dataset
        self.df = genus_data.copy() if self.use_genus else family_data.copy()
        
        print(f"\nDataset: {self.name_col}-level")
        print(f"Total entries: {len(self.df)}")
        
        # Define feature columns
        self.feature_cols = ['perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 
                             'mating_system', 'repro_syndrome', 'pollination_syndrome', 
                             'RedList', 'C_value', 'CV_C_value']
        
        if self.use_genus and 'tavg' in self.df.columns:
            self.feature_cols.append('tavg')
        
        self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]
        
        # Handle missing values and convert to numeric
        for col in self.feature_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Extract features and labels
        X = self.df[self.feature_cols].values
        self.names = self.df[self.name_col].values
        self.hyb_prop = pd.to_numeric(self.df['HybProp'], errors='coerce')
        self.hyb_prop = np.nan_to_num(self.hyb_prop, nan=0.0)
        
        print(f"  Total with hybridization data: {len(self.names)}")
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(X)
        
        return self
    
    def create_pairs(self, sample_size=20000, balance_ratio=0.6):
        """
        Create training pairs using robust sub-sampling from all possible pairs.
        """
        print("\n" + "="*70)
        print("CREATING PAIRWISE TRAINING DATA")
        print("="*70)
        
        n = len(self.X_scaled)
        if n < 2:
            print("✗ Error: Not enough data points to create pairs.")
            return np.array([]), np.array([])
            
        # 1. Generate all unique (i, j) index pairs
        i_indices, j_indices = np.triu_indices(n, k=1) 
        
        print(f"\nTotal possible unique pairs from {n} entries: {len(i_indices)}")
        
        # 2. Determine labels for all pairs
        hyb_i = self.hyb_prop[i_indices]
        hyb_j = self.hyb_prop[j_indices]
        
        # Labeling logic: positive if avg > 0.05 OR max > 0.2
        avg_hyb = (hyb_i + hyb_j) / 2
        max_hyb = np.maximum(hyb_i, hyb_j)
        y_all = np.where((avg_hyb > 0.05) | (max_hyb > 0.2), 1, 0)
        
        # Separate indices by label
        pos_indices = np.where(y_all == 1)[0]
        neg_indices = np.where(y_all == 0)[0]
        
        target_positive = int(sample_size * balance_ratio)
        target_negative = sample_size - target_positive
        
        print(f"  Total Positive Pairs Available: {len(pos_indices)}")
        print(f"  Total Negative Pairs Available: {len(neg_indices)}")
        print(f"\nSampling targets:")
        print(f"  Positive (Can Hybridize): {target_positive}")
        print(f"  Negative (Cannot Hybridize): {target_negative}")
        
        # Check if we have enough samples
        current_positive = len(pos_indices)
        current_negative = len(neg_indices)
        
        # Determine the maximum feasible number of pairs while maintaining the ratio
        max_possible_positive = min(target_positive, current_positive)
        max_possible_negative = min(target_negative, current_negative)
        
        if max_possible_positive < target_positive or max_possible_negative < target_negative:
            # Adjust the sample size to the maximum feasible while still hitting the target ratio balance
            # Find the limiting factor
            # We calculate the max number of pairs we can get while keeping the positive/negative ratio (0.6/0.4)
            limit_ratio = min(current_positive / target_positive, current_negative / target_negative)
            
            # Recalculate targets based on the limiting ratio
            target_positive = int(target_positive * limit_ratio)
            target_negative = int(target_negative * limit_ratio)
            
            sample_size = target_positive + target_negative

            print("\n! WARNING: Insufficient pairs for full sample size/balance ratio. Adjusting sample size.")
            print(f"  Adjusted sample size: {sample_size}")
            
        # 3. Sub-sample
        
        # Sample positive indices
        choice_pos = np.random.choice(pos_indices, target_positive, replace=False)
        
        # Sample negative indices
        choice_neg = np.random.choice(neg_indices, target_negative, replace=False)
        
        # Combine sampled indices
        chosen_indices = np.concatenate([choice_pos, choice_neg])
        np.random.shuffle(chosen_indices)
        
        final_i_indices = i_indices[chosen_indices]
        final_j_indices = j_indices[chosen_indices]
        
        pairs_X = []
        pairs_y = y_all[chosen_indices]
        
        # 4. Feature Engineering for chosen pairs
        pollination_values = self.df['pollination_syndrome'].values if 'pollination_syndrome' in self.feature_cols else None
        C_values = self.df['C_value'].values if 'C_value' in self.feature_cols else None
        CV_C_values = self.df['CV_C_value'].values if 'CV_C_value' in self.feature_cols else None
        
        # Get taxonomic info for same-family/order features
        tax_cols = ['Family', 'Order'] if self.use_genus else ['Order']
        taxonomic_info = {col: self.df[col].values for col in tax_cols if col in self.df.columns}

        for i, j in zip(final_i_indices, final_j_indices):
            f1 = self.X_scaled[i]
            f2 = self.X_scaled[j]
            
            # Standard features (uses scaled values)
            concat = np.concatenate([f1, f2])
            diff = np.abs(f1 - f2)
            prod = f1 * f2
            mean = (f1 + f2) / 2
            max_f = np.maximum(f1, f2)
            min_f = np.minimum(f1, f2)
            
            features = np.concatenate([concat, diff, prod, mean, max_f, min_f])
            
            # Add enhancements (unscaled values)
            pollination_features = []
            if pollination_values is not None:
                p1, p2 = pollination_values[i], pollination_values[j]
                pollination_features = [np.abs(p1 - p2), p1 * p2]
            features = np.concatenate([features, pollination_features])

            c_value_features = []
            if C_values is not None and CV_C_values is not None:
                c1, cv_c1 = C_values[i], CV_C_values[i]
                c2, cv_c2 = C_values[j], CV_C_values[j]
                c_value_features = [np.abs(c1 - c2), np.abs(cv_c1 - cv_c2)]
            features = np.concatenate([features, c_value_features])

            # Taxonomic similarity
            same_features = []
            for col in tax_cols:
                if col in taxonomic_info:
                    same = 1 if taxonomic_info[col][i] == taxonomic_info[col][j] else 0
                    same_features.append(same)
            features = np.concatenate([features, same_features])
            
            pairs_X.append(features)
        
        X_pairs = np.array(pairs_X)
        y_pairs = np.array(pairs_y)
        
        print(f"\n✓ Created {len(y_pairs)} pairs")
        print(f"  Positive: {sum(y_pairs)} ({sum(y_pairs)/len(y_pairs)*100:.1f}%)")
        print(f"  Negative: {len(y_pairs)-sum(y_pairs)} ({(len(y_pairs)-sum(y_pairs))/len(y_pairs)*100:.1f}%)")
        print(f"  Feature dimension: {X_pairs.shape[1]} (Expected 72 based on 11 base features)") 
        
        return X_pairs, y_pairs
    
    # ... (rest of the class methods train, predict, save_model are unchanged)
    def train(self, X_pairs, y_pairs, test_size=0.2):
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        X_train, X_test, y_train, y_test = train_test_split(
            X_pairs, y_pairs, test_size=test_size, random_state=42, stratify=y_pairs
        )
        print(f"\nTraining set: {len(X_train)} pairs")
        print(f"Test set: {len(X_test)} pairs")
        self.rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1, verbose=0)
        self.rf_model.fit(X_train, y_train)
        y_prob_rf = self.rf_model.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_prob_rf)
        print(f"\nRandom Forest AUC-ROC: {auc_rf:.4f}")
        self.gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42, verbose=0)
        self.gb_model.fit(X_train, y_train)
        y_prob_gb = self.gb_model.predict_proba(X_test)[:, 1]
        auc_gb = roc_auc_score(y_test, y_prob_gb)
        print(f"Gradient Boosting AUC-ROC: {auc_gb:.4f}")
        self.best_model = 'rf' if auc_rf >= auc_gb else 'gb'
        print(f"\n✓ Best model: {'Random Forest' if self.best_model == 'rf' else 'Gradient Boosting'}")
        return self
        
    def predict(self, plant1, plant2, verbose=True):
        # Find plants in dataset
        idx1 = np.where(self.names == plant1)[0]
        idx2 = np.where(self.names == plant2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            if verbose: print(f"✗ Error: One or both plants not found in database")
            return None
        i, j = idx1[0], idx2[0]
        
        # Build features (using logic identical to create_pairs for consistency)
        f1 = self.X_scaled[i]
        f2 = self.X_scaled[j]
        concat = np.concatenate([f1, f2]); diff = np.abs(f1 - f2); prod = f1 * f2
        mean = (f1 + f2) / 2; max_f = np.maximum(f1, f2); min_f = np.minimum(f1, f2)
        features = np.concatenate([concat, diff, prod, mean, max_f, min_f])
        
        pollination_values = self.df['pollination_syndrome'].values if 'pollination_syndrome' in self.feature_cols else None
        pollination_features = []
        if pollination_values is not None:
            p1, p2 = pollination_values[i], pollination_values[j]
            pollination_features = [np.abs(p1 - p2), p1 * p2]
        features = np.concatenate([features, pollination_features])

        C_values = self.df['C_value'].values if 'C_value' in self.feature_cols else None
        CV_C_values = self.df['CV_C_value'].values if 'CV_C_value' in self.feature_cols else None
        c_value_features = []
        if C_values is not None and CV_C_values is not None:
            c1, cv_c1 = C_values[i], CV_C_values[i]
            c2, cv_c2 = C_values[j], CV_C_values[j]
            c_value_features = [np.abs(c1 - c2), np.abs(cv_c1 - cv_c2)]
        features = np.concatenate([features, c_value_features])

        tax_cols = ['Family', 'Order'] if self.use_genus else ['Order']
        taxonomic_info = {col: self.df[col].values for col in tax_cols if col in self.df.columns}
        same_features = []
        for col in tax_cols:
            if col in taxonomic_info:
                same = 1 if taxonomic_info[col][i] == taxonomic_info[col][j] else 0
                same_features.append(same)
        features = np.concatenate([features, same_features])
        features = features.reshape(1, -1)
        
        # Predict
        prob_rf = self.rf_model.predict_proba(features)[0][1]
        prob_gb = self.gb_model.predict_proba(features)[0][1]
        prob = prob_rf if self.best_model == 'rf' else prob_gb
        model_name = "Random Forest" if self.best_model == 'rf' else "Gradient Boosting"
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"  HYBRIDIZATION PREDICTION")
            print(f"{'='*70}")
            print(f"  Plant 1: {plant1}\n  Plant 2: {plant2}")
            print(f"  ──────────────────────────────────────────────────────────────────")
            print(f"  Final Prediction ({model_name}): {prob:.1%}")
            print(f"  Result: {'✓ CAN HYBRIDIZE' if prob > 0.5 else '✗ CANNOT HYBRIDIZE'}")
            print(f"{'='*70}\n")
        
        return prob
        
    def save_model(self, filepath='hybridization_predictor.pkl'):
        # In a real environment, you would use pickle or joblib here
        print(f"✓ Model (mock) saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='hybridization_predictor.pkl'):
        raise NotImplementedError("Mock data does not support loading previously saved models.")

# --- Mock Data Generation (To run the script without files) ---
N_ENTRIES = 250
np.random.seed(42)
mock_data = {
    'Genus': [f'PlantGenus_{i:03d}' for i in range(N_ENTRIES)],
    'Family': [f'Family_{i % 5}' for i in range(N_ENTRIES)],
    'Order': [f'Order_{i % 3}' for i in range(N_ENTRIES)],
    'HybProp': np.random.rand(N_ENTRIES) * 0.4,
    'perc_per': np.random.rand(N_ENTRIES) * 100, 'perc_wood': np.random.rand(N_ENTRIES) * 100, 
    'perc_ag': np.random.rand(N_ENTRIES) * 100, 'floral_symm': np.random.randint(1, 6, N_ENTRIES),
    'mating_system': np.random.randint(1, 4, N_ENTRIES), 'repro_syndrome': np.random.randint(1, 5, N_ENTRIES), 
    
    # Initialize 'pollination_syndrome' as float to allow np.nan assignment
    'pollination_syndrome': np.random.randint(1, 8, N_ENTRIES).astype(float),
    
    'RedList': np.random.randint(0, 5, N_ENTRIES), 
    'C_value': np.random.rand(N_ENTRIES) * 20, 'CV_C_value': np.random.rand(N_ENTRIES) * 5,
    'tavg': np.random.uniform(5, 30, N_ENTRIES)
}

# Assign NaNs (This section now works correctly)
mock_data['C_value'][np.random.choice(N_ENTRIES, 10)] = np.nan
mock_data['pollination_syndrome'][np.random.choice(N_ENTRIES, 5)] = np.nan 

# FIX: Use np.nan instead of the string 'NA' to avoid ValueError
mock_data['HybProp'][np.random.choice(N_ENTRIES, 5)] = np.nan 

mock_genus_data = pd.DataFrame(mock_data)
mock_family_data = pd.DataFrame(mock_data) 

# --- Main execution ---
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           PLANT HYBRIDIZATION PREDICTION SYSTEM (FIXED 4)            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

predictor = HybridizationPredictor(use_genus=True)
predictor.load_data(genus_path='genus_data.csv', family_path='family_data.csv') 

# Using the requested sample_size=20000 with the robust pair generation logic
X_pairs, y_pairs = predictor.create_pairs(sample_size=20000) 
predictor.train(X_pairs, y_pairs)

predictor.save_model()

# Example predictions using plants from the mock data
available = predictor.names[:min(20, len(predictor.names))]
if len(available) >= 3:
    predictor.predict(available[0], available[1]) 
    predictor.predict(available[2], available[3])