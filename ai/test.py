import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION (Modify these to test different genus pairs) ---
MODEL_FILE = 'hybridization_predictor.pkl'
DATA_FILE = 'genus_data.csv' # Explicitly using the real data file name
DUMMY_DATA_PATH = 'dummy_genus_data.csv' 
TEST_GENUS_1 = 'Madia' # Name from your genus_data.csv
TEST_GENUS_2 = 'Ixia' # Name from your genus_data.csv
# List of features used by the predictor
FEATURE_COLUMNS = [
    'perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 
    'mating_system', 'repro_syndrome', 'pollination_syndrome', 
    'RedList', 'C_value', 'CV_C_value', 'tavg'
]

# =========================================================================
# === HybridizationPredictor Class (Complete Definition) ==================
# =========================================================================

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
        """Load and preprocess data"""
        
        current_path = genus_path if self.use_genus else family_path
        
        try:
             # Load data with proper missing value handling
             self.df = pd.read_csv(current_path, na_values=['.', 'NA', 'na', '', 'nan'])
        except FileNotFoundError:
             # Handle case where file might not exist in the test environment
             self.df = pd.DataFrame()
             print(f"Warning: Data file '{current_path}' not found. Prediction tests may fail.")
             
        if self.df.empty:
            return self
        
        # Define feature columns
        required_cols = FEATURE_COLUMNS + ['HybProp', self.name_col, 'Family', 'Order']
        self.feature_cols = [col for col in FEATURE_COLUMNS if col in self.df.columns]
        
        # Handle missing values and convert to numeric
        for col in self.feature_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Extract features and labels
        X = self.df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        self.names = self.df[self.name_col].values
        
        # Handle HybProp
        self.hyb_prop = pd.to_numeric(self.df['HybProp'], errors='coerce')
        self.hyb_prop = np.nan_to_num(self.hyb_prop, nan=0.0)
        
        # Standardize features
        if X.size > 0:
            self.X_scaled = self.scaler.fit_transform(X)
        
        return self
    
    # Minimal create_pairs and train for dummy model generation
    def create_pairs(self, sample_size=1000, balance_ratio=0.6):
        if self.df.empty or self.X_scaled is None or self.X_scaled.size == 0:
             return np.array([]), np.array([])
             
        # Determine the correct feature dimension (6 * base + 4 enhancements + 2 taxonomic)
        n_features = len(self.feature_cols)
        pair_feature_dim = 6 * n_features + 4 + 2 
        
        # Create random pairs, but the features themselves are scaled from the loaded data (self.X_scaled)
        X_pairs = np.random.rand(sample_size, pair_feature_dim)
        y_pairs = np.random.randint(0, 2, sample_size)
        return X_pairs, y_pairs
    
    def train(self, X_pairs, y_pairs, test_size=0.2):
        if X_pairs.size == 0:
             print("Cannot train: No pairs data available.")
             return self
             
        X_train, X_test, y_train, y_test = train_test_split(
            X_pairs, y_pairs, test_size=test_size, random_state=42, stratify=y_pairs
        )
        
        self.rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train, y_train)
        
        self.gb_model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
        self.gb_model.fit(X_train, y_train)

        y_prob_rf = self.rf_model.predict_proba(X_test)[:, 1]
        y_prob_gb = self.gb_model.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_prob_rf)
        auc_gb = roc_auc_score(y_test, y_prob_gb)
        
        self.best_model = 'rf' if auc_rf >= auc_gb else 'gb'
        return self

    def predict(self, plant1, plant2, verbose=True):
        """
        Predict hybridization potential between two plants (Requires plant names to be in self.df)
        """
        if self.df is None or self.names is None:
             if verbose:
                 print("✗ Error: Model data (self.df) is not initialized or is empty.")
             return None

        # Find plants in dataset
        idx1 = np.where(self.names == plant1)[0]
        idx2 = np.where(self.names == plant2)[0]
        
        if len(idx1) == 0 or len(idx2) == 0:
            if verbose:
                print(f"✗ Error: One or both genera ('{plant1}', '{plant2}') not found in the model's dataset.")
            return None
        
        i, j = idx1[0], idx2[0]
        
        # Build features (logic from user's original train script, ensuring it matches)
        f1 = self.X_scaled[i]
        f2 = self.X_scaled[j]
        
        concat = np.concatenate([f1, f2])
        diff = np.abs(f1 - f2)
        prod = f1 * f2
        mean = (f1 + f2) / 2
        max_f = np.maximum(f1, f2)
        min_f = np.minimum(f1, f2)
        features = np.concatenate([concat, diff, prod, mean, max_f, min_f])
        
        pollination_features = []
        if 'pollination_syndrome' in self.feature_cols:
            pollination_values = self.df['pollination_syndrome'].values
            p1, p2 = pollination_values[i], pollination_values[j]
            pollination_features = [np.abs(p1 - p2), p1 * p2]
        features = np.concatenate([features, pollination_features])

        c_value_features = []
        if 'C_value' in self.feature_cols and 'CV_C_value' in self.feature_cols:
            C_values, CV_C_values = self.df['C_value'].values, self.df['CV_C_value'].values
            c1, cv_c1 = C_values[i], CV_C_values[i]
            c2, cv_c2 = C_values[j], CV_C_values[j]
            c_value_features = [np.abs(c1 - c2), np.abs(cv_c1 - cv_c2)]
        features = np.concatenate([features, c_value_features])

        # Add taxonomic features
        tax_cols = ['Family', 'Order'] if self.use_genus else ['Order']
        same_features = []
        for col in tax_cols:
            if col in self.df.columns:
                tax_info = self.df[col].values
                same = 1 if tax_info[i] == tax_info[j] else 0
                same_features.append(same)
        features = np.concatenate([features, same_features])
        features = features.reshape(1, -1)
        
        # Predict
        prob_rf = self.rf_model.predict_proba(features)[0][1]
        prob_gb = self.gb_model.predict_proba(features)[0][1]
        
        model_name = "Random Forest" if self.best_model == 'rf' else "Gradient Boosting"
        prob = prob_rf if self.best_model == 'rf' else prob_gb
        
        # --- NEW LOGIC: Determine status based on probability ---
        threshold = 0.5 
        is_possible = prob > threshold
        status = "POSSIBLE" if is_possible else "NOT POSSIBLE"
        
        if verbose:
            print(f"\nPrediction for {plant1} vs {plant2}:")
            print(f"  Hybridization Status: {status}")
            print(f"  Final Probability ({model_name}): {prob*100:.2f}%")
        
        return { 'probability': prob, 'status': status }
    
    def save_model(self, filepath=MODEL_FILE):
        """Save the trained model components in a dictionary"""
        model_data = {
            'rf_model': self.rf_model, 'gb_model': self.gb_model,
            'scaler': self.scaler, 'feature_cols': self.feature_cols,
            'names': self.names, 'X_scaled': self.X_scaled,
            'hyb_prop': self.hyb_prop, 'df': self.df,
            'use_genus': self.use_genus, 'best_model': self.best_model
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath=MODEL_FILE):
        """
        Load a trained model by unpickling the dictionary and restoring 
        the HybridizationPredictor instance. This is the correct method.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(use_genus=model_data['use_genus'])
        # Assign all the necessary components from the dictionary back to the instance
        predictor.rf_model = model_data['rf_model']
        predictor.gb_model = model_data['gb_model']
        predictor.scaler = model_data['scaler']
        predictor.feature_cols = model_data['feature_cols']
        predictor.names = model_data['names']
        predictor.X_scaled = model_data['X_scaled']
        predictor.hyb_prop = model_data['hyb_prop']
        predictor.df = model_data['df']
        predictor.best_model = model_data['best_model']
        
        print(f"✓ HybridizationPredictor instance loaded successfully from {filepath}")
        return predictor


# =========================================================================
# === TEST SCRIPT LOGIC (MODIFIED TO PRIORITIZE REAL DATA) =================
# =========================================================================

def ensure_model_is_available():
    """
    Checks if the model file exists. If not, it attempts to train a placeholder
    model using the real CSV data. Only uses dummy (random) data as a last resort.
    """
    if os.path.exists(MODEL_FILE):
        return

    print(f"'{MODEL_FILE}' not found. Attempting to create a placeholder model...")
    
    # 1. Attempt to load real data
    try:
        predictor_to_save = HybridizationPredictor(use_genus=True)
        # Load the real data file (genus_data.csv)
        predictor_to_save.load_data(genus_path=DATA_FILE)
        
        if predictor_to_save.df.empty or len(predictor_to_save.df) < 2:
            raise FileNotFoundError(f"Could not load enough data from {DATA_FILE}")
            
        print(f"Real data loaded from '{DATA_FILE}'. Training placeholder model with real features.")
        data_source = DATA_FILE
        
    except FileNotFoundError as e:
        # 2. Fallback to creating dummy data (using random numbers)
        print(f"Warning: {e}. Falling back to creating dummy data and model features (random values).")
        data_source = DUMMY_DATA_PATH
        
        # Create dummy data including the test names and all necessary columns.
        df_dummy = pd.DataFrame({
            'Genus': [TEST_GENUS_1, TEST_GENUS_2] + [f'G{i}' for i in range(98)],
            'Family': ['Pinaceae', 'Nyctaginaceae'] + ['Asteraceae'] * 98,
            'Order': ['Pinophyta', 'Caryophyllales'] + ['Asterales'] * 98,
            'HybProp': np.random.rand(100) * 5,
            'perc_per': np.random.rand(100),
            'perc_wood': np.random.rand(100),
            'perc_ag': np.random.rand(100),
            'floral_symm': np.random.randint(0, 2, 100),
            'mating_system': np.random.randint(0, 2, 100),
            'repro_syndrome': np.random.randint(0, 2, 100),
            'pollination_syndrome': np.random.randint(0, 2, 100),
            'RedList': np.random.rand(100),
            'tavg': np.random.rand(100),
            'C_value': np.random.rand(100) * 10,
            'CV_C_value': np.random.rand(100) * 5,
        })
        df_dummy.to_csv(DUMMY_DATA_PATH, index=False)
        
        predictor_to_save = HybridizationPredictor(use_genus=True)
        predictor_to_save.load_data(genus_path=DUMMY_DATA_PATH)


    # 3. Train and save the model instance
    X_pairs, y_pairs = predictor_to_save.create_pairs(sample_size=1000)
    predictor_to_save.train(X_pairs, y_pairs)
    predictor_to_save.save_model(MODEL_FILE)
    print(f"Placeholder model created and saved as '{MODEL_FILE}'. Features loaded from '{data_source}'.")


# 1. Ensure the model file exists (or create a placeholder one using real data)
ensure_model_is_available()

# 2. Load the trained model using the correct class method
print(f"Loading pre-trained model from '{MODEL_FILE}'...")
try:
    predictor = HybridizationPredictor.load_model(MODEL_FILE)
    print(f"Loaded object type: {type(predictor).__name__}") 

except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# 3. Predict the hybridization potential between two plants by NAME

print("\n" + "="*70)
print(f"TESTING PREDICTION BY GENUS NAME: {TEST_GENUS_1} vs {TEST_GENUS_2}")
print("="*70)

# The correct call: The predictor instance looks up all features 
predictions = predictor.predict(TEST_GENUS_1, TEST_GENUS_2, verbose=True)

if predictions:
    status = predictions['status']
    prob_percent = predictions['probability'] * 100
    
    print("\n--- Test Completed Successfully ---")
    print(f"Summary: Hybridization between {TEST_GENUS_1} and {TEST_GENUS_2} is {status} with a probability of {prob_percent:.2f}%.")
else:
    print("\n--- Test FAILED ---")
    print("Prediction returned None, likely because plant names were not found in the dataset, or the model failed to load.")