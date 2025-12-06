"""
Complete Plant Hybridization Predictor
Predicts whether two specific plants can hybridize based on their traits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

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
        
    def load_data(self, family_path='family_data.csv', genus_path='genus_data.csv'):
        """Load and preprocess data"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load data with proper missing value handling
        family_data = pd.read_csv(family_path, na_values=['.', 'NA', 'na', '', 'nan'])
        genus_data = pd.read_csv(genus_path, na_values=['.', 'NA', 'na', '', 'nan'])
        
        # Choose dataset
        self.df = genus_data.copy() if self.use_genus else family_data.copy()
        
        print(f"\nDataset: {self.name_col}-level")
        print(f"Total entries: {len(self.df)}")
        print(f"Columns: {', '.join(self.df.columns)}")
        
        # Define feature columns
        self.feature_cols = ['perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 
                             'mating_system', 'repro_syndrome', 'pollination_syndrome', 
                             'RedList', 'C_value', 'CV_C_value']
        
        if self.use_genus and 'tavg' in self.df.columns:
            self.feature_cols.append('tavg')
        
        # Remove columns that don't exist
        self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]
        
        print(f"\nFeatures used: {len(self.feature_cols)}")
        for col in self.feature_cols:
            print(f"  - {col}")
        
        # Handle missing values
        print("\n" + "-"*70)
        print("MISSING VALUE ANALYSIS")
        print("-"*70)
        
        total_missing = 0
        for col in self.feature_cols:
            # Convert to numeric
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                pct = missing_count / len(self.df) * 100
                print(f"{col:25s}: {missing_count:4d} missing ({pct:5.1f}%)")
                total_missing += missing_count
                
                # Fill with median
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        print(f"\nTotal missing values handled: {total_missing}")
        
        # Extract features and labels
        X = self.df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0)  # Safety check
        
        self.names = self.df[self.name_col].values
        
        # Handle HybProp
        self.hyb_prop = pd.to_numeric(self.df['HybProp'], errors='coerce')
        self.hyb_prop = np.nan_to_num(self.hyb_prop, nan=0.0)
        
        print(f"\n{self.name_col} with hybridization data:")
        print(f"  Total: {len(self.names)}")
        print(f"  With HybProp > 0: {sum(self.hyb_prop > 0)} ({sum(self.hyb_prop > 0)/len(self.hyb_prop)*100:.1f}%)")
        print(f"  HybProp range: [{self.hyb_prop.min():.3f}, {self.hyb_prop.max():.3f}]")
        print(f"  HybProp mean: {self.hyb_prop.mean():.3f}")
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(X)
        
        return self
    
    def create_pairs(self, sample_size=20000, balance_ratio=0.6):
        """
        Create training pairs with enhanced features
        """
        print("\n" + "="*70)
        print("CREATING PAIRWISE TRAINING DATA")
        print("="*70)
        
        pairs_X = []
        pairs_y = []
        
        n = len(self.X_scaled)
        
        # Identify high and low hybridization plants
        high_idx = np.where(self.hyb_prop > 0.1)[0]
        low_idx = np.where(self.hyb_prop <= 0.1)[0]
        
        print(f"\nPlants with high hybridization (>0.1): {len(high_idx)}")
        print(f"Plants with low hybridization (≤0.1): {len(low_idx)}")
        
        # Get taxonomic info for same-family/order features
        if self.use_genus:
            tax_cols = ['Family', 'Order']
        else:
            tax_cols = ['Order']
        
        taxonomic_info = {}
        for col in tax_cols:
            if col in self.df.columns:
                taxonomic_info[col] = self.df[col].values
        
        pairs_created = 0
        target_positive = int(sample_size * balance_ratio)
        target_negative = sample_size - target_positive
        positive_count = 0
        negative_count = 0
        
        max_attempts = sample_size * 20
        attempts = 0
        
        print(f"\nTarget pairs:")
        print(f"  Positive (can hybridize): {target_positive}")
        print(f"  Negative (cannot hybridize): {target_negative}")
        print(f"\nGenerating pairs...")
        
        while pairs_created < sample_size and attempts < max_attempts:
            attempts += 1
            
            # Sample strategy based on what we need
            if positive_count < target_positive and len(high_idx) >= 2:
                # Try to create positive pair
                i, j = np.random.choice(high_idx, 2, replace=False)
            elif negative_count < target_negative and len(low_idx) >= 2:
                # Try to create negative pair
                i, j = np.random.choice(low_idx, 2, replace=False)
            else:
                # Random sampling
                i, j = np.random.choice(n, 2, replace=False)
            
            # Extract features
            f1 = self.X_scaled[i]
            f2 = self.X_scaled[j]
            
            # Feature engineering
            concat = np.concatenate([f1, f2])
            diff = np.abs(f1 - f2)
            prod = f1 * f2
            mean = (f1 + f2) / 2
            max_f = np.maximum(f1, f2)
            min_f = np.minimum(f1, f2)
            
            features = np.concatenate([concat, diff, prod, mean, max_f, min_f])
            
            # Taxonomic similarity
            same_features = []
            for col in tax_cols:
                if col in taxonomic_info:
                    same = 1 if taxonomic_info[col][i] == taxonomic_info[col][j] else 0
                    same_features.append(same)
            
            features = np.concatenate([features, same_features])
            
            # Label determination
            avg_hyb = (self.hyb_prop[i] + self.hyb_prop[j]) / 2
            max_hyb = max(self.hyb_prop[i], self.hyb_prop[j])
            
            # Labeling logic: positive if avg > 0.05 OR max > 0.2
            label = 1 if (avg_hyb > 0.05 or max_hyb > 0.2) else 0
            
            # Check if we need this type of pair
            if label == 1 and positive_count >= target_positive:
                continue
            if label == 0 and negative_count >= target_negative:
                continue
            
            pairs_X.append(features)
            pairs_y.append(label)
            pairs_created += 1
            
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1
            
            if pairs_created % 5000 == 0:
                print(f"  Progress: {pairs_created}/{sample_size} pairs created")
        
        X_pairs = np.array(pairs_X)
        y_pairs = np.array(pairs_y)
        
        print(f"\n✓ Created {len(y_pairs)} pairs")
        print(f"  Positive: {sum(y_pairs)} ({sum(y_pairs)/len(y_pairs)*100:.1f}%)")
        print(f"  Negative: {len(y_pairs)-sum(y_pairs)} ({(len(y_pairs)-sum(y_pairs))/len(y_pairs)*100:.1f}%)")
        print(f"  Feature dimension: {X_pairs.shape[1]}")
        
        return X_pairs, y_pairs
    
    def train(self, X_pairs, y_pairs, test_size=0.2):
        """
        Train Random Forest and Gradient Boosting models
        """
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_pairs, y_pairs, test_size=test_size, random_state=42, stratify=y_pairs
        )
        
        print(f"\nTraining set: {len(X_train)} pairs")
        print(f"Test set: {len(X_test)} pairs")
        
        # Train Random Forest
        print("\n" + "-"*70)
        print("1. Random Forest Classifier")
        print("-"*70)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate RF
        y_pred_rf = self.rf_model.predict(X_test)
        y_prob_rf = self.rf_model.predict_proba(X_test)[:, 1]
        
        acc_rf = self.rf_model.score(X_test, y_test)
        auc_rf = roc_auc_score(y_test, y_prob_rf)
        
        print(f"Accuracy: {acc_rf:.4f}")
        print(f"AUC-ROC: {auc_rf:.4f}")
        
        # Train Gradient Boosting
        print("\n" + "-"*70)
        print("2. Gradient Boosting Classifier")
        print("-"*70)
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.gb_model.fit(X_train, y_train)
        
        # Evaluate GB
        y_pred_gb = self.gb_model.predict(X_test)
        y_prob_gb = self.gb_model.predict_proba(X_test)[:, 1]
        
        acc_gb = self.gb_model.score(X_test, y_test)
        auc_gb = roc_auc_score(y_test, y_prob_gb)
        
        print(f"Accuracy: {acc_gb:.4f}")
        print(f"AUC-ROC: {auc_gb:.4f}")
        
        # Detailed evaluation
        print("\n" + "="*70)
        print("MODEL EVALUATION SUMMARY")
        print("="*70)
        
        print("\nRandom Forest - Classification Report:")
        print(classification_report(y_test, y_pred_rf, 
                                   target_names=['Cannot Hybridize', 'Can Hybridize']))
        
        print("\nGradient Boosting - Classification Report:")
        print(classification_report(y_test, y_pred_gb, 
                                   target_names=['Cannot Hybridize', 'Can Hybridize']))
        
        # Choose best model
        self.best_model = 'rf' if auc_rf >= auc_gb else 'gb'
        best_auc = max(auc_rf, auc_gb)
        
        print(f"\n✓ Best model: {'Random Forest' if self.best_model == 'rf' else 'Gradient Boosting'}")
        print(f"  AUC-ROC: {best_auc:.4f}")
        
        return self
    
    def predict(self, plant1, plant2, verbose=True):
        """
        Predict hybridization potential between two plants
        """
        # Find plants in dataset
        idx1 = np.where(self.names == plant1)[0]
        idx2 = np.where(self.names == plant2)[0]
        
        if len(idx1) == 0:
            if verbose:
                print(f"✗ Error: '{plant1}' not found in database")
            return None
        
        if len(idx2) == 0:
            if verbose:
                print(f"✗ Error: '{plant2}' not found in database")
            return None
        
        i, j = idx1[0], idx2[0]
        
        # Build features
        f1 = self.X_scaled[i]
        f2 = self.X_scaled[j]
        
        concat = np.concatenate([f1, f2])
        diff = np.abs(f1 - f2)
        prod = f1 * f2
        mean = (f1 + f2) / 2
        max_f = np.maximum(f1, f2)
        min_f = np.minimum(f1, f2)
        
        features = np.concatenate([concat, diff, prod, mean, max_f, min_f])
        
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
        
        # Predict with both models
        prob_rf = self.rf_model.predict_proba(features)[0][1]
        prob_gb = self.gb_model.predict_proba(features)[0][1]
        
        # Use best model or average
        if self.best_model == 'rf':
            prob = prob_rf
            model_name = "Random Forest"
        else:
            prob = prob_gb
            model_name = "Gradient Boosting"
        
        # Ensemble average
        prob_avg = (prob_rf + prob_gb) / 2
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"  HYBRIDIZATION PREDICTION")
            print(f"{'='*70}")
            print(f"  Plant 1: {plant1}")
            print(f"  Plant 2: {plant2}")
            print(f"  {'─'*66}")
            print(f"  Random Forest Probability:     {prob_rf:.1%}")
            print(f"  Gradient Boosting Probability: {prob_gb:.1%}")
            print(f"  Ensemble Average:              {prob_avg:.1%}")
            print(f"  {'─'*66}")
            print(f"  Final Prediction ({model_name}): {prob:.1%}")
            print(f"  Result: {'✓ CAN HYBRIDIZE' if prob > 0.5 else '✗ CANNOT HYBRIDIZE'}")
            
            confidence = abs(prob - 0.5)
            if confidence > 0.3:
                conf_level = "High"
            elif confidence > 0.15:
                conf_level = "Medium"
            else:
                conf_level = "Low"
            
            print(f"  Confidence: {conf_level}")
            
            # Additional info
            if self.use_genus:
                fam1 = self.df.loc[self.df['Genus'] == plant1, 'Family'].values[0]
                fam2 = self.df.loc[self.df['Genus'] == plant2, 'Family'].values[0]
                print(f"  {'─'*66}")
                print(f"  {plant1} Family: {fam1}")
                print(f"  {plant2} Family: {fam2}")
                print(f"  Same Family: {'Yes' if fam1 == fam2 else 'No'}")
            
            print(f"{'='*70}\n")
        
        return {
            'probability_rf': prob_rf,
            'probability_gb': prob_gb,
            'probability_ensemble': prob_avg,
            'probability': prob,
            'prediction': 'can_hybridize' if prob > 0.5 else 'cannot_hybridize',
            'confidence': conf_level if verbose else confidence
        }
    
    def save_model(self, filepath='hybridization_predictor.pkl'):
        """Save the trained model"""
        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'names': self.names,
            'X_scaled': self.X_scaled,
            'hyb_prop': self.hyb_prop,
            'df': self.df,
            'use_genus': self.use_genus,
            'best_model': self.best_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='hybridization_predictor.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(use_genus=model_data['use_genus'])
        predictor.rf_model = model_data['rf_model']
        predictor.gb_model = model_data['gb_model']
        predictor.scaler = model_data['scaler']
        predictor.feature_cols = model_data['feature_cols']
        predictor.names = model_data['names']
        predictor.X_scaled = model_data['X_scaled']
        predictor.hyb_prop = model_data['hyb_prop']
        predictor.df = model_data['df']
        predictor.best_model = model_data['best_model']
        
        print(f"✓ Model loaded from {filepath}")
        return predictor


# Main execution
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           PLANT HYBRIDIZATION PREDICTION SYSTEM                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize and train
    predictor = HybridizationPredictor(use_genus=True)
    predictor.load_data()
    
    X_pairs, y_pairs = predictor.create_pairs(sample_size=20000)
    predictor.train(X_pairs, y_pairs)
    
    # Save model
    predictor.save_model()
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    available = predictor.names[:min(20, len(predictor.names))]
    print(f"\nFirst 10 available genera:")
    for i, name in enumerate(available[:10], 1):
        hyb = predictor.hyb_prop[np.where(predictor.names == name)[0][0]]
        print(f"  {i:2d}. {name:20s} (HybProp: {hyb:.3f})")
    
    if len(available) >= 3:
        print("\n" + "-"*70)
        predictor.predict(available[0], available[1])
        predictor.predict(available[0], available[2])
        predictor.predict(available[1], available[2])
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    print("\nUsage examples:")
    print("  predictor.predict('Abies', 'Acer')")
    print("  predictor.predict('Rosa', 'Prunus')")
    print("\nTo load model later:")
    print("  predictor = HybridizationPredictor.load_model('hybridization_predictor.pkl')")
    print("="*70 + "\n")