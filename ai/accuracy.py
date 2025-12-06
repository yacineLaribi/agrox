import joblib
from sklearn.metrics import accuracy_score
from gemini import HybridizationPredictor

# load saved model
model_data = joblib.load("hybridization_predictor.pkl")

# create predictor (use_genus must match saved model)
predictor = HybridizationPredictor(use_genus=model_data["use_genus"])

# load test data properly
predictor.load_data(genus_path="gtest_data.csv")

# assign saved models & best_model
predictor.rf_model = model_data["rf_model"]
predictor.gb_model = model_data["gb_model"]
predictor.best_model = model_data["best_model"]

# generate pairwise features from test set
X_pairs, y_pairs = predictor.create_pairs(sample_size=2000)

# select best model
best_model = predictor.rf_model if predictor.best_model == "rf" else predictor.gb_model

# predict + compute accuracy
pred = best_model.predict(X_pairs)
print("Accuracy on gtest_data.csv:", accuracy_score(y_pairs, pred))
