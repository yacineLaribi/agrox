# new_script.py
from gemini import HybridizationPredictor

# 1. Load the trained model
# This restores the models AND the plant database (names, features)
print("Loading model...")
loaded_model = HybridizationPredictor.load_model('hybridization_predictor.pkl')

# 2. Make a single prediction
# You can now predict on any plants that were in your original CSVs
print("Predicting...")
result = loaded_model.predict('Rosa', 'Rosa')

# 3. Access the results programmatically
if result:
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']}")
