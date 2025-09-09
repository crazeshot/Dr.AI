# model.py
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Example dataset (expand with more symptoms + diseases)
data = [
    (["fever", "cough", "tiredness"], "Flu"),
    (["headache", "nausea", "sensitivity to light"], "Migraine"),
    (["fever", "rash", "joint pain"], "Dengue"),
    (["chest pain", "shortness of breath", "fatigue"], "Heart Disease"),
    (["itchy eyes", "runny nose", "sneezing"], "Allergy"),
    (["abdominal pain", "diarrhea", "vomiting"], "Food Poisoning"),
]

# Create vocab of symptoms
all_symptoms = sorted(list({s for symptoms, _ in data for s in symptoms}))

# Encode diseases
diseases = [d for _, d in data]
disease_encoder = LabelEncoder()
disease_encoder.fit(diseases)

# Create training data
X = []
y = []
for symptoms, disease in data:
    vector = [1 if s in symptoms else 0 for s in all_symptoms]
    X.append(vector)
    y.append(disease_encoder.transform([disease])[0])

X = np.array(X)
y = np.array(y)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model + encoders
with open("disease_model.pkl", "wb") as f:
    pickle.dump((model, all_symptoms, disease_encoder), f)

print("✅ Model trained and saved as disease_model.pkl")
