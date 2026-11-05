"""Lab 06 helper: (re)creates Mymdoel.pkl using YOUR installed scikit-learn.
Run this once if main.py complains about the model file.
    python train_model.py
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

here = Path(__file__).parent
df = pd.read_csv(here.parent / "lab05" / "diabetes.csv")
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X, y = df[features], df['Outcome']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print("Test accuracy:", round(model.score(X_te, y_te), 4))
with open(here / "Mymdoel.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved Mymdoel.pkl")
