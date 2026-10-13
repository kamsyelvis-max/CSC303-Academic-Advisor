# Run these cells in Google Colab (or paste this script into a Colab notebook).
# Demonstration data is generated for this assignment; it is not real student data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from google.colab import files

np.random.seed(42)
n = 1200

cgpa = np.random.uniform(0, 5, n)
attendance = np.random.uniform(30, 100, n)
study_hours = np.random.uniform(0, 8, n)
assignment_score = np.random.uniform(20, 100, n)

# Create demonstration labels using a transparent scoring rule.
score = (
    (cgpa / 5) * 40
    + (attendance / 100) * 20
    + (np.minimum(study_hours, 6) / 6) * 15
    + (assignment_score / 100) * 25
)

def label_for(value):
    if value < 45:
        return "At Risk"
    elif value < 60:
        return "Needs Improvement"
    elif value < 78:
        return "Good"
    return "Excellent"

labels = np.array([label_for(value) for value in score])

data = pd.DataFrame({
    "cgpa": cgpa,
    "attendance": attendance,
    "study_hours": study_hours,
    "assignment_score": assignment_score,
    "category": labels
})
display(data.head())
print(data["category"].value_counts())

X = data[["cgpa", "attendance", "study_hours", "assignment_score"]]
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model as a .pkl file.
with open("academic_advisor_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Download the model from Colab to your computer.
files.download("academic_advisor_model.pkl")
