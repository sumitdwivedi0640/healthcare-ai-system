import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("../data/sepsis/sepsis.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Target column (adjust if needed)
target_col = "SepsisLabel"

# Remove rows where target is NaN
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Fill missing values in features
X = X.fillna(X.mean())


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate imbalance ratio
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(model, "../models/sepsis_model.pkl")
joblib.dump(scaler, "../models/sepsis_scaler.pkl")

print("Model saved successfully!")

import shap
import matplotlib.pyplot as plt

# Convert X_test back to DataFrame with feature names
X_test_df = pd.DataFrame(X_test, columns=df.drop(columns=[target_col]).columns)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

# Generate summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.tight_layout()
plt.savefig("../models/sepsis_shap_summary.png")
plt.close()

print("SHAP summary plot saved successfully!")


