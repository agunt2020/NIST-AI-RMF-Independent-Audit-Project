# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from fairlearn.metrics import selection_rate, demographic_parity_difference

# 2. Load the Customer Shopping Trends dataset
# Replace with your actual path if downloading locally
df = pd.read_csv('shopping_trends_updated.csv')

print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# Create the intersectional sensitive attribute column
# 1 = Young Female (Vulnerable Group), 0 = All Others (Control Group)
df['Is_Young_Female'] = np.where((df['Age'] <= 24) & (df['Gender'] == 'Female'), 1, 0)

# Check the distribution to ensure you have an adequate sample size for auditing
distribution = df['Is_Young_Female'].value_counts(normalize=True) * 100
print(f"Control Group (All Others): {distribution[0]:.2f}%")
print(f"Vulnerable Group (Young Females): {distribution[1]:.2f}%")

# Define your features (X) and target variable (y)
# Target variable: 'Discount Applied' (Yes = 1, No = 0)
df['Discount_Target'] = np.where(df['Discount Applied'] == 'Yes', 1, 0)

# Select the features you mapped as potential proxies or predictors
feature_cols = ['Age', 'Gender', 'Purchase Amount (USD)', 'Location',
                'Size', 'Color', 'Season', 'Subscription Status',
                'Shipping Type', 'Payment Method', 'Frequency of Purchases']

X = df[feature_cols].copy()
y = df['Discount_Target']

# Encode categorical text variables into integers
le = LabelEncoder()
text_cols = X.select_dtypes(include=['object', 'string']).columns
for col in text_cols:
    X[col] = le.fit_transform(X[col])

# Split into training and testing sets (80% train, 20% test)
# We also track our sensitive attribute array for the test set
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, df['Is_Young_Female'], test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = baseline_model.predict(X_test)

# Print overall technical performance
print("Overall Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate selection rates (percentage of individuals predicted to get a discount)
sr_vulnerable = selection_rate(y_test[sensitive_test == 1], y_pred[sensitive_test == 1])
sr_control = selection_rate(y_test[sensitive_test == 0], y_pred[sensitive_test == 0])

print(f"Discount Allocation Rate for Young Women: {sr_vulnerable * 100:.2f}%")
print(f"Discount Allocation Rate for All Others: {sr_control * 100:.2f}%")

# Calculate Disparate Impact Ratio
disparate_impact_ratio = sr_vulnerable / sr_control
print(f"Disparate Impact Ratio: {disparate_impact_ratio:.4f}")

if disparate_impact_ratio < 0.80:
    print("🚨 POLICY ALERT: Systemic bias detected. The model violates the 80% threshold for fair resource distribution.")
else:
    print("✅ PASS: The model distributes discounts equitably under current thresholds.")

dem_parity_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
print(f"Demographic Parity Difference: {dem_parity_diff:.4f}")

# Extract feature importances from our Random Forest
importances = baseline_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance Weight': importances
}).sort_values(by='Importance Weight', ascending=False)

print("\n--- Technical Explanability Audit: Top Feature Influences ---")
print(feature_importance_df)

# Create a summarized evaluation matrix
audit_results = pd.DataFrame({
    "NIST Trustworthiness Category": ["Fairness & Bias", "Economic Equity", "Explainability"],
    "Technical Metric Evaluated": ["Demographic Parity Difference", "Disparate Impact Ratio", "Top Proxy Predictors"],
    "Calculated Value": [f"{dem_parity_diff:.4f}", f"{disparate_impact_ratio:.4f}", f"{feature_cols[np.argmax(importances)]}"],
    "Risk Level": ["High" if dem_parity_diff > 0.1 else "Low",
                   "Critical (Predatory)" if disparate_impact_ratio < 0.8 else "Low",
                   "Review Required"]
})

print("\n--- FINAL NIST MEASURE PHASE EVALUATION MATRIX ---")
print(audit_results.to_string(index=False))