import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os

# Create visualization directory if it doesn't exist
viz_dir = "./visualization/predictmodel"
os.makedirs(viz_dir, exist_ok=True)

# Load dataset
data = pd.read_csv(r"./data/processed/HGG_DB_cleaned.csv")

# Drop irrelevant columns and handle missing values
data_cleaned = data.drop(columns=['sample']).fillna('None')

# Encoding categorical variables
label_encoders = {}
for col in data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
    label_encoders[col] = le

# Splitting data into features (X) and target (y)
X = data_cleaned.drop(columns=['tumor_grade'])
y = data_cleaned['tumor_grade']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance analysis
feature_importance = rf_model.feature_importances_
features = X.columns

# Create DataFrame with feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=True)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'])
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), bbox_inches='tight', dpi=300)
plt.close()

# Random Forest predictions and evaluation
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))

# Plot confusion matrix for Random Forest
plt.figure(figsize=(6, 4))
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'confusion_matrix_rf.png'))
plt.close()

# Plot ROC curve for Random Forest
y_proba_rf = rf_model.predict_proba(X_test)
auc_rf = roc_auc_score(pd.get_dummies(y_test), y_proba_rf, multi_class='ovr')
fpr_rf, tpr_rf, _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_proba_rf.ravel())

plt.figure(figsize=(6, 4))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', linestyle='--')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'roc_curve_rf.png'))
plt.close()

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_resampled, y_resampled)

# XGBoost predictions and evaluation
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Classification Report:\n")
print(classification_report(y_test, y_pred_xgb))

# Plot confusion matrix for XGBoost
plt.figure(figsize=(6, 4))
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="OrRd")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'confusion_matrix_xgb.png'))
plt.close()

# Plot feature importance for XGBoost
xgb_importance = xgb_model.feature_importances_
xgb_features = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_importance})
xgb_features_sorted = xgb_features.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(12, 8))
plt.barh(xgb_features_sorted['Feature'], xgb_features_sorted['Importance'])
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'feature_importance_xgb.png'), bbox_inches='tight', dpi=300)
plt.close()

# Plot ROC curve for XGBoost
y_proba_xgb = xgb_model.predict_proba(X_test)
auc_xgb = roc_auc_score(pd.get_dummies(y_test), y_proba_xgb, multi_class='ovr')
fpr_xgb, tpr_xgb, _ = roc_curve(pd.get_dummies(y_test).values.ravel(), y_proba_xgb.ravel())

plt.figure(figsize=(6, 4))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', linestyle='-')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'roc_curve_xgb.png'))
plt.close()

# Create model comparison plot
comparison_metrics = {
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_rf, average='weighted'),
        precision_score(y_test, y_pred_xgb, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, y_pred_rf, average='weighted'),
        recall_score(y_test, y_pred_xgb, average='weighted')
    ],
    "F1-score": [
        f1_score(y_test, y_pred_rf, average='weighted'),
        f1_score(y_test, y_pred_xgb, average='weighted')
    ],
    "AUC": [
        auc_rf,
        auc_xgb
    ]
}

comparison_df = pd.DataFrame(comparison_metrics, index=['Random Forest', 'XGBoost'])

# Create the heatmap with exact styling
plt.figure(figsize=(12, 8))
sns.heatmap(comparison_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Model Comparison: Random Forest vs XGBoost')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save Random Forest results
rf_report = classification_report(y_test, y_pred_rf)
with open(os.path.join(viz_dir, 'random_forest_report.txt'), 'w') as f:
    f.write("Random Forest Classification Report:\n\n")
    f.write(rf_report)

# Save XGBoost results
xgb_report = classification_report(y_test, y_pred_xgb)
with open(os.path.join(viz_dir, 'xgboost_report.txt'), 'w') as f:
    f.write("XGBoost Classification Report:\n\n")
    f.write(xgb_report)

# Save model comparison results
with open(os.path.join(viz_dir, 'model_comparison.txt'), 'w') as f:
    f.write("Model Performance Summary:\n")
    f.write(comparison_df.to_string())

# Print to terminal for immediate feedback
print("Random Forest Classification Report:\n")
print(rf_report)
print("\nXGBoost Classification Report:\n")
print(xgb_report)
print("\nModel Performance Summary:")
print(comparison_df)

print(f"\nResults have been saved to: {viz_dir}")
