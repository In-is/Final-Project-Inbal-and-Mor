import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_score, recall_score, f1_score
)

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
DATA_DIR = Path("./data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VISUALIZATION_DIR = Path("./visualization/predictmodel")
HGG_FILE = PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"

class PredictiveModeling:
    """Class for training and evaluating machine learning models on HGG data."""
    
    def __init__(self):
        """Initialize the predictive modeling analysis."""
        self.data: pd.DataFrame = None
        self.data_cleaned: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.X_resampled: pd.DataFrame = None
        self.y_resampled: pd.Series = None
        self.rf_model: RandomForestClassifier = None
        self.xgb_model: XGBClassifier = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Create output directory
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_and_preprocess_data(self) -> None:
        """Load and preprocess the HGG dataset."""
        try:
            # Load dataset
            self.data = pd.read_csv(HGG_FILE)
            logger.info(f"Loaded data from {HGG_FILE}")
            
            # Clean and prepare data
            self.data_cleaned = self.data.drop(columns=['sample']).fillna('None')
            self._encode_categorical_variables()
            logger.info("Data preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error in data loading and preprocessing: {str(e)}")
            raise
            
    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables using LabelEncoder."""
        for col in self.data_cleaned.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data_cleaned[col] = le.fit_transform(self.data_cleaned[col])
            self.label_encoders[col] = le
            
    def prepare_train_test_data(self) -> None:
        """Prepare training and test datasets with SMOTE balancing."""
        # Split features and target
        X = self.data_cleaned.drop(columns=['tumor_grade'])
        y = self.data_cleaned['tumor_grade']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE for balancing
        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X_train, self.y_train)
        logger.info("Train-test split and SMOTE balancing completed")
        
    def train_models(self) -> None:
        """Train Random Forest and XGBoost models."""
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        logger.info("Random Forest model training completed")
        
        # Train XGBoost
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.xgb_model.fit(self.X_resampled, self.y_resampled)
        logger.info("XGBoost model training completed")
        
    def plot_feature_importance(self, model: Any, model_name: str) -> None:
        """Plot feature importance for a given model.
        
        Args:
            model: Trained model instance
            model_name: Name of the model for plot labeling
        """
        importance = model.feature_importances_
        features = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df_sorted = importance_df.sort_values(by='Importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'])
        plt.title(f"Feature Importance - {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / f'feature_importance_{model_name.lower()}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series,
                            model_name: str, cmap: str = "Blues") -> None:
        """Plot confusion matrix for model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for plot labeling
            cmap: Color map for the plot
        """
        plt.figure(figsize=(6, 4))
        conf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / f'confusion_matrix_{model_name.lower()}.png')
        plt.close()
        
    def plot_roc_curve(self, y_true: pd.Series, y_proba: pd.Series,
                      model_name: str) -> float:
        """Plot ROC curve for model predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model for plot labeling
            
        Returns:
            float: AUC score
        """
        auc = roc_auc_score(pd.get_dummies(y_true), y_proba, multi_class='ovr')
        fpr, tpr, _ = roc_curve(pd.get_dummies(y_true).values.ravel(),
                               y_proba.ravel())
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / f'roc_curve_{model_name.lower()}.png')
        plt.close()
        
        return auc
        
    def evaluate_models(self) -> None:
        """Evaluate both models and generate comparison plots."""
        # Get predictions
        y_pred_rf = self.rf_model.predict(self.X_test)
        y_pred_xgb = self.xgb_model.predict(self.X_test)
        y_proba_rf = self.rf_model.predict_proba(self.X_test)
        y_proba_xgb = self.xgb_model.predict_proba(self.X_test)
        
        # Generate plots and metrics
        self.plot_feature_importance(self.rf_model, "Random Forest")
        self.plot_feature_importance(self.xgb_model, "XGBoost")
        
        self.plot_confusion_matrix(self.y_test, y_pred_rf, "Random Forest")
        self.plot_confusion_matrix(self.y_test, y_pred_xgb, "XGBoost", "OrRd")
        
        auc_rf = self.plot_roc_curve(self.y_test, y_proba_rf, "Random Forest")
        auc_xgb = self.plot_roc_curve(self.y_test, y_proba_xgb, "XGBoost")
        
        # Create and save model comparison
        self._save_model_comparison(y_pred_rf, y_pred_xgb, auc_rf, auc_xgb)
        
    def _save_model_comparison(self, y_pred_rf: pd.Series, y_pred_xgb: pd.Series,
                             auc_rf: float, auc_xgb: float) -> None:
        """Save model comparison metrics and plots.
        
        Args:
            y_pred_rf: Random Forest predictions
            y_pred_xgb: XGBoost predictions
            auc_rf: Random Forest AUC score
            auc_xgb: XGBoost AUC score
        """
        comparison_metrics = {
            "Accuracy": [
                accuracy_score(self.y_test, y_pred_rf),
                accuracy_score(self.y_test, y_pred_xgb)
            ],
            "Precision": [
                precision_score(self.y_test, y_pred_rf, average='weighted'),
                precision_score(self.y_test, y_pred_xgb, average='weighted')
            ],
            "Recall": [
                recall_score(self.y_test, y_pred_rf, average='weighted'),
                recall_score(self.y_test, y_pred_xgb, average='weighted')
            ],
            "F1-score": [
                f1_score(self.y_test, y_pred_rf, average='weighted'),
                f1_score(self.y_test, y_pred_xgb, average='weighted')
            ],
            "AUC": [auc_rf, auc_xgb]
        }
        
        comparison_df = pd.DataFrame(comparison_metrics, index=['Random Forest', 'XGBoost'])
        
        # Plot comparison heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(comparison_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title('Model Comparison: Random Forest vs XGBoost')
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification reports
        for model_name, y_pred in [("random_forest", y_pred_rf), ("xgboost", y_pred_xgb)]:
            report = classification_report(self.y_test, y_pred)
            with open(VISUALIZATION_DIR / f'{model_name}_report.txt', 'w') as f:
                f.write(f"{model_name.replace('_', ' ').title()} Classification Report:\n\n")
                f.write(report)
                
        # Save model comparison results
        with open(VISUALIZATION_DIR / 'model_comparison.txt', 'w') as f:
            f.write("Model Performance Summary:\n")
            f.write(comparison_df.to_string())
            
        # Print reports to console
        logger.info("\nRandom Forest Classification Report:")
        logger.info("\n" + classification_report(self.y_test, y_pred_rf))
        logger.info("\nXGBoost Classification Report:")
        logger.info("\n" + classification_report(self.y_test, y_pred_xgb))
        
    def run_analysis(self) -> None:
        """Run the complete predictive modeling analysis pipeline."""
        try:
            self.load_and_preprocess_data()
            self.prepare_train_test_data()
            self.train_models()
            self.evaluate_models()
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
            
    def main(self) -> None:
        """Main method to run the predictive modeling analysis."""
        try:
            logger.info("Starting predictive modeling analysis")
            self.run_analysis()
            logger.info("Predictive modeling analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in predictive modeling analysis: {str(e)}")
            raise

if __name__ == "__main__":
    PredictiveModeling().main()
