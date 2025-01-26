import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data_analysis.predictmodels import PredictiveModeling


class TestPredictiveModeling(unittest.TestCase):
    """Test suite for the PredictiveModeling class."""

    def setUp(self):
        """Set up test environment before each test."""
        # This creates a small fake dataset that we'll use for testing
        # It contains:
        # - 15 patients (S1 to S15)
        # - Their tumor grades (0, 1, or 2)
        # - Three features about each patient (feature1, feature2, feature3)
        self.sample_data = pd.DataFrame({
            'sample': [f'S{i}' for i in range(1, 16)],
            'tumor_grade': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'feature2': [1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 1.5, 2.8, 3.2, 2.1, 3.1, 4.1, 1.8, 2.9, 3.3],
            'feature3': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X']
        })
        
        # This creates a fresh, empty model object before each test
        # Think of it like getting a clean sheet of paper before starting each homework problem
        self.predict_model = PredictiveModeling()

    def test_init(self):
        """
        This test checks if our model is set up correctly when we first create it.
        
        Just like when you start a new game, everything should be empty/zero at first.
        We check that:
        - No data is loaded yet (data should be None)
        - No processed data exists yet (data_cleaned should be None)
        - No training or testing data is created yet (X_train, X_test, etc. should be None)
        - We have an empty dictionary ready to store our label encoders
        """
        self.assertIsNone(self.predict_model.data)
        self.assertIsNone(self.predict_model.data_cleaned)
        self.assertIsNone(self.predict_model.X_train)
        self.assertIsNone(self.predict_model.X_test)
        self.assertIsNone(self.predict_model.y_train)
        self.assertIsNone(self.predict_model.y_test)
        self.assertIsInstance(self.predict_model.label_encoders, dict)
        self.assertEqual(len(self.predict_model.label_encoders), 0)

    def test_encode_categorical_variables(self):
        """
        This test checks if our program correctly converts text data into numbers.
        
        For example, if we have categories like 'small', 'medium', 'large',
        we need to convert them to numbers (0, 1, 2) because our machine learning
        models can only work with numbers, not text.
        
        We check that:
        1. The program creates a special converter (label encoder) for each text column
        2. After conversion, the data only contains numbers, not text
        """
        self.predict_model.data = self.sample_data
        self.predict_model.data_cleaned = self.sample_data.drop(columns=['sample']).fillna('None')
        self.predict_model._encode_categorical_variables()
        
        # Check if we created converters for our text columns
        self.assertIn('feature1', self.predict_model.label_encoders)
        self.assertIn('feature3', self.predict_model.label_encoders)
        
        # Check if text columns were converted to numbers
        self.assertIn(self.predict_model.data_cleaned['feature1'].dtype.name, ['int32', 'int64'])
        self.assertIn(self.predict_model.data_cleaned['feature3'].dtype.name, ['int32', 'int64'])

    def test_prepare_train_test_data(self):
        """
        This test checks if our data is properly split into training and testing sets.
        
        Think of it like studying for a test:
        - Training data is like your study materials
        - Testing data is like the actual test questions
        - SMOTE is like making photocopies of rare examples so we have enough to learn from
        
        We check that:
        1. The data is successfully split into training and testing parts
        2. After SMOTE, we have an equal number of examples for each tumor grade
           (so the model doesn't become biased towards more common grades)
        """
        self.predict_model.data = self.sample_data
        self.predict_model.data_cleaned = self.sample_data.drop(columns=['sample']).fillna('None')
        self.predict_model._encode_categorical_variables()
        
        # We create a simpler way to split the data for testing
        # This is like dividing a deck of cards: first 80% for training, last 20% for testing
        def mock_train_test_split(*args, **kwargs):
            X, y = args[0], args[1]
            split_idx = int(len(X) * 0.8)
            return (X.iloc[:split_idx], X.iloc[split_idx:],
                    y.iloc[:split_idx], y.iloc[split_idx:])
        
        import sklearn.model_selection
        original_split = sklearn.model_selection.train_test_split
        sklearn.model_selection.train_test_split = mock_train_test_split
        
        try:
            self.predict_model.prepare_train_test_data()
            
            # Check if we have both training and testing data
            self.assertIsNotNone(self.predict_model.X_train)
            self.assertIsNotNone(self.predict_model.X_test)
            self.assertIsNotNone(self.predict_model.y_train)
            self.assertIsNotNone(self.predict_model.y_test)
            
            # Check if SMOTE made the number of examples equal for each tumor grade
            unique_classes, counts = np.unique(self.predict_model.y_resampled, return_counts=True)
            self.assertEqual(len(np.unique(counts)), 1)  # All grades should have same count
        finally:
            sklearn.model_selection.train_test_split = original_split

    def test_train_models(self):
        """
        This test checks if our two machine learning models (Random Forest and XGBoost)
        can be trained successfully.
        
        It's like teaching two different students the same material and checking
        if they both learned something. We check that:
        1. Both models complete their training
        2. Both models can make predictions on new data
        3. The predictions are in the correct format (valid tumor grades)
        """
        self.predict_model.data = self.sample_data
        self.predict_model.data_cleaned = self.sample_data.drop(columns=['sample']).fillna('None')
        self.predict_model._encode_categorical_variables()
        self.predict_model.prepare_train_test_data()
        self.predict_model.train_models()
        
        # Check if models were created and trained
        self.assertIsNotNone(self.predict_model.rf_model)
        self.assertIsNotNone(self.predict_model.xgb_model)
        
        # Test if models can make predictions
        rf_preds = self.predict_model.rf_model.predict(self.predict_model.X_test)
        xgb_preds = self.predict_model.xgb_model.predict(self.predict_model.X_test)
        
        # Check if we got predictions for all test cases
        self.assertEqual(len(rf_preds), len(self.predict_model.X_test))
        self.assertEqual(len(xgb_preds), len(self.predict_model.X_test))

    def test_model_evaluation_metrics(self):
        """
        This test checks if our models can make good predictions and provide
        confidence scores.
        
        It's like grading a test where the student must:
        1. Give an answer (predict the tumor grade)
        2. Say how confident they are in their answer (probability)
        
        We check that:
        1. Predictions are valid tumor grades (0, 1, or 2)
        2. Confidence scores are between 0 and 1 (like percentages)
        3. Confidence scores for each prediction sum to 100%
        """
        self.predict_model.data = self.sample_data
        self.predict_model.data_cleaned = self.sample_data.drop(columns=['sample']).fillna('None')
        self.predict_model._encode_categorical_variables()
        self.predict_model.prepare_train_test_data()
        self.predict_model.train_models()
        
        # Get predictions from both models
        y_pred_rf = self.predict_model.rf_model.predict(self.predict_model.X_test)
        y_pred_xgb = self.predict_model.xgb_model.predict(self.predict_model.X_test)
        
        # Check that predictions are valid tumor grades
        for pred in y_pred_rf:
            self.assertIn(pred, [0, 1, 2])
        for pred in y_pred_xgb:
            self.assertIn(pred, [0, 1, 2])
        
        # Get confidence scores (probabilities) for each prediction
        rf_proba = self.predict_model.rf_model.predict_proba(self.predict_model.X_test)
        xgb_proba = self.predict_model.xgb_model.predict_proba(self.predict_model.X_test)
        
        # Check that we have confidence scores for all three possible grades
        self.assertEqual(rf_proba.shape[1], 3)  # Three tumor grades
        self.assertEqual(xgb_proba.shape[1], 3)
        
        # Check that confidence scores sum to 100% (1.0)
        for probs in rf_proba:
            self.assertTrue(np.allclose(sum(probs), 1.0))
        for probs in xgb_proba:
            self.assertTrue(np.allclose(sum(probs), 1.0))


if __name__ == '__main__':
    # This will work with both command line and IDE run button
    import sys
    from unittest import main
    main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)
