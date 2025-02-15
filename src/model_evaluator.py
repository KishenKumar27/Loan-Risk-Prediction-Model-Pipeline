import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (ClassifierMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass

# Concrete Strategy for Classification Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, label_encoder) -> dict:
        """Evaluates a classification model with Accuracy, Precision, Recall, and F1-score."""

        # Convert string labels to numeric labels
        y_test = y_test.dropna()
        y_test_encoded = label_encoder.transform(y_test)
        
        # Get all classes from the LabelEncoder
        all_classes = label_encoder.classes_

        y_pred_proba = model.predict(X_test)
        
        # Convert probabilities to class labels
        y_pred = y_pred_proba.argmax(axis=1)
        
        # Binarize y_test_encoded to match y_pred_proba shape
        y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(len(all_classes)))


        metrics = {
            "Accuracy": accuracy_score(y_test_encoded, y_pred),
            "Precision": precision_score(y_test_encoded, y_pred, average="weighted"),
            "Recall": recall_score(y_test_encoded, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test_encoded, y_pred, average="weighted"),
            "AUC": roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr", average="weighted") if len(all_classes) > 2 else roc_auc_score(y_test_encoded, y_pred_proba[:, 1]),
            "LogLoss": log_loss(y_test_binarized, y_pred_proba),
            "Classification Report": classification_report(y_test_encoded, y_pred, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test_encoded, y_pred).tolist()
        }
        
        return metrics

# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, label_encoder) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (ClassifierMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        metrics = self._strategy.evaluate_model(model, X_test, y_test, label_encoder)
        
        return metrics