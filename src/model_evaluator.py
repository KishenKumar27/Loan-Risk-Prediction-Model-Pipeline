import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss, classification_report, confusion_matrix
)

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (BaseEstimator): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass

# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a regression model using R-squared and Mean Squared Error.

        Parameters:
        model (BaseEstimator): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Mean Squared Error": mse, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

# Concrete Strategy for Classification Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a classification model using Accuracy, Precision, Recall, and F1-score.

        Parameters:
        model (BaseEstimator): The trained classification model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing Accuracy, Precision, Recall, and F1-score.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC-ROC for multi-class (if probabilities are available)
        auc_roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted') if y_proba is not None else None
        
        # Log Loss (if probabilities are available)
        logloss = log_loss(y_test, y_proba) if y_proba is not None else None
        
        # Classification Report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC-ROC": auc_roc,
            "Log Loss": logloss,
            "Classification Report": class_report,
            "Confusion Matrix": conf_matrix
        }
        
        logging.info(f"Model Evaluation Metrics: {metrics}")
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

    def evaluate(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (BaseEstimator): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)

# Example usage
# if __name__ == "__main__":
#     pass