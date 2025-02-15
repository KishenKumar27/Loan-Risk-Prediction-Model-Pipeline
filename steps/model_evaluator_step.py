import logging
import pandas as pd
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step
from typing import Dict

@step(enable_cache=False)
def model_evaluator_step(trained_model: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the trained model using ModelEvaluator and ClassificationModelEvaluationStrategy."""

    # Ensure input types are correct
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    # Extract model and preprocessor
    model = trained_model["model"]
    preprocessor = trained_model["preprocessor"]
    label_encoder = trained_model["label_encoder"]

    logging.info("Applying preprocessing to the test data.")
    X_test_transformed = preprocessor.transform(X_test)  # Preprocessing step

    # Use strategy pattern for evaluation
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())
    evaluation_metrics = evaluator.evaluate(model, X_test_transformed, y_test, label_encoder)

    logging.info(f"Model Evaluation Metrics: {evaluation_metrics}")

    return evaluation_metrics
