import logging
from typing import Tuple, Dict
import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict:
    """
    Evaluates the trained LightGBM model using ModelEvaluator and ClassificationModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing classification evaluation metrics.
    float: The accuracy of the model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply preprocessing
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Make predictions
    model = trained_model.named_steps["model"]

    # Initialize the evaluator with classification strategy
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())
    
    # Perform evaluation
    evaluation_metrics = evaluator.evaluate(model, X_test_processed, y_test)
    
    logging.info(f"Evaluation Metrics: {evaluation_metrics}")
    
    # Save the evaluation metrics to a text file
    #output_file = '/Users/kishenkumarsivalingam/Documents/MLE 2/evaluation_metrics.txt'
    
    output_dir = "evaluation_metrics"
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"metrics_{timestamp}.txt")

    try:
        with open(output_file, "w") as file:
            for metric, value in evaluation_metrics.items():
                file.write(f"{metric}: {value}\n")
        logging.info(f'Evaluation metrics saved to {output_file}')
    except Exception as e:
        logging.error(f"Failed to save evaluation metrics to file: {e}")

    # try:
    #     with open(output_file, "a") as file:
    #         for metric, value in evaluation_metrics.items():
    #             file.write(f"{metric}: {value}\n")
    #         file.write("\n")  # Add a newline for separation between different evaluations
    # except Exception as e:
    #     logging.error(f"Failed to save evaluation metrics to file: {e}")

    return evaluation_metrics
