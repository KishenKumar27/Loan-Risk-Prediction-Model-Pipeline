import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_tuning_step import model_tuning_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step
import logging


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="loan_risk_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="data.zip"
    )

    # Drop Irrelevant Columns and Handling Missing Values Step
    filled_data = handle_missing_values_step(df=raw_data)
    
    # Feature Engineering Step
    engineered_data = feature_engineering_step(filled_data)

    # Outlier Detection Step
    clean_data = outlier_detection_step(engineered_data, column_names=["apr","loanAmount","leadCost"])

    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="loanStatus")
    
    # Hyperparameter Tuning Step
    best_params = model_tuning_step(X_train=X_train, y_train=y_train)

    # # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train, best_params=best_params)

    # Model Evaluation Step
    evaluation_metrics = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
