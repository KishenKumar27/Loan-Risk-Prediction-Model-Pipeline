import logging
from typing import Annotated
import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from typing import Dict, Optional
import pickle


# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="loan_risk_predictor",
    version=None,
    license="Apache 2.0",
    description="Loan risk prediction model using LightGBM.",
)

# Directory to save trained models
MODEL_DIR = "models_pipeline"
os.makedirs(MODEL_DIR, exist_ok=True)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: Optional[Dict] = None
) -> Annotated[Pipeline, ArtifactConfig(name="lgbm_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a LightGBM model for loan risk prediction using a scikit-learn pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the LightGBM model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
        
    # Define the LightGBM model training pipeline with SMOTE
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(**best_params))
    ])

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging for LightGBM
        mlflow.lightgbm.autolog()

        logging.info("Building and training the LightGBM model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")
        
        # Generate model filename based on timestamp or versioning logic
        model_filename = f"{MODEL_DIR}/loan_risk_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Save the model
        with open(model_filename, "wb") as f:
            pickle.dump(pipeline, f)
        logging.info(f"Model saved to {model_filename}")


    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline
