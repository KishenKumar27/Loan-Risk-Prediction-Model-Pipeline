import logging
from typing import Annotated, Dict, Optional

import mlflow
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

# Ensure the 'models' directory exists
model_dir = "/Users/kishenkumarsivalingam/Documents/MLE/models"
os.makedirs(model_dir, exist_ok=True)

model = Model(
    name="loan_risk_predictor",
    version=None,
    license="Apache 2.0",
    description="LightGBM model to predict loan risk status.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: Optional[Dict] = None
) -> Dict:
    """
    Builds and trains a LightGBM model using lgb.train for consistency with model tuning.
    
    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.
    
    Returns:
    Dict: Trained LightGBM model and preprocessing pipeline.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
    
    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")
    
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    # Handle NaN in y_train before encoding
    y_train = y_train.dropna()

    
    # Label Encoding for y_train
    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))
    
    # Apply Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train_encoded)
    logging.info("Applied Random Oversampling to balance class distribution.")


    X_resampled_processed = preprocessor.fit_transform(X_resampled)
    train_data = lgb.Dataset(X_resampled_processed, label=y_resampled)
    
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.lightgbm.autolog()
        logging.info("Training LightGBM model using lgb.train().")
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
        )
        logging.info("Model training completed.")
        
        # Define file paths
        model_filename = os.path.join(model_dir, "final_model.pkl")
        preprocessor_filename = os.path.join(model_dir, "preprocessor.pkl")
        label_encoder_filename = os.path.join(model_dir, "label_encoder.pkl")

        # Save each component separately
        joblib.dump(model, model_filename)
        joblib.dump(preprocessor, preprocessor_filename)
        joblib.dump(le, label_encoder_filename)

        logging.info(f"Model saved as {model_filename}")
        logging.info(f"Preprocessor saved as {preprocessor_filename}")
        logging.info(f"Label Encoder saved as {label_encoder_filename}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e
    
    finally:
        mlflow.end_run()

    return {"model": model, "preprocessor": preprocessor, "label_encoder": le}