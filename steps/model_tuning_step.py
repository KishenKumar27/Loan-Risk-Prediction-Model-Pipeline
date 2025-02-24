import logging
from typing import Annotated, Tuple
import os
import mlflow
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from zenml import ArtifactConfig, step
from zenml.client import Client
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler


# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_tuning_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[dict, ArtifactConfig(name="tuned_hyperparameters")]:
    """
    Performs hyperparameter tuning on a LightGBM model using Optuna.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Tuple[Pipeline, dict]: The tuned scikit-learn pipeline and best hyperparameters.
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

    # Define the objective function for Optuna
    def objective(trial):
        """Objective function for Optuna hyperparameter tuning"""
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        }

        # Define the model pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lgb.LGBMClassifier(**params))])

        # Use stratified K-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy").mean()
        
        return score

    # Start MLflow run
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        logging.info("Starting hyperparameter tuning with Optuna.")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")

        # Train the best model with the optimal parameters
        best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lgb.LGBMClassifier(**best_params))])
        best_pipeline.fit(X_train, y_train)

        logging.info("Model tuning completed with the best parameters.")

        # Log parameters and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", study.best_value)
        
        # Save best parameters to a file
        output_dir = "tuned_params"
        
        # Save best parameters to a file with a timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"hyperparameters_{timestamp}.txt")
        
        with open(output_file, "w") as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Best hyperparameters saved to {output_file}")


    except Exception as e:
        logging.error(f"Error during model tuning: {e}")
        raise e

    finally:
        mlflow.end_run()

    return best_params
