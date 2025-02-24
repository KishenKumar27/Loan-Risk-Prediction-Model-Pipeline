from sklearn.pipeline import Pipeline
from zenml import Model, step
import joblib
import os

@step
def model_loader(model_name: str) -> Pipeline:
    """
    Loads the current production model pipeline.

    Args:
        model_name: Name of the Model to load.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    # Define the model file path
    model_dir = "/models_pipeline"
    model_filename = os.path.join(model_dir, "loan_risk_model.pkl")
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file {model_filename} not found.")
    
    # Load the model artifacts
    model_artifacts = joblib.load(model_filename)
    model = model_artifacts["model"]
    preprocessor = model_artifacts["preprocessor"]
    label_encoder = model_artifacts["label_encoder"]
    
    return {"model": model, "preprocessor": preprocessor, "label_encoder": label_encoder}
