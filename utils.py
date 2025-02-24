import pandas as pd
import os
import pickle
import logging
import numpy as np
from config import MODEL_DIR, ENCODER_DIR 
import re
import logging

# Configure logging
logger = logging.getLogger("uvicorn")


def map_loan_status_to_risk(status):
    high_risk = [
        'Charged Off', 
        'Charged Off Paid Off', 
        'External Collection', 
        'Settled Bankruptcy', 
        'Settlement Paid Off', 
        'Settlement Pending Paid Off', 
        'Rejected'
    ]
    
    medium_risk = [
        'Internal Collection', 
        'Pending Rescind', 
        'Returned Item'
    ]
    
    low_risk = [
        'New Loan', 
        'Paid Off Loan', 
        'Pending Paid Off', 
        'Voided New Loan', 
        'Pending Application', 
        'Pending Application Fee'
    ]
    
    very_low_risk = [
        'CSR Voided New Loan', 
        'Credit Return Void', 
        'Customer Voided New Loan', 
        'Withdrawn Application'
    ]
    
    if status in high_risk:
        return 'High Risk'
    elif status in medium_risk:
        return 'Medium Risk'
    elif status in low_risk:
        return 'Low Risk'
    elif status in very_low_risk:
        return 'Very Low Risk'
    elif pd.isna(status):
        return 'Unknown Risk'
    else:
        return 'Unknown Risk'


def get_latest_models(model_dir, encoder_dir):
    """Find and return the latest preprocessor, model and label encoder based on timestamp."""
    model_files = [
        f for f in os.listdir(model_dir)
        if re.match(r"loan_risk_model_\d{8}_\d{6}\.pkl$", f)  # Validate format
    ]
    
    if not model_files:
        raise FileNotFoundError("No valid pipeline files found in the directory.")

    # Sort files by full timestamp
    model_files = sorted(model_files, reverse=True, key=lambda f: pd.to_datetime(
        '_'.join(f.split('_')[-2:]).split('.')[0].strip(), format='%Y%m%d_%H%M%S'
    ))
    
    latest_pipeline_path = os.path.join(model_dir, model_files[0])
    
    try:
        with open(latest_pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        # Extract the preprocessor and model separately
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        logging.info(f"Loaded pipeline from: {latest_pipeline_path}")
    except Exception as e:
        logging.error(f"Error loading pipeline: {e}", exc_info=True)
        raise RuntimeError("Failed to load the model pipeline.")
    
    # Find the latest label encoder
    encoder_files = [
        f for f in os.listdir(encoder_dir)
        if re.match(r"label_encoder_\d{8}_\d{6}\.pkl$", f)  # Validate format
    ]
    
    if encoder_files:
        encoder_files = sorted(encoder_files, reverse=True, key=lambda f: pd.to_datetime(
            '_'.join(f.split('_')[-2:]).split('.')[0].strip(), format='%Y%m%d_%H%M%S'
        ))
        latest_encoder_path = os.path.join(encoder_dir, encoder_files[0])
        
        try:
            with open(latest_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            logging.info(f"Loaded label encoder from: {latest_encoder_path}")
        except Exception as e:
            logging.error(f"Error loading label encoder: {e}", exc_info=True)
            label_encoder = None  # Return None if loading fails
    else:
        label_encoder = None
        logging.warning("No label encoder found. Returning None.")
    
    return preprocessor, model, label_encoder


def load_models():
    """Load the latest machine learning model and label encoder."""
    logger.info("Loading latest model and encoder...")
    return get_latest_models(MODEL_DIR, ENCODER_DIR)


def preprocess_input(loan_application, preprocessor):
    """Convert input data to DataFrame, apply preprocessing, and transform date columns."""
    logger.info("Preprocessing input data...")
    loan_df = pd.DataFrame([loan_application.dict()])

    # Drop unnecessary columns
    columns_to_drop = ["loanId", "anon_ss", "clarityFraudId"]
    loan_df = loan_df.drop(columns=columns_to_drop, errors='ignore')

    # Convert date columns and extract features
    date_columns = ["originatedDate", "applicationDate"]
    for col in date_columns:
        if col in loan_df.columns:
            loan_df[col] = pd.to_datetime(loan_df[col], errors='coerce')
            loan_df[f"{col}_year"] = loan_df[col].dt.year
            loan_df[f"{col}_month"] = loan_df[col].dt.month
            loan_df[f"{col}_day"] = loan_df[col].dt.day

    loan_df = loan_df.drop(columns=date_columns, errors='ignore')

    # Apply preprocessing
    preprocessed_data = preprocessor.transform(loan_df)
    return preprocessed_data.to_numpy() if isinstance(preprocessed_data, pd.DataFrame) else preprocessed_data


def make_prediction(preprocessed_data, model, label_encoder):
    """Make prediction using the trained model and map to labels."""
    logger.info("Making prediction...")
    prediction_encoded = np.array(model.predict(preprocessed_data)).reshape(-1)
    prediction_status = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_risk = map_loan_status_to_risk(prediction_status)
    logger.info(f"Predicted loan status: {prediction_status}, Mapped risk category: {prediction_risk}")
    return prediction_status, prediction_risk
