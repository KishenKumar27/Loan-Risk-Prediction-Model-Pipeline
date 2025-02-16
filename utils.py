import pandas as pd
import joblib
from config import PREPROCESSOR_PATH 

preprocessor = joblib.load(PREPROCESSOR_PATH)


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

def preprocess_input(input_data: dict):
    df = pd.DataFrame([input_data])
    
    # Apply the full preprocessing pipeline to match the training process
    processed_data = preprocessor.transform(df)
    
    return processed_data