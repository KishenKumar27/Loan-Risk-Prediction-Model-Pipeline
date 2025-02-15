import pandas as pd
from zenml import step

@step
def dynamic_importer() -> str:
    """Dynamically imports loan data for testing the model."""
    file_path = "/Users/kishenkumarsivalingam/Documents/MLE/extracted_data/data/loan.csv"
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select a subset of columns for testing (customize as needed)
    selected_columns = [
        "loanId", "payFrequency", "apr", "applicationDate", "loanStatus", 
        "loanAmount", "approved", "isFunded", "state", "leadType"
    ]
    df = df[selected_columns]
    
    # Sample a small subset of rows to keep it manageable
    df_sample = df.sample(n=5, random_state=42)
    
    # Convert the DataFrame to a JSON string
    json_data = df_sample.to_json(orient="split")
    
    return json_data