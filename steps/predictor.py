import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service for loan risk assessment.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """
    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)
    data.pop("columns", None)
    data.pop("index", None)

    # Define the columns the model expects
    expected_columns = [
        "loanAmount", "apr", "nPaidOff", "isFunded", "state", "leadCost", "payFrequency", "originallyScheduledPaymentAmount", "originatedDate_day", "applicationDate_day", 
        "loan_to_payment_ratio", "applicationDate_month", "originatedDate_year", "originatedDate_month", "is_monthly_payment", "leadType", "fpStatus", "originated",
        "hasCF", "applicationDate_year", "approved"
    ]
    
    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)
    
    return prediction
