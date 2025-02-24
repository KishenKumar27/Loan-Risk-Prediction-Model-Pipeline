from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
import warnings
from utils import load_models, preprocess_input, make_prediction

# Suppress warnings
warnings.simplefilter("ignore")

# Initialize FastAPI app
app = FastAPI()

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()


class LoanApplication(BaseModel):
    loanId: str
    anon_ssn: str
    clarityFraudId: str
    loanAmount: float
    apr: float
    nPaidOff: int
    isFunded: bool
    state: str
    leadCost: float
    payFrequency: str
    originallyScheduledPaymentAmount: float
    originatedDate: str 
    applicationDate: str
    loan_to_payment_ratio: float
    is_monthly_payment: bool
    leadType: str
    fpStatus: str
    originated: bool
    hasCF: bool
    approved: bool



@app.post("/predict")
async def predict(loan_application: LoanApplication):
    try:
        preprocessor, model, label_encoder = load_models()
        preprocessed_data = preprocess_input(loan_application, preprocessor)
        loan_status, risk_category = make_prediction(preprocessed_data, model, label_encoder)
        return {"loanStatus": loan_status, "riskCategory": risk_category}
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.get('/health')
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
