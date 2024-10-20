from fastapi import APIRouter
from src.application.services import predict
from src.application.schemas.predict import SinglePredict
from src.application.schemas.predict import BatchPredict


router = APIRouter()

@router.post("/single_predict")
async def single_predict(request: SinglePredict):
    
    email = request.email
    
    y_pred = predict.predict(email)
    predict.save(email)
    
    return y_pred

@router.post("/batch_predict")
async def batch_predict(request: BatchPredict):
    
    emails = request.emails
    
    y_preds = []
    for email in emails:
        y_pred = predict.predict(email)
        y_preds.append(y_pred)
        predict.save(email)
    
    return y_preds
