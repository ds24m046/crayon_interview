from fastapi import APIRouter
from ..services import predict


router = APIRouter()

@router.get("/health")
async def health_check():
    
    ok = {
        "status": "OK",
        "message": "Model ready for inference"
    }
    
    nok = {
        "status": "ERROR",
        "message": "Model is not ready!"
    }
    
    if not hasattr(predict.predict, "keywords"):
        return nok
    
    if not 'predictor' in predict.predict.keywords:
        return nok
    
    if predict.predict.keywords['predictor'] is None:
        return nok
    
    return ok
