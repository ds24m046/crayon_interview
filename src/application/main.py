from fastapi import FastAPI
from src.application.api import predict
from src.application.api import health

app = FastAPI()

app.include_router(predict.router)
app.include_router(health.router)
