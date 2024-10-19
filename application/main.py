from fastapi import FastAPI
from application.api import predict
from application.api import health

app = FastAPI()

app.include_router(predict.router)
app.include_router(health.router)
