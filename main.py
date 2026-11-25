
from fastapi import FastAPI
from classification.ingestion_adapter import router

app = FastAPI(
    title="AuraStream",
    description="Multimodal Disaster Intelligence Platform",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "Welcome to AuraStream",
        "status": "running"
    }
