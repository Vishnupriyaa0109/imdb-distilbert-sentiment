import os
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# ============================================================
# Hugging Face Model
# ============================================================

MODEL_ID = "Kianaa0109/imdb-distilbert-sentiment"

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="IMDB DistilBERT Sentiment API",
    description=(
        "Production API for sentiment classification "
        "using the fine-tuned DistilBERT model."
    ),
    version="1.0.0",
)


# ============================================================
# Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter(
    "sentiment_prediction_requests_total",
    "Total number of sentiment prediction requests.",
    ["sentiment"],
)

REQUEST_LATENCY = Histogram(
    "sentiment_prediction_latency_seconds",
    "Inference latency in seconds.",
)


# ============================================================
# Request Schema
# ============================================================

class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Movie review text.",
    )


# ============================================================
# Load Model from Private Hugging Face Hub
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1,
)


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_ID,
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = perf_counter()

    try:

        result = classifier(
            request.text,
            truncation=True,
            max_length=256,
        )[0]

        raw_label = result["label"]

        confidence = float(
            result["score"]
        )

        if raw_label in (
            "LABEL_1",
            "positive",
        ):

            sentiment = "positive"

        elif raw_label in (
            "LABEL_0",
            "negative",
        ):

            sentiment = "negative"

        else:

            sentiment = raw_label.lower()

        REQUEST_COUNT.labels(
            sentiment=sentiment
        ).inc()

        return {
            "sentiment": sentiment,
            "confidence": round(
                confidence,
                4,
            ),
            "model": MODEL_ID,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        elapsed = (
            perf_counter()
            - start_time
        )

        REQUEST_LATENCY.observe(
            elapsed
        )


# ============================================================
# Prometheus Metrics Endpoint
# ============================================================

metrics_app = make_asgi_app()

app.mount(
    "/metrics",
    metrics_app,
)