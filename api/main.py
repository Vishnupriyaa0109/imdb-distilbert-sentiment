import os
from time import perf_counter

import numpy as np
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from transformers import AutoTokenizer
import onnxruntime as ort


# ============================================================
# Hugging Face Model
# ============================================================

MODEL_ID = "Kianaa0109/imdb-distilbert-sentiment"

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required.")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="IMDB DistilBERT Sentiment API",
    description=(
        "Production API for sentiment classification "
        "using the fine-tuned DistilBERT model with ONNX Runtime."
    ),
    version="2.0.0",
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
# Model Loading
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="model.int8.onnx",
    token=HF_TOKEN,
)

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)


# ============================================================
# Prediction
# ============================================================

def predict_sentiment(text: str):

    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=256,
    )

    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(
            np.int64
        )

    outputs = session.run(
        None,
        inputs,
    )

    logits = outputs[0]

    probabilities = np.exp(
        logits - np.max(logits, axis=-1, keepdims=True)
    )

    probabilities /= probabilities.sum(
        axis=-1,
        keepdims=True,
    )

    predicted_index = int(
        np.argmax(probabilities, axis=-1)[0]
    )

    confidence = float(
        probabilities[0][predicted_index]
    )

    label_map = {
        0: "negative",
        1: "positive",
    }

    sentiment = label_map.get(
        predicted_index,
        str(predicted_index),
    )

    return sentiment, confidence


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_ID,
        "runtime": "onnxruntime",
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = perf_counter()

    try:

        sentiment, confidence = predict_sentiment(
            request.text
        )

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
            "runtime": "onnxruntime",
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
