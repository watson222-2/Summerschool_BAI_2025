#!/usr/bin/env python
"""
FastMCP server exposing four tools:
  • train_logistics_model(data)
  • predict_next_shipment(last_timestamp)
  • echo(text)
  • do_web_request(url)
"""
import os
import logging
import joblib
import requests
import numpy as np
from typing import List, Tuple
from fastmcp import FastMCP  # FastMCP 2.0 SDK

# --- set up logging so your async tool can call logger.info() ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = "trained_model.joblib"
mcp = FastMCP(name="logistics")

@mcp.tool()
def train_logistics_model(data: List[Tuple[int, int]]) -> str:
    """Train & persist a linear model on (timestamp, value) pairs.
       data: List[Tuple[int, int]]
    """
    if len(data) < 10:
        return "Error: need ≥10 points."
    X = np.array([t for t, _ in data]).reshape(-1, 1)
    y = np.array([v for _, v in data])
    model = LinearRegression().fit(X, y)
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Model trained on {len(data)} points and saved to {MODEL_FILE}")
    return "✅ Model trained & saved."

@mcp.tool()
def predict_next_shipment(last_timestamp: int) -> str:
    """Load saved model & predict value for next timestamp.
       last_timestamp: int
    """
    if not os.path.exists(MODEL_FILE):
        return "Error: no model – train first."
    model = joblib.load(MODEL_FILE)
    pred = model.predict([[last_timestamp + 1]])[0]
    logger.info(f"Predicted for t={last_timestamp+1}: {pred:.2f}")
    return f"📦 Prediction for t={last_timestamp+1}: {pred:.2f}"

@mcp.tool()
def echo(text: str) -> str:
    """Return input verbatim.
    text: str
    """
    logger.info(f"Echo tool received: {text}")
    return text

@mcp.tool()
def do_web_request(url: str) -> str:
    """Given a URL, return the GET response text (first 2000 chars).
    url: str
    """
    response_text: str = requests.get(url, timeout=10).text
    snippet = response_text
    logger.info(f"do_web_request fetched {len(snippet)} chars from {url}")
    return snippet

if __name__ == "__main__":
    # Expose as streamable HTTP as recommended by FastMCP docs
    mcp.run(transport="streamable-http", port=8000)
