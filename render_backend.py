"""
Render entrypoint for the Warehouse MARL WebSocket backend.

Exposes the FastAPI app as `app` for uvicorn.
Reads checkpoint paths from environment variables set in Render dashboard:
  QMIX_CHECKPOINT_N5   – path to qmix_n5 checkpoint file  (optional)
  QMIX_CHECKPOINT_N10  – path to qmix_n10 checkpoint file (optional)
  QMIX_CHECKPOINT_N20  – path to qmix_n20 checkpoint file (optional)
  IQL_CHECKPOINT_N5    – path to iql_n5 checkpoint dir    (optional)
  IQL_CHECKPOINT_N10   – path to iql_n10 checkpoint dir   (optional)
  IQL_CHECKPOINT_N20   – path to iql_n20 checkpoint dir   (optional)

Without checkpoints the backend runs in Rule-Based mode (always available).
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("render_backend")

# Ensure project root is on sys.path so src.* imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.api.warehouse_backend import create_app

logger.info("Creating warehouse FastAPI application…")
app = create_app(policy_source="auto")
logger.info("Warehouse backend ready — WebSocket endpoint: /ws")
