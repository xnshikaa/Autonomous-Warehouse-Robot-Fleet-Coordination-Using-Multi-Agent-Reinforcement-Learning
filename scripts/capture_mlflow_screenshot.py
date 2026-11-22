"""
MLflow dashboard screenshot capture.
Runs MLflow UI server in a background thread and captures via Chrome headless.
Everything runs in the SAME process so pure-python SQLAlchemy (no .pyd) works.
"""

import threading
import time
import os
import sys
import urllib.request
import subprocess
import signal

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 5002
MLFLOW_URL = f"http://127.0.0.1:{PORT}"
OUTPUT_DIR = "presentation_materials/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# 1. Import mlflow now (while disabled pyds are absent, pure-python fallback works)
import mlflow
from mlflow.server import _run_server

def run_server():
    """Run MLflow UI server in a thread."""
    try:
        _run_server(
            backend_store_uri="sqlite:///mlflow.db",
            default_artifact_root=None,
            serve_artifacts=False,
            artifacts_only=False,
            artifacts_destination=None,
            host="127.0.0.1",
            port=PORT,
            static_prefix=None,
            workers=1,
            gunicorn_opts=None,
            waitress_opts=None,
            expose_prometheus=None,
            app_name="basic-auth",
        )
    except Exception as e:
        # _run_server signature may vary by mlflow version
        print(f"_run_server with kwargs failed: {e}")
        # Fallback: direct flask run
        try:
            from mlflow.server import app as mlflow_app
            mlflow_app.run(host="127.0.0.1", port=PORT, threaded=True)
        except Exception as e2:
            print(f"flask fallback also failed: {e2}")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
print(f"Server thread started, waiting for http://127.0.0.1:{PORT}...")

# 2. Poll until ready
start_time = time.time()
server_ready = False
while time.time() - start_time < 30:
    try:
        with urllib.request.urlopen(MLFLOW_URL, timeout=2) as response:
            if response.status == 200:
                print(f"MLflow UI is UP! ({response.status})")
                server_ready = True
                break
    except Exception:
        time.sleep(1)

if not server_ready:
    print("Server did not come up in 30 seconds. Exiting.")
    sys.exit(1)

time.sleep(3)  # Let the React SPA fully hydrate

# 3. Take screenshots with Chrome headless
targets = [
    {
        "url": f"{MLFLOW_URL}/#/experiments/4",
        "output": os.path.join(OUTPUT_DIR, "mlflow_dashboard.png"),
        "name": "Experiment 4 Runs Table"
    },
    {
        "url": f"{MLFLOW_URL}/#/experiments/4/runs/3c34dbe51255429ca17adbd3ae94e7b2",
        "output": os.path.join(OUTPUT_DIR, "mlflow_run_detail.png"),
        "name": "QMIX N5 Low-LR Run Detail"
    },
    {
        "url": f"{MLFLOW_URL}/#/experiments/4/runs/85b173b15fc94640a5cc6900cd11ae8c",
        "output": os.path.join(OUTPUT_DIR, "mlflow_run_warm.png"),
        "name": "QMIX N5 Warm-Start Run Detail"
    }
]

for target in targets:
    print(f"\nCapturing: {target['name']}")
    out_abs = os.path.abspath(target["output"])
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--window-size=1920,1080",
        "--virtual-time-budget=12000",
        f"--screenshot={out_abs}",
        target["url"]
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    if os.path.exists(target["output"]):
        size = os.path.getsize(target["output"])
        print(f"  OK: {target['output']} ({size:,} bytes)")
    else:
        print(f"  FAILED. returncode={res.returncode}")
        if res.stderr:
            print(f"  stderr: {res.stderr[:300]}")

print("\nAll done. Screenshots saved to:", OUTPUT_DIR)
