"""Streamlit dashboard runtime."""

import modal

from .common import data_volume, models_volume, webapp_image

app = modal.App("modal-social-sentiment-web")


@app.function(
    image=webapp_image,
    volumes={"/data": data_volume, "/models": models_volume},
)
@modal.web_server(port=8501, startup_timeout=60)
def dashboard():
    """Run Streamlit dashboard server."""
    import subprocess

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/root/src/app/main.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
    )

