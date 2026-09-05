"""Local FastAPI app for the locked image-level hold flag.

Positive = hold for review. Not a scrap command, not a box, not an
interlock. The serving path reads the cutoff from metrics.json.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from ksdd2_serve import Inspector

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "application/octet-stream"}

UI_PATH = Path(__file__).with_name("ksdd2_api_ui.html")


def create_app(inspector: Inspector | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.inspector is None:
            app.state.inspector = Inspector()
        yield

    app = FastAPI(
        title="KSDD2 surface inspection",
        description=(
            "Image-level defective vs ok. Positive = hold for review. "
            "Portfolio demo, not a plant-ready system."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.inspector = inspector

    @app.get("/health")
    def health() -> dict[str, bool | str]:
        loaded = app.state.inspector is not None
        return {"ok": loaded, "model_loaded": loaded}

    @app.get("/meta")
    def meta() -> dict:
        return app.state.inspector.meta()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if file.content_type and file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="Send a PNG or JPEG image.")
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty upload.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image larger than 10 MB.")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                result = app.state.inspector.predict_image(image)
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Could not read image.") from exc
        return result.as_dict()

    @app.get("/", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    return app


app = create_app()
