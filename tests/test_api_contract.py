"""Hold-for-review API contract. Inspector is mocked; no checkpoint."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ksdd2_serve import (
    DECISION_CONTINUE,
    DECISION_HOLD,
    LABEL_DEFECTIVE,
    LABEL_OK,
    InspectionResult,
)
from ksdd2_api import create_app


class FakeInspector:
    def __init__(self, threshold: float, score: float) -> None:
        self.threshold = threshold
        self.score = score
        self.rule = (
            "highest_threshold_on_val_with_defective_recall>=0.938776; "
            "tie_break_higher_precision"
        )

    def meta(self) -> dict:
        return {
            "backbone": "mobilenet_v3_small",
            "input_width": 224,
            "input_height": 448,
            "threshold": self.threshold,
            "rule": self.rule,
            "best_epoch": 7,
            "best_val_pr_auc": 0.9072,
            "positive_means": DECISION_HOLD,
            "not_a_scrap_command": True,
        }

    def predict_image(self, image: Image.Image) -> InspectionResult:
        width, height = image.size
        positive = self.score >= self.threshold
        return InspectionResult(
            score=self.score,
            threshold=self.threshold,
            label=LABEL_DEFECTIVE if positive else LABEL_OK,
            decision=DECISION_HOLD if positive else DECISION_CONTINUE,
            positive=positive,
            width=width,
            height=height,
        )


def _png_bytes(size: tuple[int, int] = (32, 64)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (12, 24, 36)).save(buf, format="PNG")
    return buf.getvalue()


def _client(threshold: float, score: float) -> TestClient:
    return TestClient(create_app(FakeInspector(threshold, score)))


def test_health_with_injected_inspector(locked_threshold: float) -> None:
    response = _client(locked_threshold, 0.0).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_loaded"] is True


def test_meta_exposes_hold_flag_and_metrics_json_threshold(
    locked_threshold: float,
) -> None:
    body = _client(locked_threshold, 0.0).get("/meta").json()
    assert body["threshold"] == locked_threshold
    assert body["positive_means"] == "hold_for_review"
    assert body["not_a_scrap_command"] is True
    assert body["positive_means"] != "scrap"


def test_score_above_threshold_is_hold_for_review(locked_threshold: float) -> None:
    client = _client(locked_threshold, locked_threshold + 0.05)
    response = client.post(
        "/predict", files={"file": ("part.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "hold_for_review"
    assert body["label"] == "defective"
    assert body["positive"] is True
    assert body["threshold"] == locked_threshold
    assert "scrap" not in body["decision"]


def test_score_below_threshold_is_continue(locked_threshold: float) -> None:
    client = _client(locked_threshold, locked_threshold - 0.05)
    response = client.post(
        "/predict", files={"file": ("part.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "continue"
    assert body["label"] == "ok"
    assert body["positive"] is False
    assert body["threshold"] == locked_threshold


def test_score_equal_to_threshold_is_hold(locked_threshold: float) -> None:
    client = _client(locked_threshold, locked_threshold)
    body = client.post(
        "/predict", files={"file": ("part.png", _png_bytes(), "image/png")}
    ).json()
    assert body["decision"] == "hold_for_review"
    assert body["positive"] is True


def test_empty_upload_is_400(locked_threshold: float) -> None:
    response = _client(locked_threshold, 0.0).post(
        "/predict", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_unsupported_type_is_415(locked_threshold: float) -> None:
    response = _client(locked_threshold, 0.0).post(
        "/predict", files={"file": ("notes.txt", b"not-an-image", "text/plain")}
    )
    assert response.status_code == 415
