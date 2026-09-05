"""Letterbox geometry. No dataset required."""

from __future__ import annotations

from PIL import Image

from ksdd2_transforms import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    Letterbox,
    letterbox_geometry,
)


def test_canvas_is_portrait_224x448() -> None:
    assert INPUT_WIDTH == 224
    assert INPUT_HEIGHT == 448
    assert INPUT_HEIGHT == 2 * INPUT_WIDTH


def test_median_ksdd2_shape_is_height_limited() -> None:
    geom = letterbox_geometry(229, 637)
    assert geom.scale == min(INPUT_WIDTH / 229, INPUT_HEIGHT / 637)
    assert geom.new_w <= INPUT_WIDTH
    assert geom.new_h <= INPUT_HEIGHT
    assert geom.new_h == INPUT_HEIGHT or geom.new_w == INPUT_WIDTH
    assert geom.new_w + 2 * geom.pad_x <= INPUT_WIDTH
    assert geom.new_h + 2 * geom.pad_y <= INPUT_HEIGHT


def test_aspect_ratio_is_preserved() -> None:
    geom = letterbox_geometry(229, 637)
    src_aspect = geom.src_w / geom.src_h
    new_aspect = geom.new_w / geom.new_h
    assert abs(new_aspect - src_aspect) < 0.02


def test_already_target_size_has_no_pad() -> None:
    geom = letterbox_geometry(INPUT_WIDTH, INPUT_HEIGHT)
    assert geom.scale == 1.0
    assert geom.new_w == INPUT_WIDTH
    assert geom.new_h == INPUT_HEIGHT
    assert geom.pad_x == 0
    assert geom.pad_y == 0


def test_square_image_pads_vertically() -> None:
    geom = letterbox_geometry(400, 400)
    assert geom.new_w == INPUT_WIDTH
    assert geom.new_h == INPUT_WIDTH
    assert geom.pad_x == 0
    assert geom.pad_y == (INPUT_HEIGHT - INPUT_WIDTH) // 2


def test_wide_range_of_ksdd2_shapes_fit_canvas() -> None:
    for src_w, src_h in ((184, 597), (184, 665), (241, 597), (241, 665)):
        geom = letterbox_geometry(src_w, src_h)
        assert 1 <= geom.new_w <= INPUT_WIDTH
        assert 1 <= geom.new_h <= INPUT_HEIGHT
        assert geom.pad_x >= 0
        assert geom.pad_y >= 0


def test_letterbox_output_size_is_always_the_canvas() -> None:
    out = Letterbox()(Image.new("RGB", (229, 637), (200, 10, 10)))
    assert out.size == (INPUT_WIDTH, INPUT_HEIGHT)
    assert out.mode == "RGB"
