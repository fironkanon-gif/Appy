# ============================================================
# GOTHIC OCR — REAL DECODER TESTS
# ============================================================

from pathlib import Path

import numpy as np
import pytest

from services.text_decoder import TextDecoder


# ============================================================
# HELPERS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "labels.json"


def make_detection(
    class_id=0,
    score=0.9,
    box=(10.0, 10.0, 30.0, 30.0),
    letter=None,
):
    if letter is None:
        letter = str(class_id)

    return {
        "class_id": class_id,
        "letter": letter,
        "score": score,
        "box": box,
    }


@pytest.fixture
def decoder():
    return TextDecoder(
        labels_path=LABELS_PATH
    )


# ============================================================
# LABELS
# ============================================================

def test_labels_file_has_25_classes(decoder):
    assert len(decoder.labels) == 25


def test_labels_are_non_empty(decoder):
    assert all(
        isinstance(label, str)
        and label.strip()
        for label in decoder.labels
    )


# ============================================================
# IOU
# ============================================================

def test_iou_identical_boxes(decoder):
    box = (10, 10, 30, 30)

    assert decoder.iou(
        box,
        box,
    ) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes(decoder):
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 30, 30)

    assert decoder.iou(
        box_a,
        box_b,
    ) == pytest.approx(0.0)


def test_iou_partial_overlap(decoder):
    box_a = (0, 0, 20, 20)
    box_b = (10, 10, 30, 30)

    # intersection = 100
    # union = 700
    expected = 100.0 / 700.0

    assert decoder.iou(
        box_a,
        box_b,
    ) == pytest.approx(
        expected
    )


# ============================================================
# NMS
# ============================================================

def test_nms_keeps_highest_confidence(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            score=0.95,
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=0,
            score=0.70,
            box=(11, 11, 31, 31),
        ),
    ]

    result = decoder.nms(
        detections
    )

    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(
        0.95
    )


def test_nms_keeps_different_classes(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            score=0.90,
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=1,
            score=0.85,
            box=(11, 11, 31, 31),
        ),
    ]

    result = decoder.nms(
        detections
    )

    assert len(result) == 2


def test_nms_keeps_non_overlapping_same_class(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            score=0.90,
            box=(0, 0, 20, 20),
        ),
        make_detection(
            class_id=0,
            score=0.85,
            box=(40, 40, 60, 60),
        ),
    ]

    result = decoder.nms(
        detections
    )

    assert len(result) == 2


def test_nms_empty_input(decoder):
    assert decoder.nms([]) == []


# ============================================================
# SORTING
# ============================================================

def test_sort_detections_by_y_then_x(
    decoder,
):
    detections = [
        make_detection(
            box=(100, 50, 120, 70),
        ),
        make_detection(
            box=(20, 10, 40, 30),
        ),
        make_detection(
            box=(50, 10, 70, 30),
        ),
    ]

    result = decoder.sort_detections(
        detections
    )

    assert result[0]["box"][0] == 20
    assert result[1]["box"][0] == 50
    assert result[2]["box"][0] == 100


# ============================================================
# BOX CONVERSION
# ============================================================

def test_normalized_box_conversion(
    decoder,
):
    x, y, w, h = decoder._box_to_canvas(
        0.5,
        0.5,
        0.25,
        0.25,
    )

    assert x == pytest.approx(
        512.0
    )

    assert y == pytest.approx(
        512.0
    )

    assert w == pytest.approx(
        256.0
    )

    assert h == pytest.approx(
        256.0
    )


def test_pixel_box_conversion(
    decoder,
):
    x, y, w, h = decoder._box_to_canvas(
        512,
        400,
        100,
        120,
    )

    assert x == pytest.approx(
        512.0
    )

    assert y == pytest.approx(
        400.0
    )

    assert w == pytest.approx(
        100.0
    )

    assert h == pytest.approx(
        120.0
    )


# ============================================================
# COMPOSE RESULT
# ============================================================

def test_compose_result_empty(
    decoder,
):
    result = decoder.compose_result(
        []
    )

    assert result["text"] == ""
    assert result["detections"] == []
    assert result["lines"] == []


def test_compose_result_orders_letters(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            score=0.9,
            box=(40, 10, 60, 30),
        ),
        make_detection(
            class_id=1,
            letter="B",
            score=0.9,
            box=(10, 10, 30, 30),
        ),
    ]

    result = decoder.compose_result(
        detections
    )

    assert result["text"] == "BA"


# ============================================================
# LINE GROUPING
# ============================================================

def test_group_lines_same_line(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=1,
            letter="B",
            box=(40, 11, 60, 31),
        ),
    ]

    lines = decoder._group_lines(
        detections
    )

    assert len(lines) == 1
    assert len(
        lines[0]["items"]
    ) == 2


def test_group_lines_two_lines(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=1,
            letter="B",
            box=(10, 100, 30, 120),
        ),
    ]

    lines = decoder._group_lines(
        detections
    )

    assert len(lines) == 2


# ============================================================
# WORD SPACING
# ============================================================

def test_build_text_adds_space_for_large_gap(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=1,
            letter="B",
            box=(70, 10, 90, 30),
        ),
    ]

    result = decoder.compose_result(
        detections
    )

    assert result["text"] == "A B"


def test_build_text_no_space_for_small_gap(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            box=(10, 10, 30, 30),
        ),
        make_detection(
            class_id=1,
            letter="B",
            box=(32, 10, 52, 30),
        ),
    ]

    result = decoder.compose_result(
        detections
    )

    assert result["text"] == "AB"


# ============================================================
# MERGING TILE RESULTS
# ============================================================

def test_merge_detections_removes_tile_duplicate(
    decoder,
):
    """
    محاكاة حرف ظهر في Tile 1 وTile 2
    بسبب الـOverlap.
    """

    detections = [
        make_detection(
            class_id=0,
            letter="A",
            score=0.95,
            box=(100, 100, 140, 140),
        ),
        make_detection(
            class_id=0,
            letter="A",
            score=0.80,
            box=(102, 102, 142, 142),
        ),
    ]

    result = decoder.merge_detections(
        detections
    )

    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(
        0.95
    )


def test_merge_detections_preserves_different_letters(
    decoder,
):
    detections = [
        make_detection(
            class_id=0,
            letter="A",
            score=0.95,
            box=(100, 100, 140, 140),
        ),
        make_detection(
            class_id=1,
            letter="B",
            score=0.90,
            box=(102, 102, 142, 142),
        ),
    ]

    result = decoder.merge_detections(
        detections
    )

    assert len(result) == 2


# ============================================================
# DECODE OUTPUT VALIDATION
# ============================================================

def test_decode_rejects_invalid_image_size(
    decoder,
):
    output = np.zeros(
        (1, 29, 1),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):
        decoder.decode(
            output=output,
            original_width=0,
            original_height=100,
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
        )


def test_decode_rejects_invalid_scale(
    decoder,
):
    output = np.zeros(
        (1, 29, 1),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):
        decoder.decode(
            output=output,
            original_width=100,
            original_height=100,
            scale=0.0,
            pad_x=0.0,
            pad_y=0.0,
        )


def test_decode_rejects_nan_output(
    decoder,
):
    output = np.zeros(
        (1, 29, 1),
        dtype=np.float32,
    )

    output[0, 0, 0] = np.nan

    with pytest.raises(
        ValueError
    ):
        decoder.decode(
            output=output,
            original_width=100,
            original_height=100,
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
        )


# ============================================================
# REALISTIC OUTPUT SHAPE
# ============================================================

def test_decode_accepts_model_output_shape(
    decoder,
):
    """
    اختبار أن decoder يتعامل مع شكل
    نموذج GothicOCR الحقيقي.

    لا نفترض أن النموذج سيكتشف حرفًا؛
    نختبر فقط صحة الشكل والمسار.
    """

    output = np.zeros(
        (1, 29, 21504),
        dtype=np.float32,
    )

    result = decoder.decode(
        output=output,
        original_width=1024,
        original_height=1024,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )

    assert isinstance(
        result,
        dict,
    )

    assert "text" in result
    assert "detections" in result
    assert "lines" in result

    assert result["text"] == ""
    assert result["detections"] == []
