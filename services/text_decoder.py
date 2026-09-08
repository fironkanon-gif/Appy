from pathlib import Path
import json
import math

import numpy as np


class TextDecoder:
    """
    Decodes YOLO-style OCR predictions into ordered Gothic text.

    Model:
        Input:  1024 x 1024
        Output: [1, 29, N]
        4 box values + 25 class scores
    """

    INPUT_SIZE = 1024
    NUM_CLASSES = 25

    CONFIDENCE_THRESHOLD = 0.05
    NMS_IOU_THRESHOLD = 0.45

    def __init__(
        self,
        labels_path=None,
        confidence_threshold=None,
        nms_iou_threshold=None,
        space_threshold_factor=0.45,
    ):
        self.confidence_threshold = (
            self.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        self.nms_iou_threshold = (
            self.NMS_IOU_THRESHOLD
            if nms_iou_threshold is None
            else float(nms_iou_threshold)
        )

        self.space_threshold_factor = float(space_threshold_factor)

        if labels_path is None:
            labels_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "labels.json"
            )

        self.labels_path = Path(labels_path)
        self.labels = self._load_labels()

    # ============================================================
    # LABELS
    # ============================================================

    def _load_labels(self):
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )

        with self.labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if not isinstance(labels, list):
            raise ValueError("labels.json must contain a JSON list.")

        if len(labels) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} labels, "
                f"got {len(labels)}."
            )

        cleaned = []

        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Every label must be a non-empty string.")

            cleaned.append(label)

        return cleaned

    # ============================================================
    # IOU
    # ============================================================

    @staticmethod
    def iou(box_a, box_b):
        """
        IoU for boxes in:
            [x1, y1, x2, y2]
        """

        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)

        intersection = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    # ============================================================
    # NMS
    # ============================================================

    def nms(self, detections):
        """
        Class-aware Non-Maximum Suppression.

        Detections are dictionaries containing:
            box
            score
            class_id
            char
        """

        if not detections:
            return []

        grouped = {}

        for detection in detections:
            class_id = int(detection["class_id"])
            grouped.setdefault(class_id, []).append(detection)

        kept = []

        for class_id, class_detections in grouped.items():
            ordered = sorted(
                class_detections,
                key=lambda item: float(item["score"]),
                reverse=True,
            )

            selected = []

            while ordered:
                best = ordered.pop(0)
                selected.append(best)

                remaining = []

                for candidate in ordered:
                    overlap = self.iou(
                        best["box"],
                        candidate["box"],
                    )

                    if overlap < self.nms_iou_threshold:
                        remaining.append(candidate)

                ordered = remaining

            kept.extend(selected)

        kept.sort(
            key=lambda item: (
                float(item["box"][1]),
                float(item["box"][0]),
            )
        )

        return kept

    # ============================================================
    # BOX CONVERSION
    # ============================================================

    def _box_to_canvas(self, box):
        """
        Converts normalized YOLO-style coordinates to 1024 canvas
        coordinates when necessary.

        Supported input:
            normalized values around 0..1
            pixel values around 0..1024
        """

        values = np.asarray(box, dtype=np.float32)

        if values.shape != (4,):
            raise ValueError(
                f"Expected box shape (4,), got {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Box contains NaN or infinite values.")

        if np.max(np.abs(values)) <= 1.5:
            values = values * self.INPUT_SIZE

        return values.astype(np.float32)

    # ============================================================
    # MODEL OUTPUT DECODER
    # ============================================================

    def decode(
        self,
        output,
        original_width,
        original_height,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    ):
        """
        Decode model output back into original image coordinates.
        """

        original_width = int(original_width)
        original_height = int(original_height)

        if original_width <= 0 or original_height <= 0:
            raise ValueError("Original image dimensions must be positive.")

        scale = float(scale)
        pad_x = float(pad_x)
        pad_y = float(pad_y)

        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a positive finite number.")

        if not math.isfinite(pad_x) or not math.isfinite(pad_y):
            raise ValueError("Padding must be finite.")

        predictions = np.asarray(output, dtype=np.float32)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Model output contains NaN or infinite values.")

        # Remove batch dimension.
        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError(
                    "Expected batch size 1 for model output."
                )

            predictions = predictions[0]

        if predictions.ndim != 2:
            raise ValueError(
                "Model output must have shape [1,29,N] "
                "or [1,N,29]."
            )

        # Accept both:
        #   [29, N]
        #   [N, 29]
        if predictions.shape[0] == 4 + self.NUM_CLASSES:
            pass

        elif predictions.shape[1] == 4 + self.NUM_CLASSES:
            predictions = predictions.T

        else:
            raise ValueError(
                f"Invalid model output shape: {predictions.shape}. "
                f"Expected [29,N] or [N,29]."
            )

        if predictions.shape[0] != 29:
            raise ValueError(
                f"Expected 29 prediction channels, "
                f"got {predictions.shape[0]}."
            )

        boxes = predictions[:4]
        class_scores = predictions[4:29]

        detections = []

        number_of_candidates = predictions.shape[1]

        for index in range(number_of_candidates):
            scores = class_scores[:, index]

            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if score < self.confidence_threshold:
                continue

            cx, cy, width, height = self._box_to_canvas(
                boxes[:, index]
            )

            if width <= 0 or height <= 0:
                continue

            x1 = float(cx - width / 2.0)
            y1 = float(cy - height / 2.0)
            x2 = float(cx + width / 2.0)
            y2 = float(cy + height / 2.0)

            # Reverse letterbox transformation.
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clip to original image.
            x1 = max(0.0, min(float(original_width), x1))
            y1 = max(0.0, min(float(original_height), y1))
            x2 = max(0.0, min(float(original_width), x2))
            y2 = max(0.0, min(float(original_height), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "class_id": class_id,
                    "char": self.labels[class_id],
                }
            )

        return self.compose_result(detections)

    # ============================================================
    # MERGE DETECTIONS
    # ============================================================

    def merge_detections(self, detections):
        """
        Merge overlapping duplicate detections.

        NMS is class-aware, so two different Gothic characters
        occupying different classes are preserved.
        """

        if not detections:
            return []

        normalized = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            if "box" not in detection:
                continue

            box = list(map(float, detection["box"]))

            if len(box) != 4:
                continue

            score = float(detection.get("score", 0.0))
            class_id = int(detection.get("class_id", 0))

            char = detection.get("char")

            if char is None:
                if 0 <= class_id < len(self.labels):
                    char = self.labels[class_id]
                else:
                    char = ""

            normalized.append(
                {
                    **detection,
                    "box": box,
                    "score": score,
                    "class_id": class_id,
                    "char": char,
                }
            )

        return self.nms(normalized)

    # ============================================================
    # LINE GROUPING
    # ============================================================

    def _group_lines(self, detections):
        """
        Group detections into text lines according to vertical position.
        """

        if not detections:
            return []

        ordered = sorted(
            detections,
            key=lambda item: (
                (item["box"][1] + item["box"][3]) / 2.0,
                item["box"][0],
            ),
        )

        lines = []

        for detection in ordered:
            x1, y1, x2, y2 = detection["box"]

            center_y = (y1 + y2) / 2.0
            height = max(1.0, y2 - y1)

            best_line = None
            best_distance = float("inf")

            for line in lines:
                line_center = line["center_y"]
                tolerance = max(
                    height,
                    line["average_height"],
                ) * 0.60

                distance = abs(center_y - line_center)

                if distance <= tolerance and distance < best_distance:
                    best_line = line
                    best_distance = distance

            if best_line is None:
                lines.append(
                    {
                        "detections": [detection],
                        "center_y": center_y,
                        "average_height": height,
                        "total_y_sum": center_y,
                        "total_h_sum": height,
                    }
                )
            else:
                best_line["detections"].append(detection)

                best_line["total_y_sum"] += center_y
                best_line["total_h_sum"] += height

                count = len(best_line["detections"])

                best_line["center_y"] = (
                    best_line["total_y_sum"] / count
                )

                best_line["average_height"] = (
                    best_line["total_h_sum"] / count
                )

        for line in lines:
            line["detections"].sort(
                key=lambda item: item["box"][0]
            )

        lines.sort(key=lambda line: line["center_y"])

        return lines

    # ============================================================
    # BUILD TEXT
    # ============================================================

    def _build_text(self, lines):
        """
        Convert grouped lines into final text.
        """

        if not lines:
            return ""

        text_lines = []

        for line in lines:
            detections = line["detections"]

            if not detections:
                continue

            detections = sorted(
                detections,
                key=lambda item: item["box"][0]
            )

            characters = []

            for index, detection in enumerate(detections):
                char = str(detection.get("char", ""))

                if not char:
                    continue

                if index == 0:
                    characters.append(char)
                    continue

                previous = detections[index - 1]

                previous_x1, previous_y1, previous_x2, previous_y2 = (
                    previous["box"]
                )

                current_x1, current_y1, current_x2, current_y2 = (
                    detection["box"]
                )

                gap = current_x1 - previous_x2

                previous_width = max(
                    1.0,
                    previous_x2 - previous_x1,
                )

                current_width = max(
                    1.0,
                    current_x2 - current_x1,
                )

                average_width = (
                    previous_width + current_width
                ) / 2.0

                if gap > average_width * self.space_threshold_factor:
                    characters.append(" ")

                characters.append(char)

            if characters:
                text_lines.append("".join(characters))

        return "\n".join(text_lines)

    # ============================================================
    # COMPOSE RESULT
    # ============================================================

    def compose_result(self, detections):
        """
        Build the final structured OCR result.

        Returns:
            {
                "text": "...",
                "detections": [...],
                "lines": [...]
            }
        """

        if not detections:
            return {
                "text": "",
                "detections": [],
                "lines": [],
            }

        merged = self.merge_detections(detections)

        if not merged:
            return {
                "text": "",
                "detections": [],
                "lines": [],
            }

        lines = self._group_lines(merged)
        text = self._build_text(lines)

        return {
            "text": text,
            "detections": merged,
            "lines": lines,
        }

    # ============================================================
    # PUBLIC TEXT DECODER
    # ============================================================

    def decode_text(self, detections):
        """
        Convenience method for converting detections directly to text.
        """

        result = self.compose_result(detections)
        return result["text"]    box_a = (0, 0, 20, 20)
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
