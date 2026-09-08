# ============================================================
# GOTHIC OCR — TEXT DECODER
# Model Output + NMS + Line Grouping + Text Reconstruction
# ============================================================

from pathlib import Path
import json

import numpy as np


class TextDecoder:
    """
    Decode GothicOCR TensorFlow Lite model output.

    Supported features:
    - YOLO-style output decoding
    - Normalized / pixel coordinate handling
    - Letterbox coordinate restoration
    - Class-aware NMS
    - Tile result merging
    - Detection sorting
    - Line grouping
    - Word spacing
    - Text reconstruction
    """

    INPUT_SIZE = 1024
    NUM_CLASSES = 25

    CONFIDENCE_THRESHOLD = 0.05
    NMS_IOU_THRESHOLD = 0.45

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        labels_path=None,
        confidence_threshold=None,
        nms_iou_threshold=None,
        space_threshold_factor=0.45,
    ):

        if labels_path is None:
            labels_path = (
                Path(__file__).resolve()
                .parent.parent
                / "data"
                / "labels.json"
            )

        self.labels_path = Path(labels_path)

        self.labels = self._load_labels()

        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.CONFIDENCE_THRESHOLD
        )

        self.nms_iou_threshold = (
            float(nms_iou_threshold)
            if nms_iou_threshold is not None
            else self.NMS_IOU_THRESHOLD
        )

        self.space_threshold_factor = float(
            space_threshold_factor
        )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.nms_iou_threshold
            <= 1.0
        ):
            raise ValueError(
                "nms_iou_threshold must be "
                "between 0 and 1."
            )

        if self.space_threshold_factor < 0:
            raise ValueError(
                "space_threshold_factor must "
                "be >= 0."
            )

    # ============================================================
    # LOAD LABELS
    # ============================================================

    def _load_labels(self):

        if not self.labels_path.is_file():
            raise FileNotFoundError(
                f"Labels file not found: "
                f"{self.labels_path}"
            )

        try:

            with self.labels_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                labels = json.load(file)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Invalid JSON in labels file: "
                f"{self.labels_path}"
            ) from exc

        if not isinstance(
            labels,
            list,
        ):
            raise ValueError(
                "labels.json must contain "
                "a list."
            )

        if len(labels) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} "
                f"labels, found {len(labels)}."
            )

        cleaned_labels = []

        for label in labels:

            if not isinstance(
                label,
                str,
            ):
                label = str(label)

            label = label.strip()

            if not label:
                raise ValueError(
                    "labels.json contains "
                    "an empty label."
                )

            cleaned_labels.append(
                label
            )

        return cleaned_labels

    # ============================================================
    # IOU
    # ============================================================

    @staticmethod
    def iou(
        box_a,
        box_b,
    ):

        ax1, ay1, ax2, ay2 = (
            float(box_a[0]),
            float(box_a[1]),
            float(box_a[2]),
            float(box_a[3]),
        )

        bx1, by1, bx2, by2 = (
            float(box_b[0]),
            float(box_b[1]),
            float(box_b[2]),
            float(box_b[3]),
        )

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        intersection_width = max(
            0.0,
            ix2 - ix1,
        )

        intersection_height = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = (
            max(
                0.0,
                ax2 - ax1,
            )
            * max(
                0.0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0.0,
                bx2 - bx1,
            )
            * max(
                0.0,
                by2 - by1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0.0:
            return 0.0

        return (
            intersection
            / union
        )

    # ============================================================
    # SORT DETECTIONS
    # ============================================================

    def sort_detections(
        self,
        detections,
    ):

        if not detections:
            return []

        return sorted(
            detections,
            key=lambda detection: (
                (
                    float(
                        detection["box"][1]
                    )
                    + float(
                        detection["box"][3]
                    )
                )
                / 2.0,
                float(
                    detection["box"][0]
                ),
            ),
        )

    # ============================================================
    # NMS
    # ============================================================

    def nms(
        self,
        detections,
    ):

        if not detections:
            return []

        class_groups = {}

        for detection in detections:

            class_id = int(
                detection["class_id"]
            )

            class_groups.setdefault(
                class_id,
                [],
            ).append(
                detection
            )

        kept = []

        for group in class_groups.values():

            remaining = sorted(
                group,
                key=lambda item: float(
                    item["score"]
                ),
                reverse=True,
            )

            while remaining:

                best = remaining.pop(
                    0
                )

                kept.append(
                    best
                )

                survivors = []

                for other in remaining:

                    overlap = self.iou(
                        best["box"],
                        other["box"],
                    )

                    if (
                        overlap
                        < self.nms_iou_threshold
                    ):
                        survivors.append(
                            other
                        )

                remaining = survivors

        return self.sort_detections(
            kept
        )

    # ============================================================
    # MERGE DETECTIONS
    # ============================================================

    def merge_detections(
        self,
        detections,
    ):
        """
        Merge detections from multiple tiles.

        Duplicate detections created by tile overlap
        are removed using global class-aware NMS.
        """

        if not detections:
            return []

        return self.nms(
            detections
        )

    # ============================================================
    # BOX TO MODEL CANVAS
    # ============================================================

    def _box_to_canvas(
        self,
        x,
        y,
        w,
        h,
    ):

        values = np.asarray(
            [
                x,
                y,
                w,
                h,
            ],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Box contains invalid values."
            )

        # Normalized coordinates
        if np.all(
            np.abs(values)
            <= 1.5
        ):

            canvas_x = (
                float(x)
                * self.INPUT_SIZE
            )

            canvas_y = (
                float(y)
                * self.INPUT_SIZE
            )

            canvas_w = (
                float(w)
                * self.INPUT_SIZE
            )

            canvas_h = (
                float(h)
                * self.INPUT_SIZE
            )

        # Pixel coordinates
        else:

            canvas_x = float(x)
            canvas_y = float(y)
            canvas_w = float(w)
            canvas_h = float(h)

        return (
            canvas_x,
            canvas_y,
            canvas_w,
            canvas_h,
        )

    # ============================================================
    # META VALUE
    # ============================================================

    @staticmethod
    def _meta_value(
        meta,
        names,
        default=None,
    ):

        if not isinstance(
            meta,
            dict,
        ):
            return default

        for name in names:

            if name in meta:
                return meta[name]

        return default

    # ============================================================
    # EXTRACT META
    # ============================================================

    def _extract_meta(
        self,
        meta=None,
        original_width=None,
        original_height=None,
        scale=None,
        pad_x=None,
        pad_y=None,
    ):
        """
        Support both decoder APIs:

        Old:
            decode(
                output,
                original_width,
                original_height,
                scale,
                pad_x,
                pad_y
            )

        New:
            decode(
                output,
                meta=meta
            )
        """

        if meta is not None:

            if original_width is None:

                original_width = (
                    self._meta_value(
                        meta,
                        [
                            "original_width",
                            "width",
                            "image_width",
                            "tile_width",
                        ],
                    )
                )

            if original_height is None:

                original_height = (
                    self._meta_value(
                        meta,
                        [
                            "original_height",
                            "height",
                            "image_height",
                            "tile_height",
                        ],
                    )
                )

            if scale is None:

                scale = self._meta_value(
                    meta,
                    [
                        "scale",
                        "resize_scale",
                    ],
                    1.0,
                )

            if pad_x is None:

                pad_x = self._meta_value(
                    meta,
                    [
                        "pad_x",
                        "padding_x",
                    ],
                    0.0,
                )

            if pad_y is None:

                pad_y = self._meta_value(
                    meta,
                    [
                        "pad_y",
                        "padding_y",
                    ],
                    0.0,
                )

        if original_width is None:
            original_width = self.INPUT_SIZE

        if original_height is None:
            original_height = self.INPUT_SIZE

        if scale is None:
            scale = 1.0

        if pad_x is None:
            pad_x = 0.0

        if pad_y is None:
            pad_y = 0.0

        try:

            original_width = int(
                original_width
            )

            original_height = int(
                original_height
            )

            scale = float(
                scale
            )

            pad_x = float(
                pad_x
            )

            pad_y = float(
                pad_y
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Invalid image metadata."
            ) from exc

        if (
            original_width <= 0
            or original_height <= 0
        ):
            raise ValueError(
                "original_width and "
                "original_height must be > 0."
            )

        if (
            not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError(
                f"Invalid image scale: "
                f"{scale}"
            )

        if not np.isfinite(
            [
                pad_x,
                pad_y,
            ]
        ).all():

            raise ValueError(
                "pad_x/pad_y contain "
                "invalid values."
            )

        return (
            original_width,
            original_height,
            scale,
            pad_x,
            pad_y,
        )

    # ============================================================
    # DECODE
    # ============================================================

    def decode(
        self,
        output,
        original_width=None,
        original_height=None,
        scale=None,
        pad_x=None,
        pad_y=None,
        meta=None,
    ):
        """
        Decode model output.

        Supports:

        decode(
            output,
            original_width=...,
            original_height=...,
            scale=...,
            pad_x=...,
            pad_y=...
        )

        And:

        decode(
            output,
            meta=meta
        )
        """

        (
            original_width,
            original_height,
            scale,
            pad_x,
            pad_y,
        ) = self._extract_meta(
            meta=meta,
            original_width=original_width,
            original_height=original_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(output)
        ):
            raise ValueError(
                "Model output contains "
                "NaN or Inf."
            )

        # --------------------------------------------------------
        # REMOVE BATCH DIMENSION
        # --------------------------------------------------------

        if output.ndim == 3:

            if output.shape[0] != 1:

                raise ValueError(
                    "Expected batch size 1, "
                    f"got shape {output.shape}"
                )

            output = output[0]

        if output.ndim != 2:

            raise ValueError(
                "Unexpected output shape: "
                f"{output.shape}"
            )

        expected_channels = (
            4
            + self.NUM_CLASSES
        )

        # --------------------------------------------------------
        # MODEL OUTPUT ORIENTATION
        # --------------------------------------------------------

        if (
            output.shape[0]
            == expected_channels
        ):

            predictions = output

        elif (
            output.shape[1]
            == expected_channels
        ):

            predictions = output.T

        else:

            raise ValueError(
                "Expected one output dimension "
                f"to contain {expected_channels} "
                "channels. Got: "
                f"{output.shape}"
            )

        # --------------------------------------------------------
        # EMPTY PREDICTIONS
        # --------------------------------------------------------

        if predictions.shape[1] == 0:

            return self.compose_result(
                []
            )

        boxes = predictions[:4]

        class_scores = predictions[
            4:
            4 + self.NUM_CLASSES
        ]

        scores = np.max(
            class_scores,
            axis=0,
        )

        class_ids = np.argmax(
            class_scores,
            axis=0,
        )

        positions = np.where(

            np.isfinite(scores)

            & (
                scores
                >= self.confidence_threshold
            )

        )[0]

        detections = []

        # ========================================================
        # DECODE DETECTIONS
        # ========================================================

        for position in positions:

            x = float(
                boxes[0, position]
            )

            y = float(
                boxes[1, position]
            )

            w = float(
                boxes[2, position]
            )

            h = float(
                boxes[3, position]
            )

            score = float(
                scores[position]
            )

            class_id = int(
                class_ids[position]
            )

            values = [
                x,
                y,
                w,
                h,
                score,
            ]

            if not np.isfinite(
                values
            ).all():
                continue

            if (
                w <= 0.0
                or h <= 0.0
            ):
                continue

            if not (
                0
                <= class_id
                < len(self.labels)
            ):
                continue

            (
                canvas_x,
                canvas_y,
                canvas_w,
                canvas_h,
            ) = self._box_to_canvas(
                x,
                y,
                w,
                h,
            )

            # ----------------------------------------------------
            # XYWH CENTER → XYXY
            # ----------------------------------------------------

            canvas_x1 = (
                canvas_x
                - canvas_w / 2.0
            )

            canvas_y1 = (
                canvas_y
                - canvas_h / 2.0
            )

            canvas_x2 = (
                canvas_x
                + canvas_w / 2.0
            )

            canvas_y2 = (
                canvas_y
                + canvas_h / 2.0
            )

            # ----------------------------------------------------
            # REMOVE LETTERBOX PADDING
            # ----------------------------------------------------

            x1 = (
                canvas_x1
                - pad_x
            ) / scale

            y1 = (
                canvas_y1
                - pad_y
            ) / scale

            x2 = (
                canvas_x2
                - pad_x
            ) / scale

            y2 = (
                canvas_y2
                - pad_y
            ) / scale

            # ----------------------------------------------------
            # CLIP TO ORIGINAL TILE
            # ----------------------------------------------------

            x1 = float(
                np.clip(
                    x1,
                    0.0,
                    float(
                        original_width
                    ),
                )
            )

            y1 = float(
                np.clip(
                    y1,
                    0.0,
                    float(
                        original_height
                    ),
                )
            )

            x2 = float(
                np.clip(
                    x2,
                    0.0,
                    float(
                        original_width
                    ),
                )
            )

            y2 = float(
                np.clip(
                    y2,
                    0.0,
                    float(
                        original_height
                    ),
                )
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "letter": self.labels[
                        class_id
                    ],
                    "score": score,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

        # --------------------------------------------------------
        # LOCAL NMS
        # --------------------------------------------------------

        detections = self.nms(
            detections
        )

        return self.compose_result(
            detections
        )

    # ============================================================
    # GROUP LINES
    # ============================================================

    def _group_lines(
        self,
        detections,
    ):

        if not detections:
            return []

        vertical = sorted(
            detections,
            key=lambda detection: (
                float(
                    detection["box"][1]
                )
                + float(
                    detection["box"][3]
                )
            )
            / 2.0,
        )

        lines = []

        for detection in vertical:

            box = detection["box"]

            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])

            center_y = (
                y1
                + y2
            ) / 2.0

            height = max(
                1.0,
                y2 - y1,
            )

            best_line = None

            best_distance = float(
                "inf"
            )

            for line in lines:

                line_center_y = (
                    line["center_y"]
                )

                average_height = (
                    line["avg_height"]
                )

                tolerance = max(

                    height * 0.50,

                    average_height * 0.60,

                    8.0,

                )

                distance = abs(
                    center_y
                    - line_center_y
                )

                if (
                    distance <= tolerance
                    and distance < best_distance
                ):

                    best_distance = distance

                    best_line = line

            # ----------------------------------------------------
            # EXISTING LINE
            # ----------------------------------------------------

            if best_line is not None:

                best_line["items"].append(
                    detection
                )

                count = len(
                    best_line["items"]
                )

                best_line[
                    "total_y_sum"
                ] += center_y

                best_line[
                    "total_h_sum"
                ] += height

                best_line[
                    "center_y"
                ] = (
                    best_line[
                        "total_y_sum"
                    ]
                    / count
                )

                best_line[
                    "avg_height"
                ] = (
                    best_line[
                        "total_h_sum"
                    ]
                    / count
                )

            # ----------------------------------------------------
            # NEW LINE
            # ----------------------------------------------------

            else:

                lines.append(
                    {
                        "center_y": float(
                            center_y
                        ),

                        "avg_height": float(
                            height
                        ),

                        "total_y_sum": float(
                            center_y
                        ),

                        "total_h_sum": float(
                            height
                        ),

                        "items": [
                            detection
                        ],
                    }
                )

        # --------------------------------------------------------
        # SORT LINES
        # --------------------------------------------------------

        lines.sort(
            key=lambda line: float(
                line["center_y"]
            )
        )

        # --------------------------------------------------------
        # SORT LETTERS INSIDE EACH LINE
        # --------------------------------------------------------

        for line in lines:

            line["items"].sort(
                key=lambda detection: float(
                    detection["box"][0]
                )
            )

        return lines

    # ============================================================
    # BUILD LINE TEXT
    # ============================================================

    def _build_line_text(
        self,
        items,
    ):

        if not items:
            return ""

        ordered = sorted(
            items,
            key=lambda detection: float(
                detection["box"][0]
            ),
        )

        characters = []

        for index, detection in enumerate(
            ordered
        ):

            characters.append(
                str(
                    detection["letter"]
                )
            )

            # ----------------------------------------------------
            # LAST LETTER
            # ----------------------------------------------------

            if (
                index
                >= len(ordered) - 1
            ):
                continue

            next_detection = ordered[
                index + 1
            ]

            current_box = (
                detection["box"]
            )

            next_box = (
                next_detection["box"]
            )

            current_x2 = float(
                current_box[2]
            )

            next_x1 = float(
                next_box[0]
            )

            gap = (
                next_x1
                - current_x2
            )

            current_width = max(
                1.0,
                float(
                    current_box[2]
                )
                - float(
                    current_box[0]
                ),
            )

            next_width = max(
                1.0,
                float(
                    next_box[2]
                )
                - float(
                    next_box[0]
                ),
            )

            average_width = (

                current_width
                + next_width

            ) / 2.0

            if (
                gap
                > (
                    average_width
                    * self.space_threshold_factor
                )
            ):

                characters.append(
                    " "
                )

        return "".join(
            characters
        )

    # ============================================================
    # COMPOSE RESULT
    # ============================================================

    def compose_result(
        self,
        detections,
    ):
        """
        Create final OCR result.

        Returns:
            {
                "text": str,
                "detections": list,
                "lines": list
            }
        """

        if not detections:

            return {
                "text": "",
                "detections": [],
                "lines": [],
            }

        ordered_detections = self.sort_detections(
            detections
        )

        lines = self._group_lines(
            ordered_detections
        )

        text_lines = []

        for line in lines:

            line_text = self._build_line_text(
                line["items"]
            )

            if line_text.strip():

                text_lines.append(
                    line_text
                )

        text = "\n".join(
            text_lines
        )

        return {
            "text": text,
            "detections": (
                ordered_detections
            ),
            "lines": lines,
        }

    # ============================================================
    # DECODE TEXT ONLY
    # ============================================================

    def decode_text(
        self,
        output,
        original_width=None,
        original_height=None,
        scale=None,
        pad_x=None,
        pad_y=None,
        meta=None,
    ):

        result = self.decode(

            output=output,

            original_width=original_width,

            original_height=original_height,

            scale=scale,

            pad_x=pad_x,

            pad_y=pad_y,

            meta=meta,

        )

        return result["text"]
