# ============================================================
# GOTHIC OCR — TEXT DECODER
# Compatible with:
# - TFLite YOLO output [1, 29, 21504]
# - ImageService metadata
# - Tiled inference
# - Global NMS
# - Text reconstruction
# ============================================================

from pathlib import Path
import json

import numpy as np


class TextDecoder:
    """
    Decode GothicOCR model output.

    Expected model output:
        [1, 29, 21504]

    29 channels:
        4 bounding box values
        25 class scores
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

        if labels_path is None:

            labels_path = (
                Path(__file__)
                .resolve()
                .parent
                .parent
                / "data"
                / "labels.json"
            )

        self.labels_path = Path(
            labels_path
        )

        self.labels = (
            self._load_labels()
        )

        self.confidence_threshold = (

            float(
                confidence_threshold
            )

            if confidence_threshold
            is not None

            else self.CONFIDENCE_THRESHOLD
        )

        self.nms_iou_threshold = (

            float(
                nms_iou_threshold
            )

            if nms_iou_threshold
            is not None

            else self.NMS_IOU_THRESHOLD
        )

        self.space_threshold_factor = (
            float(
                space_threshold_factor
            )
        )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):

            raise ValueError(
                "confidence_threshold "
                "must be between 0 and 1."
            )

        if not (
            0.0
            <= self.nms_iou_threshold
            <= 1.0
        ):

            raise ValueError(
                "nms_iou_threshold "
                "must be between 0 and 1."
            )

        if (
            self.space_threshold_factor
            < 0.0
        ):

            raise ValueError(
                "space_threshold_factor "
                "must be >= 0."
            )

    # ============================================================
    # LOAD LABELS
    # ============================================================

    def _load_labels(self):

        if not self.labels_path.is_file():

            raise FileNotFoundError(
                "Labels file not found: "
                f"{self.labels_path}"
            )

        try:

            with self.labels_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                labels = json.load(
                    file
                )

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

        if len(labels) != (
            self.NUM_CLASSES
        ):

            raise ValueError(
                f"Expected "
                f"{self.NUM_CLASSES} labels, "
                f"found {len(labels)}."
            )

        labels = [

            str(label)

            for label
            in labels

        ]

        if any(
            not label
            for label
            in labels
        ):

            raise ValueError(
                "labels.json contains "
                "an empty label."
            )

        return labels

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

            *

            max(
                0.0,
                ay2 - ay1,
            )

        )

        area_b = (

            max(
                0.0,
                bx2 - bx1,
            )

            *

            max(
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
                detection[
                    "class_id"
                ]
            )

            class_groups.setdefault(
                class_id,
                [],
            ).append(
                detection
            )

        kept = []

        for group in (
            class_groups.values()
        ):

            group = sorted(

                group,

                key=lambda item:
                float(
                    item.get(
                        "score",
                        0.0,
                    )
                ),

                reverse=True,

            )

            while group:

                best = group.pop(
                    0
                )

                kept.append(
                    best
                )

                remaining = []

                for other in group:

                    overlap = self.iou(

                        best[
                            "box"
                        ],

                        other[
                            "box"
                        ],

                    )

                    if overlap < (

                        self
                        .nms_iou_threshold

                    ):

                        remaining.append(
                            other
                        )

                group = remaining

        kept.sort(

            key=lambda item: (

                (
                    float(
                        item[
                            "box"
                        ][1]
                    )

                    +

                    float(
                        item[
                            "box"
                        ][3]
                    )

                )

                / 2.0,

                float(
                    item[
                        "box"
                    ][0]
                ),

            )

        )

        return kept

    # ============================================================
    # BOX NORMALIZATION
    # ============================================================

    def _box_to_canvas(
        self,
        x,
        y,
        width,
        height,
    ):

        values = np.array(

            [
                x,
                y,
                width,
                height,
            ],

            dtype=np.float32,

        )

        if np.all(
            np.abs(values)
            <= 1.5
        ):

            canvas_x = (
                x
                * self.INPUT_SIZE
            )

            canvas_y = (
                y
                * self.INPUT_SIZE
            )

            canvas_width = (
                width
                * self.INPUT_SIZE
            )

            canvas_height = (
                height
                * self.INPUT_SIZE
            )

        else:

            canvas_x = x

            canvas_y = y

            canvas_width = width

            canvas_height = height

        return (

            float(
                canvas_x
            ),

            float(
                canvas_y
            ),

            float(
                canvas_width
            ),

            float(
                canvas_height
            ),

        )

    # ============================================================
    # EXTRACT IMAGE METADATA
    # ============================================================

    @staticmethod
    def _extract_meta(
        meta,
    ):

        if meta is None:

            raise ValueError(
                "Image metadata is required."
            )

        if not isinstance(
            meta,
            dict,
        ):

            raise ValueError(
                "Image metadata must "
                "be a dictionary."
            )

        original_width = meta.get(
            "original_width"
        )

        original_height = meta.get(
            "original_height"
        )

        if original_width is None:

            original_width = meta.get(
                "width"
            )

        if original_height is None:

            original_height = meta.get(
                "height"
            )

        scale = meta.get(
            "scale"
        )

        pad_x = meta.get(
            "pad_x",
            0.0,
        )

        pad_y = meta.get(
            "pad_y",
            0.0,
        )

        if original_width is None:

            raise ValueError(
                "Metadata missing "
                "original_width."
            )

        if original_height is None:

            raise ValueError(
                "Metadata missing "
                "original_height."
            )

        if scale is None:

            raise ValueError(
                "Metadata missing scale."
            )

        return (

            int(
                original_width
            ),

            int(
                original_height
            ),

            float(
                scale
            ),

            float(
                pad_x
            ),

            float(
                pad_y
            ),

        )

    # ============================================================
    # DECODE
    # ============================================================

    def decode(
        self,
        output,
        meta,
    ):

        (
            original_width,
            original_height,
            scale,
            pad_x,
            pad_y,
        ) = self._extract_meta(
            meta
        )

        if (
            original_width <= 0
            or original_height <= 0
        ):

            raise ValueError(
                "Invalid original "
                "image dimensions."
            )

        if (
            not np.isfinite(
                scale
            )
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

        output = np.asarray(

            output,

            dtype=np.float32,

        )

        if not np.all(
            np.isfinite(
                output
            )
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
        # OUTPUT FORMAT
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

                "Expected one output "
                "dimension to contain "

                f"{expected_channels} channels. "

                f"Got: {output.shape}"

            )

        # --------------------------------------------------------
        # BOXES + CLASS SCORES
        # --------------------------------------------------------

        boxes = predictions[
            :4
        ]

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

            np.isfinite(
                scores
            )

            &

            (
                scores
                >= self.confidence_threshold
            )

        )[0]

        detections = []

        # --------------------------------------------------------
        # DECODE DETECTIONS
        # --------------------------------------------------------

        for position in positions:

            x = float(

                boxes[
                    0,
                    position
                ]

            )

            y = float(

                boxes[
                    1,
                    position
                ]

            )

            width = float(

                boxes[
                    2,
                    position
                ]

            )

            height = float(

                boxes[
                    3,
                    position
                ]

            )

            score = float(

                scores[
                    position
                ]

            )

            class_id = int(

                class_ids[
                    position
                ]

            )

            if not np.isfinite(

                [
                    x,
                    y,
                    width,
                    height,
                    score,
                ]

            ).all():

                continue

            if (

                width <= 0.0
                or height <= 0.0

            ):

                continue

            if not (

                0
                <= class_id
                < len(
                    self.labels
                )

            ):

                continue

            (
                canvas_x,
                canvas_y,
                canvas_width,
                canvas_height,
            ) = self._box_to_canvas(

                x,
                y,
                width,
                height,

            )

            canvas_x1 = (

                canvas_x
                - canvas_width / 2.0

            )

            canvas_y1 = (

                canvas_y
                - canvas_height / 2.0

            )

            canvas_x2 = (

                canvas_x
                + canvas_width / 2.0

            )

            canvas_y2 = (

                canvas_y
                + canvas_height / 2.0

            )

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

            detections.append({

                "class_id": class_id,

                "letter": self.labels[
                    class_id
                ],

                "score": score,

                "box": [

                    x1,
                    y1,
                    x2,
                    y2,

                ],

            })

        # Local NMS first.
        detections = self.nms(
            detections
        )

        lines = self._group_lines(
            detections
        )

        return {

            "detections": detections,

            "lines": lines,

            "text": self._lines_to_text(
                lines
            ),

        }

    # ============================================================
    # GLOBAL MERGE
    # ============================================================

    def merge_detections(
        self,
        detections,
    ):
        """
        Merge detections from all tiles
        using global NMS.
        """

        if not detections:

            return []

        normalized = []

        for detection in detections:

            if not isinstance(
                detection,
                dict,
            ):

                continue

            if (
                "box"
                not in detection
            ):

                continue

            if (
                "class_id"
                not in detection
            ):

                continue

            if (
                "score"
                not in detection
            ):

                continue

            box = detection[
                "box"
            ]

            if (
                not isinstance(
                    box,
                    (
                        list,
                        tuple,
                    ),
                )
                or len(box) != 4
            ):

                continue

            try:

                x1 = float(
                    box[0]
                )

                y1 = float(
                    box[1]
                )

                x2 = float(
                    box[2]
                )

                y2 = float(
                    box[3]
                )

                score = float(
                    detection[
                        "score"
                    ]
                )

                class_id = int(
                    detection[
                        "class_id"
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if not np.isfinite(

                [
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                ]

            ).all():

                continue

            if (

                x2 <= x1
                or y2 <= y1

            ):

                continue

            if not (

                0
                <= class_id
                < len(
                    self.labels
                )

            ):

                continue

            item = dict(
                detection
            )

            item["class_id"] = (
                class_id
            )

            item["score"] = (
                score
            )

            item["letter"] = (

                self.labels[
                    class_id
                ]

            )

            item["box"] = [

                x1,
                y1,
                x2,
                y2,

            ]

            normalized.append(
                item
            )

        return self.nms(
            normalized
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

            key=lambda item: (

                (
                    float(
                        item[
                            "box"
                        ][1]
                    )

                    +

                    float(
                        item[
                            "box"
                        ][3]
                    )

                )

                / 2.0

            ),

        )

        lines = []

        for detection in vertical:

            x1, y1, x2, y2 = (

                float(
                    detection[
                        "box"
                    ][0]
                ),

                float(
                    detection[
                        "box"
                    ][1]
                ),

                float(
                    detection[
                        "box"
                    ][2]
                ),

                float(
                    detection[
                        "box"
                    ][3]
                ),

            )

            center_y = (

                y1 + y2

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

                tolerance = max(

                    height * 0.50,

                    line[
                        "avg_height"
                    ] * 0.60,

                    8.0,

                )

                distance = abs(

                    center_y
                    - line[
                        "center_y"
                    ]

                )

                if (

                    distance <= tolerance

                    and

                    distance
                    < best_distance

                ):

                    best_distance = (
                        distance
                    )

                    best_line = (
                        line
                    )

            if best_line is None:

                lines.append({

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

                })

            else:

                best_line[
                    "items"
                ].append(
                    detection
                )

                count = len(

                    best_line[
                        "items"
                    ]

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

        lines.sort(

            key=lambda line:
            line[
                "center_y"
            ]

        )

        for line in lines:

            line[
                "items"
            ].sort(

                key=lambda item:
                float(
                    item[
                        "box"
                    ][0]
                )

            )

        # Remove internal calculation fields.
        clean_lines = []

        for line in lines:

            clean_lines.append({

                "center_y": float(
                    line[
                        "center_y"
                    ]
                ),

                "items": list(
                    line[
                        "items"
                    ]
                ),

            })

        return clean_lines

    # ============================================================
    # LINES TO TEXT
    # ============================================================

    def _lines_to_text(
        self,
        lines,
    ):

        text_lines = []

        for line in lines:

            items = list(

                line.get(
                    "items",
                    [],
                )

            )

            if not items:

                continue

            items.sort(

                key=lambda item:
                float(
                    item[
                        "box"
                    ][0]
                )

            )

            characters = []

            for index, item in enumerate(
                items
            ):

                characters.append(

                    str(
                        item.get(
                            "letter",
                            "",
                        )
                    )

                )

                if index >= (
                    len(items) - 1
                ):

                    continue

                next_item = items[
                    index + 1
                ]

                current_x2 = float(

                    item[
                        "box"
                    ][2]

                )

                next_x1 = float(

                    next_item[
                        "box"
                    ][0]

                )

                gap = (

                    next_x1
                    - current_x2

                )

                current_width = max(

                    1.0,

                    float(
                        item[
                            "box"
                        ][2]
                    )

                    -

                    float(
                        item[
                            "box"
                        ][0]
                    ),

                )

                next_width = max(

                    1.0,

                    float(
                        next_item[
                            "box"
                        ][2]
                    )

                    -

                    float(
                        next_item[
                            "box"
                        ][0]
                    ),

                )

                average_width = (

                    current_width
                    + next_width

                ) / 2.0

                if (

                    gap
                    >

                    (
                        average_width
                        * self.space_threshold_factor
                    )

                ):

                    characters.append(
                        " "
                    )

            line_text = "".join(
                characters
            )

            if line_text.strip():

                text_lines.append(
                    line_text
                )

        return "\n".join(
            text_lines
        )

    # ============================================================
    # COMPOSE RESULT
    # ============================================================

    def compose_result(
        self,
        detections,
    ):
        """
        Build the final OCR result
        after global tile merging.
        """

        detections = list(
            detections
            or []
        )

        lines = self._group_lines(
            detections
        )

        text = self._lines_to_text(
            lines
        )

        return {

            "text": text,

            "detections": detections,

            "lines": lines,

        }

    # ============================================================
    # COMPATIBILITY METHOD
    # ============================================================

    def decode_text(
        self,
        output,
        meta,
    ):

        result = self.decode(

            output=output,

            meta=meta,

        )

        return result[
            "text"
        ]
