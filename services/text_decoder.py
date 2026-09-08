# ============================================================
# GOTHIC OCR — TEXT DECODER (OPTIMIZED)
# ============================================================

from pathlib import Path
import json
import numpy as np


class TextDecoder:

    INPUT_SIZE = 1024
    NUM_CLASSES = 25

    CONFIDENCE_THRESHOLD = 0.05
    NMS_IOU_THRESHOLD = 0.45

    def __init__(
        self,
        labels_path=None,
        confidence_threshold=None,
        nms_iou_threshold=None,
        space_threshold_factor=0.45,  # نسبة تحديد المسافة بين الكلمات
    ):
        if labels_path is None:
            labels_path = (
                Path(__file__).resolve().parent.parent / "data" / "labels.json"
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

        self.space_threshold_factor = space_threshold_factor

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold يجب أن يكون بين 0 و1.")

        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold يجب أن يكون بين 0 و1.")

    def _load_labels(self):
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        try:
            with open(self.labels_path, "r", encoding="utf-8") as file:
                labels = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in labels file: {self.labels_path}"
            ) from exc

        if not isinstance(labels, list):
            raise ValueError("labels.json must contain a list.")

        if len(labels) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} labels, found {len(labels)}."
            )

        labels = [str(label) for label in labels]

        if any(not label for label in labels):
            raise ValueError("labels.json contains an empty label.")

        return labels

    @staticmethod
    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    def nms(self, detections):
        if not detections:
            return []

        class_groups = {}
        for detection in detections:
            class_id = detection["class_id"]
            class_groups.setdefault(class_id, []).append(detection)

        kept = []
        for group in class_groups.values():
            group = sorted(group, key=lambda d: d["score"], reverse=True)

            while group:
                best = group.pop(0)
                kept.append(best)

                remaining = []
                for other in group:
                    overlap = self.iou(best["box"], other["box"])
                    if overlap < self.nms_iou_threshold:
                        remaining.append(other)

                group = remaining

        kept.sort(
            key=lambda d: (
                (d["box"][1] + d["box"][3]) / 2.0,
                d["box"][0],
            )
        )

        return kept

    def _box_to_canvas(self, x, y, w, h):
        values = np.array([x, y, w, h], dtype=np.float32)

        if np.all(np.abs(values) <= 1.5):
            canvas_x = x * self.INPUT_SIZE
            canvas_y = y * self.INPUT_SIZE
            canvas_w = w * self.INPUT_SIZE
            canvas_h = h * self.INPUT_SIZE
        else:
            canvas_x, canvas_y, canvas_w, canvas_h = x, y, w, h

        return float(canvas_x), float(canvas_y), float(canvas_w), float(canvas_h)

    def decode(
        self,
        output,
        original_width,
        original_height,
        scale,
        pad_x,
        pad_y,
    ):
        original_width = int(original_width)
        original_height = int(original_height)
        scale = float(scale)
        pad_x, pad_y = float(pad_x), float(pad_y)

        if original_width <= 0 or original_height <= 0:
            raise ValueError("original_width and original_height must be > 0.")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid image scale: {scale}")

        if not np.isfinite([pad_x, pad_y]).all():
            raise ValueError("pad_x/pad_y contain invalid values.")

        output = np.asarray(output, dtype=np.float32)

        if not np.all(np.isfinite(output)):
            raise ValueError("Model output contains NaN or Inf.")

        if output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape {output.shape}")
            output = output[0]

        if output.ndim != 2:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        if output.shape[0] == (4 + self.NUM_CLASSES):
            predictions = output
        elif output.shape[1] == (4 + self.NUM_CLASSES):
            predictions = output.T
        else:
            raise ValueError(
                f"Expected one output dimension to contain {4 + self.NUM_CLASSES} channels. Got: {output.shape}"
            )

        boxes = predictions[:4]
        class_scores = predictions[4 : 4 + self.NUM_CLASSES]

        scores = np.max(class_scores, axis=0)
        class_ids = np.argmax(class_scores, axis=0)

        positions = np.where(
            np.isfinite(scores) & (scores >= self.confidence_threshold)
        )[0]

        detections = []

        for position in positions:
            x = float(boxes[0, position])
            y = float(boxes[1, position])
            w = float(boxes[2, position])
            h = float(boxes[3, position])
            score = float(scores[position])
            class_id = int(class_ids[position])

            if not np.isfinite([x, y, w, h, score]).all():
                continue

            if w <= 0.0 or h <= 0.0:
                continue

            if not (0 <= class_id < len(self.labels)):
                continue

            canvas_x, canvas_y, canvas_w, canvas_h = self._box_to_canvas(x, y, w, h)

            canvas_x1 = canvas_x - canvas_w / 2.0
            canvas_y1 = canvas_y - canvas_h / 2.0
            canvas_x2 = canvas_x + canvas_w / 2.0
            canvas_y2 = canvas_y + canvas_h / 2.0

            x1 = (canvas_x1 - pad_x) / scale
            y1 = (canvas_y1 - pad_y) / scale
            x2 = (canvas_x2 - pad_x) / scale
            y2 = (canvas_y2 - pad_y) / scale

            x1 = float(np.clip(x1, 0.0, float(original_width)))
            y1 = float(np.clip(y1, 0.0, float(original_height)))
            x2 = float(np.clip(x2, 0.0, float(original_width)))
            y2 = float(np.clip(y2, 0.0, float(original_height)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "letter": self.labels[class_id],
                "score": score,
                "box": (x1, y1, x2, y2),
            })

        detections = self.nms(detections)
        lines = self._group_lines(detections)

        # بناء النص مع إمكانية تقدير الفواصل بين الكلمات
        text_lines = []
        for line in lines:
            items = line["items"]
            items.sort(key=lambda d: d["box"][0])

            if not items:
                continue

            line_chars = []
            for idx, d in enumerate(items):
                line_chars.append(d["letter"])

                # تحقق مما إذا كان يجب وضع مسافة قبل الحرف التالي
                if idx < len(items) - 1:
                    next_d = items[idx + 1]
                    curr_x2 = d["box"][2]
                    next_x1 = next_d["box"][0]
                    gap = next_x1 - curr_x2

                    curr_w = d["box"][2] - d["box"][0]
                    next_w = next_d["box"][2] - next_d["box"][0]
                    avg_w = (curr_w + next_w) / 2.0

                    if gap > (avg_w * self.space_threshold_factor):
                        line_chars.append(" ")

            line_text = "".join(line_chars)
            if line_text.strip():
                text_lines.append(line_text)

        text = "\n".join(text_lines)

        return {
            "text": text,
            "detections": detections,
            "lines": lines,
        }

    def _group_lines(self, detections):
        if not detections:
            return []

        vertical = sorted(
            detections,
            key=lambda d: (d["box"][1] + d["box"][3]) / 2.0,
        )

        lines = []

        for detection in vertical:
            x1, y1, x2, y2 = detection["box"]
            center_y = (y1 + y2) / 2.0
            height = max(1.0, y2 - y1)

            best_line = None
            best_distance = float("inf")

            for line in lines:
                line_center_y = line["center_y"]
                average_height = line["avg_height"]

                tolerance = max(
                    height * 0.50,
                    average_height * 0.60,
                    8.0,
                )

                distance = abs(center_y - line_center_y)

                if distance <= tolerance and distance < best_distance:
                    best_distance = distance
                    best_line = line

            if best_line is not None:
                best_line["items"].append(detection)

                # تحديث التجمعات التراكمية بكفاءة
                count = len(best_line["items"])
                best_line["total_y_sum"] += center_y
                best_line["total_h_sum"] += height

                best_line["center_y"] = best_line["total_y_sum"] / count
                best_line["avg_height"] = best_line["total_h_sum"] / count
            else:
                lines.append({
                    "center_y": float(center_y),
                    "avg_height": float(height),
                    "total_y_sum": float(center_y),
                    "total_h_sum": float(height),
                    "items": [detection],
                })

        lines.sort(key=lambda line: line["center_y"])

        for line in lines:
            line["items"].sort(key=lambda d: d["box"][0])

        return lines

    def decode_text(
        self, output, original_width, original_height, scale, pad_x, pad_y
    ):
        result = self.decode(
            output=output,
            original_width=original_width,
            original_height=original_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        return result["text"]
