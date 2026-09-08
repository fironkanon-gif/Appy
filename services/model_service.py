# ============================================================
# GOTHIC OCR — MODEL SERVICE
# ============================================================

from pathlib import Path
import threading

import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder


class GothicOCR:
    """
    خدمة تشغيل نموذج Gothic OCR.

    المسؤوليات:
    1. تحميل الصورة الأصلية.
    2. تقسيم الصور الكبيرة إلى Tiles.
    3. تشغيل TFLite على كل Tile.
    4. تحويل الإحداثيات إلى الصورة الأصلية.
    5. دمج النتائج وإزالة التكرارات.
    6. بناء النص النهائي.
    """

    TILE_SIZE = 1024
    TILE_OVERLAP = 0.15

    def __init__(
        self,
        model_path,
        labels_path=None,
    ):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.image_service = ImageService()

        if labels_path is None:
            labels_path = (
                self.model_path.parent.parent
                / "data"
                / "labels.json"
            )

        self.decoder = TextDecoder(
            labels_path=labels_path
        )

        self._interpreter = None
        self._input_details = None
        self._output_details = None

        self._lock = threading.Lock()

        self._closed = False

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def _load_interpreter(self):
        """
        تحميل TensorFlow Lite Interpreter.

        على Android:
        يستخدم TFLiteBridge إذا كان متاحًا.

        على Desktop:
        يستخدم tensorflow.lite.Interpreter.
        """

        if self._closed:
            raise RuntimeError(
                "GothicOCR is already closed."
            )

        if self._interpreter is not None:
            return

        # ----------------------------------------------------
        # Android / Java Bridge
        # ----------------------------------------------------

        try:
            from jnius import autoclass

            TFLiteBridge = autoclass(
                "org.gothicocr.TFLiteBridge"
            )

            self._interpreter = TFLiteBridge(
                str(self.model_path)
            )

            return

        except Exception:
            # إذا لم يكن Android bridge متاحًا
            # ننتقل إلى TensorFlow Lite العادي.
            pass

        # ----------------------------------------------------
        # Desktop fallback
        # ----------------------------------------------------

        try:
            import tensorflow as tf

        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow Lite runtime is not available."
            ) from exc

        interpreter = (
            tf.lite.Interpreter(
                model_path=str(
                    self.model_path
                )
            )
        )

        interpreter.allocate_tensors()

        self._interpreter = interpreter

        self._input_details = (
            interpreter.get_input_details()
        )

        self._output_details = (
            interpreter.get_output_details()
        )

    # ========================================================
    # SINGLE TILE INFERENCE
    # ========================================================

    def _predict_tile(
        self,
        tile_input,
    ):
        """
        تشغيل النموذج على Tile واحدة.
        """

        self._load_interpreter()

        # ----------------------------------------------------
        # Android bridge
        # ----------------------------------------------------

        if self._is_android_bridge():

            return self._predict_android(
                tile_input
            )

        # ----------------------------------------------------
        # Desktop TFLite
        # ----------------------------------------------------

        interpreter = (
            self._interpreter
        )

        input_index = (
            self._input_details[0]["index"]
        )

        output_index = (
            self._output_details[0]["index"]
        )

        interpreter.set_tensor(
            input_index,
            tile_input,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        return np.asarray(
            output,
            dtype=np.float32,
        )

    # ========================================================
    # ANDROID
    # ========================================================

    def _is_android_bridge(self):
        """
        التحقق من أن الـinterpreter هو Java Bridge.
        """

        return not hasattr(
            self._interpreter,
            "set_tensor",
        )

    def _predict_android(
        self,
        tile_input,
    ):
        """
        تشغيل النموذج من خلال TFLiteBridge.

        نحول الإدخال إلى float32 contiguous
        ثم نرسله إلى Java.
        """

        array = np.asarray(
            tile_input,
            dtype=np.float32,
        )

        array = np.ascontiguousarray(
            array
        )

        # ----------------------------------------------------
        # محاولة الواجهات المحتملة للـBridge
        # ----------------------------------------------------

        if hasattr(
            self._interpreter,
            "run"
        ):
            output = (
                self._interpreter.run(
                    array
                )
            )

        elif hasattr(
            self._interpreter,
            "predict"
        ):
            output = (
                self._interpreter.predict(
                    array
                )
            )

        else:
            raise RuntimeError(
                "TFLiteBridge does not expose "
                "a supported inference method."
            )

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        return output

    # ========================================================
    # MOVE DETECTIONS TO SOURCE IMAGE
    # ========================================================

    @staticmethod
    def _move_detections_to_source(
        detections,
        offset_x,
        offset_y,
    ):
        """
        تحويل Box من إحداثيات Tile
        إلى إحداثيات الصورة الأصلية.
        """

        moved = []

        for detection in detections:

            box = detection.get(
                "box"
            )

            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = box

            updated = dict(
                detection
            )

            updated["box"] = (
                float(x1 + offset_x),
                float(y1 + offset_y),
                float(x2 + offset_x),
                float(y2 + offset_y),
            )

            moved.append(
                updated
            )

        return moved

    # ========================================================
    # TILE COUNT
    # ========================================================

    def _axis_positions(
        self,
        length,
    ):
        """
        حساب مواقع Tiles على محور واحد.

        نفس منطق ImageService:
        - يبدأ من 0.
        - يستخدم overlap.
        - يضمن تغطية الطرف الأخير.
        """

        tile = self.TILE_SIZE

        if length <= tile:
            return [0]

        stride = max(
            1,
            int(
                round(
                    tile
                    * (
                        1.0
                        - self.TILE_OVERLAP
                    )
                )
            ),
        )

        positions = [0]

        current = 0

        while True:

            next_position = (
                current + stride
            )

            if (
                next_position + tile
                >= length
            ):

                final_position = (
                    length - tile
                )

                if (
                    final_position
                    != positions[-1]
                ):
                    positions.append(
                        final_position
                    )

                break

            positions.append(
                next_position
            )

            current = next_position

        return positions

    def _count_tiles(
        self,
        width,
        height,
    ):
        x_count = len(
            self._axis_positions(
                width
            )
        )

        y_count = len(
            self._axis_positions(
                height
            )
        )

        return (
            x_count
            * y_count
        )

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        image_path,
        progress_callback=None,
    ):
        """
        تحليل صورة كاملة.

        الصور الصغيرة:
            Tile واحدة.

        الصور الكبيرة:
            عدة Tiles مع overlap.

        progress_callback:
            callback(current_tile, total_tiles)
        """

        with self._lock:

            if self._closed:
                raise RuntimeError(
                    "GothicOCR is already closed."
                )

            image = (
                self.image_service.load_rgb(
                    image_path
                )
            )

            original_height, original_width = (
                image.shape[:2]
            )

            total_tiles = (
                self._count_tiles(
                    original_width,
                    original_height,
                )
            )

            all_detections = []

            tiles_processed = 0

            # =================================================
            # IMPORTANT:
            # لا نستخدم list(iter_tiles(...))
            #
            # لأن ذلك يحتفظ بكل الـTiles في الذاكرة
            # في نفس الوقت.
            # =================================================

            for tile in (
                self.image_service.iter_tiles(
                    image,
                    overlap=self.TILE_OVERLAP,
                    tile_size=self.TILE_SIZE,
                )
            ):

                tiles_processed += 1

                # ---------------------------------------------
                # Inference
                # ---------------------------------------------

                output = (
                    self._predict_tile(
                        tile["input"]
                    )
                )

                # ---------------------------------------------
                # Decode Tile
                # ---------------------------------------------

                meta = tile["meta"]

                decoded = (
                    self.decoder.decode(
                        output=output,
                        original_width=meta[
                            "original_width"
                        ],
                        original_height=meta[
                            "original_height"
                        ],
                        scale=meta[
                            "scale"
                        ],
                        pad_x=meta[
                            "pad_x"
                        ],
                        pad_y=meta[
                            "pad_y"
                        ],
                    )
                )

                detections = decoded.get(
                    "detections",
                    [],
                )

                # ---------------------------------------------
                # Move Tile coordinates
                # → Original image coordinates
                # ---------------------------------------------

                moved = (
                    self._move_detections_to_source(
                        detections,
                        tile["offset_x"],
                        tile["offset_y"],
                    )
                )

                all_detections.extend(
                    moved
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                if progress_callback:

                    try:
                        progress_callback(
                            tiles_processed,
                            total_tiles,
                        )

                    except Exception:
                        # Progress UI must never
                        # break OCR itself.
                        pass

            detections_before_nms = len(
                all_detections
            )

            # =================================================
            # GLOBAL MERGE
            # =================================================

            final_detections = (
                self.decoder.merge_detections(
                    all_detections
                )
            )

            detections_after_nms = len(
                final_detections
            )

            # =================================================
            # FINAL TEXT
            # =================================================

            result = (
                self.decoder.compose_result(
                    final_detections
                )
            )

            # =================================================
            # EXTRA INFORMATION
            # =================================================

            result.update(
                {
                    "image_width": int(
                        original_width
                    ),
                    "image_height": int(
                        original_height
                    ),
                    "tiles_processed": int(
                        tiles_processed
                    ),
                    "tiles_expected": int(
                        total_tiles
                    ),
                    "detections_before_nms": int(
                        detections_before_nms
                    ),
                    "detections_after_nms": int(
                        detections_after_nms
                    ),
                }
            )

            return result

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        تحرير الموارد.
        """

        with self._lock:

            self._closed = True

            self._interpreter = None
            self._input_details = None
            self._output_details = None

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    # ========================================================
    # DESTRUCTOR
    # ========================================================

    def __del__(self):

        try:
            self.close()

        except Exception:
            pass
