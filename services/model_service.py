# ============================================================
# GOTHIC OCR — MODEL SERVICE
# ============================================================

from pathlib import Path

import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder


class GothicOCR:
    """
    محرك GothicOCR.

    المسؤوليات:
        - تحميل نموذج TFLite.
        - تشغيل inference على Android أو Desktop.
        - تحليل الصور الصغيرة مباشرة.
        - تحليل الصور الكبيرة باستخدام Tiles + Overlap.
        - تحويل إحداثيات الـTiles إلى الصورة الأصلية.
        - دمج النتائج.
        - إزالة التكرارات باستخدام NMS.
        - بناء النص النهائي.
    """

    # ========================================================
    # MODEL SPECIFICATIONS
    # ========================================================

    INPUT_SHAPE = (
        1,
        1024,
        1024,
        3,
    )

    OUTPUT_SHAPE = (
        1,
        29,
        21504,
    )

    INPUT_SIZE = 1024

    # ========================================================
    # LARGE IMAGE SETTINGS
    # ========================================================

    TILE_OVERLAP = 0.15

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        model_path,
        tile_overlap=None,
        confidence_threshold=None,
        nms_iou_threshold=None,
    ):
        self.model_path = (
            Path(model_path)
            .resolve()
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: "
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # BYTE SIZES
        # ----------------------------------------------------

        self.input_bytes = int(
            np.prod(self.INPUT_SHAPE)
            * np.dtype(np.float32).itemsize
        )

        self.output_bytes = int(
            np.prod(self.OUTPUT_SHAPE)
            * np.dtype(np.float32).itemsize
        )

        # ----------------------------------------------------
        # TILE CONFIGURATION
        # ----------------------------------------------------

        if tile_overlap is None:
            tile_overlap = self.TILE_OVERLAP

        self.tile_overlap = float(
            tile_overlap
        )

        if not 0.0 <= self.tile_overlap < 0.5:
            raise ValueError(
                "tile_overlap يجب أن يكون "
                "بين 0.0 و0.5."
            )

        # ----------------------------------------------------
        # SERVICES
        # ----------------------------------------------------

        self.image_service = ImageService(
            target_size=self.INPUT_SIZE
        )

        self.decoder = TextDecoder(
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        self.interpreter = None
        self.bridge = None
        self.runtime = None

        # ----------------------------------------------------
        # ANDROID / DESKTOP
        # ----------------------------------------------------

        try:
            from kivy.utils import platform

            is_android = (
                platform == "android"
            )

        except Exception:
            is_android = False

        if is_android:
            self._init_android_runtime()
        else:
            self._init_desktop_runtime()

    # ========================================================
    # ANDROID TFLITE
    # ========================================================

    def _init_android_runtime(self):
        try:
            from jnius import autoclass

            Bridge = autoclass(
                "org.gothicocr.TFLiteBridge"
            )

            self.bridge = Bridge(
                str(self.model_path),
                self.output_bytes,
            )

            self.runtime = (
                "android-tflite-java"
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Android "
                "TensorFlow Lite runtime. "
                "Check android.gradle_dependencies "
                "and android.add_src. "
                f"Details: {exc}"
            ) from exc

    # ========================================================
    # DESKTOP TFLITE
    # ========================================================

    def _init_desktop_runtime(self):

        Interpreter = None

        # ----------------------------------------------------
        # Try tflite-runtime first
        # ----------------------------------------------------

        try:
            from tflite_runtime.interpreter import (
                Interpreter as RuntimeInterpreter
            )

            Interpreter = RuntimeInterpreter

            self.runtime = (
                "tflite-runtime"
            )

        except ImportError:

            # ------------------------------------------------
            # Fallback to TensorFlow
            # ------------------------------------------------

            try:
                from tensorflow.lite import (
                    Interpreter as TensorFlowInterpreter
                )

                Interpreter = TensorFlowInterpreter

                self.runtime = (
                    "tensorflow-lite"
                )

            except ImportError as exc:

                raise RuntimeError(
                    "Desktop testing requires "
                    "tflite-runtime or TensorFlow. "
                    "Android does not require either "
                    "Python package."
                ) from exc

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            self.interpreter = Interpreter(
                model_path=str(
                    self.model_path
                )
            )

            self.interpreter.allocate_tensors()

            self.input_details = (
                self.interpreter.get_input_details()
            )

            self.output_details = (
                self.interpreter.get_output_details()
            )

        except Exception as exc:

            self.interpreter = None

            raise RuntimeError(
                f"Failed to load TFLite model: {exc}"
            ) from exc

        # ----------------------------------------------------
        # TENSOR COUNT
        # ----------------------------------------------------

        if (
            len(self.input_details) != 1
            or len(self.output_details) != 1
        ):
            raise RuntimeError(
                "The model must have exactly "
                "one input and one output tensor."
            )

        # ----------------------------------------------------
        # INPUT SHAPE
        # ----------------------------------------------------

        input_shape = tuple(
            int(v)
            for v in self.input_details[0]["shape"]
        )

        output_shape = tuple(
            int(v)
            for v in self.output_details[0]["shape"]
        )

        if input_shape != self.INPUT_SHAPE:
            raise ValueError(
                f"Unexpected input shape "
                f"{input_shape}; "
                f"expected {self.INPUT_SHAPE}."
            )

        if output_shape != self.OUTPUT_SHAPE:
            raise ValueError(
                f"Unexpected output shape "
                f"{output_shape}; "
                f"expected {self.OUTPUT_SHAPE}."
            )

        # ----------------------------------------------------
        # DTYPE
        # ----------------------------------------------------

        if (
            np.dtype(
                self.input_details[0]["dtype"]
            )
            != np.dtype(np.float32)
        ):
            raise ValueError(
                "Model input must be float32."
            )

        if (
            np.dtype(
                self.output_details[0]["dtype"]
            )
            != np.dtype(np.float32)
        ):
            raise ValueError(
                "Model output must be float32."
            )

    # ========================================================
    # PREPARE SINGLE IMAGE
    # ========================================================

    def _prepare_image(
        self,
        image,
    ):

        if isinstance(
            image,
            (str, Path),
        ):

            prepared, meta = (
                self.image_service.load_and_prepare(
                    image
                )
            )

        else:

            array = np.asarray(
                image
            )

            if (
                array.ndim != 3
                or array.shape[2] != 3
            ):
                raise ValueError(
                    "Image must be RGB "
                    "[H,W,3], "
                    f"got {array.shape}."
                )

            if array.dtype != np.uint8:
                array = np.clip(
                    array,
                    0,
                    255,
                ).astype(
                    np.uint8
                )

            prepared, meta = (
                self.image_service.prepare(
                    array
                )
            )

        prepared = np.ascontiguousarray(
            prepared,
            dtype=np.float32,
        )

        self._validate_prepared(
            prepared
        )

        return (
            prepared,
            meta,
        )

    # ========================================================
    # VALIDATE PREPARED IMAGE
    # ========================================================

    def _validate_prepared(
        self,
        prepared,
    ):

        if prepared.shape != self.INPUT_SHAPE:
            raise ValueError(
                f"Prepared image shape "
                f"{prepared.shape}; "
                f"expected {self.INPUT_SHAPE}."
            )

        if prepared.nbytes != self.input_bytes:
            raise ValueError(
                "Prepared image byte size "
                "is invalid."
            )

        if not np.isfinite(
            prepared
        ).all():
            raise ValueError(
                "Prepared image contains "
                "NaN or Inf."
            )

    # ========================================================
    # RUN TFLITE INFERENCE
    # ========================================================

    def _run_inference(
        self,
        prepared,
    ):

        self._validate_prepared(
            prepared
        )

        # ====================================================
        # ANDROID
        # ====================================================

        if self.bridge is not None:

            try:

                raw = self.bridge.run(
                    prepared.tobytes(
                        order="C"
                    )
                )

                output = np.frombuffer(
                    bytes(raw),
                    dtype=np.float32,
                ).copy()

                output = output.reshape(
                    self.OUTPUT_SHAPE
                )

            except Exception as exc:

                raise RuntimeError(
                    "Android TensorFlow Lite "
                    f"inference failed: {exc}"
                ) from exc

        # ====================================================
        # DESKTOP
        # ====================================================

        else:

            try:

                input_index = (
                    self.input_details[0][
                        "index"
                    ]
                )

                output_index = (
                    self.output_details[0][
                        "index"
                    ]
                )

                self.interpreter.set_tensor(
                    input_index,
                    prepared,
                )

                self.interpreter.invoke()

                output = (
                    self.interpreter.get_tensor(
                        output_index
                    )
                )

            except Exception as exc:

                raise RuntimeError(
                    "TensorFlow Lite "
                    f"inference failed: {exc}"
                ) from exc

            output = np.asarray(
                output,
                dtype=np.float32,
            )

        # ====================================================
        # OUTPUT VALIDATION
        # ====================================================

        if output.shape != self.OUTPUT_SHAPE:
            raise ValueError(
                f"Unexpected output shape "
                f"{output.shape}; "
                f"expected {self.OUTPUT_SHAPE}."
            )

        if not np.isfinite(
            output
        ).all():
            raise ValueError(
                "Model output contains "
                "NaN or Inf."
            )

        return output

    # ========================================================
    # TRANSFORM TILE DETECTIONS
    # ========================================================

    @staticmethod
    def _move_detections_to_source(
        detections,
        offset_x,
        offset_y,
    ):
        """
        تحويل إحداثيات detections من Tile
        إلى الصورة الأصلية.
        """

        transformed = []

        offset_x = float(
            offset_x
        )

        offset_y = float(
            offset_y
        )

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["box"]
            )

            item = dict(
                detection
            )

            item["box"] = (
                float(x1 + offset_x),
                float(y1 + offset_y),
                float(x2 + offset_x),
                float(y2 + offset_y),
            )

            transformed.append(
                item
            )

        return transformed

    # ========================================================
    # BUILD TEXT FROM FINAL DETECTIONS
    # ========================================================

    def _build_text(
        self,
        detections,
    ):
        """
        إعادة بناء النص بعد دمج جميع الـTiles.
        """

        if not detections:
            return "", []

        lines = (
            self.decoder._group_lines(
                detections
            )
        )

        text_lines = []

        for line in lines:

            items = line["items"]

            items.sort(
                key=lambda d: d["box"][0]
            )

            if not items:
                continue

            line_chars = []

            for index, detection in enumerate(
                items
            ):

                line_chars.append(
                    detection["letter"]
                )

                if index < len(items) - 1:

                    next_detection = (
                        items[index + 1]
                    )

                    curr_x2 = (
                        detection["box"][2]
                    )

                    next_x1 = (
                        next_detection["box"][0]
                    )

                    gap = (
                        next_x1 - curr_x2
                    )

                    curr_w = (
                        detection["box"][2]
                        - detection["box"][0]
                    )

                    next_w = (
                        next_detection["box"][2]
                        - next_detection["box"][0]
                    )

                    average_width = (
                        curr_w + next_w
                    ) / 2.0

                    if gap > (
                        average_width
                        * self.decoder.space_threshold_factor
                    ):
                        line_chars.append(
                            " "
                        )

            line_text = "".join(
                line_chars
            )

            if line_text.strip():
                text_lines.append(
                    line_text
                )

        return (
            "\n".join(text_lines),
            lines,
        )

    # ========================================================
    # PREDICT — MAIN OCR ENTRY POINT
    # ========================================================

    def predict(
        self,
        image,
        progress_callback=None,
    ):
        """
        تحليل الصورة.

        الصور الصغيرة:
            inference واحد.

        الصور الكبيرة:
            Tiles + Overlap + Global NMS.

        progress_callback:
            دالة اختيارية تستقبل:
                progress_callback(current, total)
        """

        # ====================================================
        # LOAD ORIGINAL RGB IMAGE
        # ====================================================

        if isinstance(
            image,
            (str, Path),
        ):

            source_image = (
                self.image_service.load_rgb(
                    image
                )
            )

        else:

            source_image = np.asarray(
                image
            )

            if (
                source_image.ndim != 3
                or source_image.shape[2] != 3
            ):
                raise ValueError(
                    "Image must be RGB "
                    "[H,W,3], "
                    f"got {source_image.shape}."
                )

            if (
                source_image.dtype
                != np.uint8
            ):
                source_image = np.clip(
                    source_image,
                    0,
                    255,
                ).astype(
                    np.uint8
                )

        # ====================================================
        # SOURCE DIMENSIONS
        # ====================================================

        source_height, source_width = (
            source_image.shape[:2]
        )

        if (
            source_width <= 0
            or source_height <= 0
        ):
            raise ValueError(
                "Invalid source image dimensions."
            )

        # ====================================================
        # GENERATE TILES
        # ====================================================

        tiles = self.image_service.iter_tiles(
            image=source_image,
            overlap=self.tile_overlap,
            tile_size=self.INPUT_SIZE,
        )

        # ----------------------------------------------------
        # Generator لا يعطي total مباشرة.
        # نحسب القطع أولًا.
        # ----------------------------------------------------

        tile_list = list(
            tiles
        )

        total_tiles = len(
            tile_list
        )

        if total_tiles <= 0:
            raise RuntimeError(
                "No image tiles were generated."
            )

        # ====================================================
        # GLOBAL DETECTIONS
        # ====================================================

        all_detections = []

        # ====================================================
        # PROCESS EACH TILE
        # ====================================================

        for index, tile_data in enumerate(
            tile_list,
            start=1,
        ):

            prepared = tile_data[
                "input"
            ]

            meta = tile_data[
                "meta"
            ]

            offset_x = tile_data[
                "offset_x"
            ]

            offset_y = tile_data[
                "offset_y"
            ]

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            output = self._run_inference(
                prepared
            )

            # ------------------------------------------------
            # DECODE TILE
            # ------------------------------------------------

            result = self.decoder.decode(
                output=output,
                original_width=int(
                    meta["original_width"]
                ),
                original_height=int(
                    meta["original_height"]
                ),
                scale=float(
                    meta["scale"]
                ),
                pad_x=float(
                    meta["pad_x"]
                ),
                pad_y=float(
                    meta["pad_y"]
                ),
            )

            # ------------------------------------------------
            # MOVE BOXES TO SOURCE IMAGE
            # ------------------------------------------------

            tile_detections = (
                self._move_detections_to_source(
                    result["detections"],
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            )

            all_detections.extend(
                tile_detections
            )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if progress_callback is not None:

                try:
                    progress_callback(
                        index,
                        total_tiles,
                    )
                except Exception:
                    # Progress callback must never
                    # break OCR.
                    pass

        # ====================================================
        # GLOBAL NMS
        # ====================================================

        final_detections = (
            self.decoder.nms(
                all_detections
            )
        )

        # ====================================================
        # SORT
        # ====================================================

        final_detections.sort(
            key=lambda detection: (
                (
                    detection["box"][1]
                    + detection["box"][3]
                )
                / 2.0,
                detection["box"][0],
            )
        )

        # ====================================================
        # BUILD FINAL TEXT
        # ====================================================

        text, lines = (
            self._build_text(
                final_detections
            )
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {
            "text": text,

            "detections": final_detections,

            "lines": lines,

            "image_width": int(
                source_width
            ),

            "image_height": int(
                source_height
            ),

            "tiles_processed": int(
                total_tiles
            ),

            "detections_before_nms": int(
                len(all_detections)
            ),

            "detections_after_nms": int(
                len(final_detections)
            ),
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.bridge is not None:

            try:
                self.bridge.close()

            except Exception:
                pass

            self.bridge = None

        self.interpreter = None

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

        return False

    # ========================================================
    # DESTRUCTOR
    # ========================================================

    def __del__(self):

        try:
            self.close()

        except Exception:
            pass
