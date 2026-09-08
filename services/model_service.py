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
    1. تحميل النموذج.
    2. تجهيز الصورة.
    3. تقسيم الصور الكبيرة إلى Tiles.
    4. تشغيل TFLite على كل Tile.
    5. تحويل الإحداثيات إلى الصورة الأصلية.
    6. دمج النتائج وإزالة التكرارات.
    7. بناء النص النهائي.
    """

    # ========================================================
    # MODEL SPEC
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

    INPUT_DTYPE = np.float32
    OUTPUT_DTYPE = np.float32

    TILE_SIZE = 1024
    TILE_OVERLAP = 0.15

    # ========================================================
    # INITIALIZATION
    # ========================================================

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
                f"TFLite model not found: "
                f"{self.model_path}"
            )

        if labels_path is None:
            labels_path = (
                self.model_path.parent.parent
                / "data"
                / "labels.json"
            )

        self.labels_path = Path(
            labels_path
        )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: "
                f"{self.labels_path}"
            )

        self.image_service = (
            ImageService()
        )

        self.decoder = TextDecoder(
            labels_path=self.labels_path
        )

        self.interpreter = None

        self._input_details = None
        self._output_details = None

        self.input_shape = None
        self.output_shape = None

        self.input_bytes = 0
        self.output_bytes = 0

        self._ByteBuffer = None
        self._ByteOrder = None

        self._lock = threading.Lock()

        self._closed = False

    # ========================================================
    # LOAD TFLITE
    # ========================================================

    def _load_interpreter(self):
        """
        تحميل TensorFlow Lite.

        Android:
            PyJNIus + org.tensorflow.lite.Interpreter

        Desktop:
            tensorflow.lite.Interpreter
        """

        if self._closed:
            raise RuntimeError(
                "GothicOCR is already closed."
            )

        if self.interpreter is not None:
            return

        # ====================================================
        # ANDROID / PYJNIUS
        # ====================================================

        try:
            from jnius import autoclass

            File = autoclass(
                "java.io.File"
            )

            Interpreter = autoclass(
                "org.tensorflow.lite.Interpreter"
            )

            ByteBuffer = autoclass(
                "java.nio.ByteBuffer"
            )

            ByteOrder = autoclass(
                "java.nio.ByteOrder"
            )

            interpreter = Interpreter(
                File(
                    str(
                        self.model_path
                    )
                )

            )

            self.interpreter = (
                interpreter
            )

            self._ByteBuffer = (
                ByteBuffer
            )

            self._ByteOrder = (
                ByteOrder
            )

            self._read_model_shape_android()

            return

        except ImportError:
            pass

        except Exception as error:
            # إذا كان PyJNIus موجودًا ولكن
            # TensorFlow Lite Java غير متاح،
            # نحاول Desktop fallback.
            self.interpreter = None

            self._ByteBuffer = None
            self._ByteOrder = None

            android_error = error

        # ====================================================
        # DESKTOP FALLBACK
        # ====================================================

        try:
            import tensorflow as tf

        except ImportError as error:
            if "android_error" in locals():
                raise RuntimeError(
                    "TensorFlow Lite is not available. "
                    f"Android error: {android_error}"
                ) from error

            raise RuntimeError(
                "TensorFlow Lite runtime "
                "is not available."
            ) from error

        interpreter = (
            tf.lite.Interpreter(
                model_path=str(
                    self.model_path
                )
            )
        )

        interpreter.allocate_tensors()

        self.interpreter = (
            interpreter
        )

        self._input_details = (
            interpreter.get_input_details()
        )

        self._output_details = (
            interpreter.get_output_details()
        )

        self.input_shape = tuple(
            int(v)
            for v in
            self._input_details[0][
                "shape"
            ]
        )

        self.output_shape = tuple(
            int(v)
            for v in
            self._output_details[0][
                "shape"
            ]
        )

        self._validate_model_shape()

    # ========================================================
    # ANDROID MODEL SHAPE
    # ========================================================

    def _read_model_shape_android(self):
        """
        قراءة شكل Tensor من Android TFLite.
        """

        input_tensor = (
            self.interpreter
            .getInputTensor(0)
        )

        output_tensor = (
            self.interpreter
            .getOutputTensor(0)
        )

        self.input_shape = tuple(
            int(v)
            for v in
            input_tensor.shape()
        )

        self.output_shape = tuple(
            int(v)
            for v in
            output_tensor.shape()
        )

        self._validate_model_shape()

        self.input_bytes = (
            int(
                np.prod(
                    self.input_shape
                )
            )
            * np.dtype(
                self.INPUT_DTYPE
            ).itemsize
        )

        self.output_bytes = (
            int(
                np.prod(
                    self.output_shape
                )
            )
            * np.dtype(
                self.OUTPUT_DTYPE
            ).itemsize
        )

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    def _validate_model_shape(self):
        """
        التأكد أن النموذج هو نموذج GothicOCR المتوقع.
        """

        if tuple(
            self.input_shape
        ) != self.INPUT_SHAPE:

            raise ValueError(
                "Unexpected TFLite input shape: "
                f"{self.input_shape}; "
                f"expected {self.INPUT_SHAPE}"
            )

        if tuple(
            self.output_shape
        ) != self.OUTPUT_SHAPE:

            raise ValueError(
                "Unexpected TFLite output shape: "
                f"{self.output_shape}; "
                f"expected {self.OUTPUT_SHAPE}"
            )

        self.input_bytes = (
            int(
                np.prod(
                    self.INPUT_SHAPE
                )
            )
            * np.dtype(
                self.INPUT_DTYPE
            ).itemsize
        )

        self.output_bytes = (
            int(
                np.prod(
                    self.OUTPUT_SHAPE
                )
            )
            * np.dtype(
                self.OUTPUT_DTYPE
            ).itemsize
        )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _prepare_input(
        self,
        tile_input,
    ):
        """
        تجهيز Tile لتطابق TensorFlow Lite.
        """

        array = np.asarray(
            tile_input,
            dtype=self.INPUT_DTYPE,
        )

        if array.shape != (
            self.INPUT_SHAPE
        ):
            raise ValueError(
                "Invalid TFLite input shape: "
                f"{array.shape}; "
                f"expected {self.INPUT_SHAPE}"
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                "Input tensor contains "
                "NaN or infinite values."
            )

        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(
                array
            )

        return array

    # ========================================================
    # DIRECT BYTE BUFFER
    # ========================================================

    def _new_direct_buffer(
        self,
        size_bytes,
    ):
        """
        إنشاء Direct ByteBuffer.
        """

        if (
            self._ByteBuffer is None
            or self._ByteOrder is None
        ):
            raise RuntimeError(
                "Java ByteBuffer is not available."
            )

        buffer = (
            self._ByteBuffer
            .allocateDirect(
                int(size_bytes)
            )
        )

        buffer.order(
            self._ByteOrder
            .nativeOrder()
        )

        return buffer

    # ========================================================
    # ANDROID INFERENCE
    # ========================================================

    def _run_android(
        self,
        input_data,
    ):
        """
        تشغيل TFLite على Android
        باستخدام Direct ByteBuffers.
        """

        input_data = (
            self._prepare_input(
                input_data
            )
        )

        raw_input = (
            input_data.tobytes(
                order="C"
            )
        )

        if len(
            raw_input
        ) != self.input_bytes:

            raise ValueError(
                "Input byte size mismatch: "
                f"{len(raw_input)} != "
                f"{self.input_bytes}"
            )

        input_buffer = (
            self._new_direct_buffer(
                self.input_bytes
            )
        )

        output_buffer = (
            self._new_direct_buffer(
                self.output_bytes
            )
        )

        try:
            # كتابة الإدخال
            input_buffer.put(
                raw_input
            )

            input_buffer.rewind()

            # تشغيل النموذج
            self.interpreter.run(
                input_buffer,
                output_buffer,
            )

            # قراءة الناتج
            output_buffer.rewind()

            raw_output = bytearray(
                self.output_bytes
            )

            output_buffer.get(
                raw_output
            )

        finally:
            input_buffer = None
            output_buffer = None

        output = np.frombuffer(
            raw_output,
            dtype=self.OUTPUT_DTYPE,
        ).copy()

        output = output.reshape(
            self.OUTPUT_SHAPE
        )

        if not np.isfinite(
            output
        ).all():
            raise ValueError(
                "TFLite output contains "
                "NaN or infinite values."
            )

        return output

    # ========================================================
    # DESKTOP INFERENCE
    # ========================================================

    def _run_desktop(
        self,
        input_data,
    ):
        """
        تشغيل TFLite على Desktop.
        """

        input_data = (
            self._prepare_input(
                input_data
            )
        )

        input_index = (
            self._input_details[0][
                "index"
            ]
        )

        output_index = (
            self._output_details[0][
                "index"
            ]
        )

        self.interpreter.set_tensor(
            input_index,
            input_data,
        )

        self.interpreter.invoke()

        output = (
            self.interpreter
            .get_tensor(
                output_index
            )
        )

        output = np.asarray(
            output,
            dtype=self.OUTPUT_DTYPE,
        )

        if output.shape != (
            self.OUTPUT_SHAPE
        ):
            raise ValueError(
                "Unexpected model output: "
                f"{output.shape}; "
                f"expected {self.OUTPUT_SHAPE}"
            )

        if not np.isfinite(
            output
        ).all():
            raise ValueError(
                "TFLite output contains "
                "NaN or infinite values."
            )

        return output

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

        if self._ByteBuffer is not None:
            return self._run_android(
                tile_input
            )

        return self._run_desktop(
            tile_input
        )

    # ========================================================
    # MOVE DETECTIONS
    # ========================================================

    @staticmethod
    def _move_detections_to_source(
        detections,
        offset_x,
        offset_y,
    ):
        """
        نقل إحداثيات Detection من Tile
        إلى الصورة الأصلية.
        """

        moved = []

        for detection in detections:

            box = detection.get(
                "box"
            )

            if (
                not box
                or len(box) != 4
            ):
                continue

            x1, y1, x2, y2 = box

            updated = dict(
                detection
            )

            updated["box"] = (
                float(
                    x1 + offset_x
                ),
                float(
                    y1 + offset_y
                ),
                float(
                    x2 + offset_x
                ),
                float(
                    y2 + offset_y
                ),
            )

            moved.append(
                updated
            )

        return moved

    # ========================================================
    # TILE POSITIONS
    # ========================================================

    def _axis_positions(
        self,
        length,
    ):
        """
        حساب مواقع Tiles على محور واحد.
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

            current = (
                next_position
            )

        return positions

    def _count_tiles(
        self,
        width,
        height,
    ):
        return (
            len(
                self._axis_positions(
                    width
                )
            )
            *
            len(
                self._axis_positions(
                    height
                )
            )
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
        تحليل الصورة كاملة.

        الصور الصغيرة:
            Tile واحدة.

        الصور الكبيرة:
            عدة Tiles مع Overlap.

        progress_callback:
            callback(current_tile, total_tiles)
        """

        with self._lock:

            if self._closed:
                raise RuntimeError(
                    "GothicOCR is already closed."
                )

            # -----------------------------------------------
            # Load original image
            # -----------------------------------------------

            image = (
                self.image_service
                .load_rgb(
                    image_path
                )
            )

            original_height = (
                image.shape[0]
            )

            original_width = (
                image.shape[1]
            )

            # -----------------------------------------------
            # Tile count
            # -----------------------------------------------

            total_tiles = (
                self._count_tiles(
                    original_width,
                    original_height,
                )
            )

            all_detections = []

            tiles_processed = 0

            # -----------------------------------------------
            # Process one Tile at a time
            # -----------------------------------------------

            for tile in (
                self.image_service
                .iter_tiles(
                    image,
                    overlap=self.TILE_OVERLAP,
                    tile_size=self.TILE_SIZE,
                )
            ):

                tiles_processed += 1

                # -------------------------------------------
                # Inference
                # -------------------------------------------

                output = (
                    self._predict_tile(
                        tile["input"]
                    )
                )

                # -------------------------------------------
                # Decode
                # -------------------------------------------

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

                detections = (
                    decoded.get(
                        "detections",
                        [],
                    )
                )

                # -------------------------------------------
                # Convert Tile → Original image
                # -------------------------------------------

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

                # -------------------------------------------
                # Progress
                # -------------------------------------------

                if progress_callback:

                    try:
                        progress_callback(
                            tiles_processed,
                            total_tiles,
                        )

                    except Exception:
                        # UI progress must never
                        # interrupt OCR.
                        pass

            # -----------------------------------------------
            # Before global NMS
            # -----------------------------------------------

            detections_before_nms = len(
                all_detections
            )

            # -----------------------------------------------
            # Global merge
            # -----------------------------------------------

            final_detections = (
                self.decoder
                .merge_detections(
                    all_detections
                )
            )

            detections_after_nms = len(
                final_detections
            )

            # -----------------------------------------------
            # Final result
            # -----------------------------------------------

            result = (
                self.decoder
                .compose_result(
                    final_detections
                )
            )

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

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
    # CLEANUP
    # ========================================================

    def close(self):
        """
        تحرير موارد TFLite.
        """

        with self._lock:

            interpreter = getattr(
                self,
                "interpreter",
                None,
            )

            if interpreter is not None:

                try:
                    interpreter.close()

                except Exception:
                    pass

            self.interpreter = None

            self._input_details = None
            self._output_details = None

            self._ByteBuffer = None
            self._ByteOrder = None

            self.input_shape = None
            self.output_shape = None

            self._closed = True

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
