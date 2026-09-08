# ============================================================
# GOTHIC OCR — MODEL SERVICE
# TFLite + Android PyJNIus + Tiling + NMS + Cache
# ============================================================

from pathlib import Path
import copy
import threading

import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder
from services.cache_service import OCRCache


class GothicOCR:
    """
    Main OCR inference service.

    Features:
    - TensorFlow Lite
    - Android PyJNIus support
    - Desktop TensorFlow fallback
    - NHWC input: [1, 1024, 1024, 3]
    - Tiling for large images
    - Global NMS
    - Text reconstruction
    - Persistent OCR cache
    """

    INPUT_SIZE = 1024
    EXPECTED_INPUT_SHAPE = (1, 1024, 1024, 3)
    EXPECTED_OUTPUT_SHAPE = (1, 29, 21504)

    TILE_OVERLAP = 0.15

    # Changing this invalidates previous OCR cache entries.
    CACHE_VERSION = "gothicocr-v2-tile1024-overlap15"

    def __init__(
        self,
        model_path=None,
        labels_path=None,
        cache_dir=None,
        cache_enabled=True,
    ):
        self._lock = threading.RLock()
        self._closed = False

        base_dir = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = base_dir / "models" / "gothic_ocr.tflite"

        if labels_path is None:
            labels_path = base_dir / "data" / "labels.json"

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {self.model_path}"
            )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}"
            )

        # --------------------------------------------------------
        # Image + decoder services
        # --------------------------------------------------------

        self.image_service = ImageService()

        self.decoder = TextDecoder(
            labels_path=self.labels_path
        )

        # --------------------------------------------------------
        # Cache
        # --------------------------------------------------------

        self.cache_enabled = bool(cache_enabled)

        if self.cache_enabled:
            self.cache = OCRCache(
                cache_dir=cache_dir,
                version=self.CACHE_VERSION,
            )
        else:
            self.cache = None

        # --------------------------------------------------------
        # TFLite interpreter
        # --------------------------------------------------------

        self.interpreter = None
        self._android_interpreter = False

        self._load_interpreter()

    # ============================================================
    # INTERPRETER
    # ============================================================

    def _load_interpreter(self):
        """
        Load TensorFlow Lite interpreter.

        Android:
            org.tensorflow.lite.Interpreter

        Desktop:
            tensorflow.lite.Interpreter
        """

        # --------------------------------------------------------
        # Android / PyJNIus
        # --------------------------------------------------------

        try:
            from jnius import autoclass

            Interpreter = autoclass(
                "org.tensorflow.lite.Interpreter"
            )

            FileInputStream = autoclass(
                "java.io.FileInputStream"
            )

            FileChannelMapMode = autoclass(
                "java.nio.channels.FileChannel$MapMode"
            )

            RandomAccessFile = autoclass(
                "java.io.RandomAccessFile"
            )

            model_file = RandomAccessFile(
                str(self.model_path),
                "r",
            )

            channel = model_file.getChannel()

            mapped_buffer = channel.map(
                FileChannelMapMode.READ_ONLY,
                0,
                self.model_path.stat().st_size,
            )

            self.interpreter = Interpreter(
                mapped_buffer
            )

            self._android_interpreter = True

            model_file.close()

            self._validate_android_interpreter()

            return

        except Exception:
            self.interpreter = None
            self._android_interpreter = False

        # --------------------------------------------------------
        # Desktop fallback
        # --------------------------------------------------------

        try:
            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.model_path)
            )

            self.interpreter.allocate_tensors()

            self._validate_desktop_interpreter()

        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow Lite interpreter could not be loaded. "
                "On Android, make sure TensorFlow Lite and PyJNIus "
                "are available."
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TFLite model: {exc}"
            ) from exc

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_android_interpreter(self):
        """Validate Android TFLite tensors."""

        input_tensor = self.interpreter.getInputTensor(0)
        output_tensor = self.interpreter.getOutputTensor(0)

        input_shape = tuple(
            int(x)
            for x in input_tensor.shape()
        )

        output_shape = tuple(
            int(x)
            for x in output_tensor.shape()
        )

        if input_shape != self.EXPECTED_INPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected model input shape: "
                f"{input_shape}. "
                f"Expected {self.EXPECTED_INPUT_SHAPE}."
            )

        if output_shape != self.EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{output_shape}. "
                f"Expected {self.EXPECTED_OUTPUT_SHAPE}."
            )

    def _validate_desktop_interpreter(self):
        """Validate desktop TFLite tensors."""

        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()

        if not inputs:
            raise RuntimeError("Model has no input tensor.")

        if not outputs:
            raise RuntimeError("Model has no output tensor.")

        input_shape = tuple(
            int(x)
            for x in inputs[0]["shape"]
        )

        output_shape = tuple(
            int(x)
            for x in outputs[0]["shape"]
        )

        if input_shape != self.EXPECTED_INPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected model input shape: "
                f"{input_shape}. "
                f"Expected {self.EXPECTED_INPUT_SHAPE}."
            )

        if output_shape != self.EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected model output shape: "
                f"{output_shape}. "
                f"Expected {self.EXPECTED_OUTPUT_SHAPE}."
            )

    # ============================================================
    # SINGLE TENSOR INFERENCE
    # ============================================================

    def _run_single_android(self, tensor):
        """
        Run one 1024x1024 tensor on Android.
        """

        input_tensor = self.interpreter.getInputTensor(0)
        output_tensor = self.interpreter.getOutputTensor(0)

        input_dtype = str(
            input_tensor.dataType()
        )

        output_dtype = str(
            output_tensor.dataType()
        )

        if "FLOAT32" not in input_dtype:
            raise RuntimeError(
                f"Expected FLOAT32 input, got {input_dtype}"
            )

        if "FLOAT32" not in output_dtype:
            raise RuntimeError(
                f"Expected FLOAT32 output, got {output_dtype}"
            )

        input_array = np.asarray(
            tensor,
            dtype=np.float32,
            order="C",
        )

        output_array = np.empty(
            self.EXPECTED_OUTPUT_SHAPE,
            dtype=np.float32,
            order="C",
        )

        # --------------------------------------------------------
        # Direct ByteBuffer
        # --------------------------------------------------------

        from jnius import autoclass

        ByteBuffer = autoclass(
            "java.nio.ByteBuffer"
        )

        ByteOrder = autoclass(
            "java.nio.ByteOrder"
        )

        input_buffer = ByteBuffer.allocateDirect(
            input_array.nbytes
        )

        input_buffer.order(
            ByteOrder.nativeOrder()
        )

        input_buffer.put(
            input_array.tobytes()
        )

        input_buffer.rewind()

        output_buffer = ByteBuffer.allocateDirect(
            output_array.nbytes
        )

        output_buffer.order(
            ByteOrder.nativeOrder()
        )

        self.interpreter.run(
            input_buffer,
            output_buffer,
        )

        output_buffer.rewind()

        output_buffer.get(
            output_array
        )

        return output_array

    def _run_single_desktop(self, tensor):
        """
        Run one 1024x1024 tensor on desktop TensorFlow Lite.
        """

        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()

        input_index = inputs[0]["index"]
        output_index = outputs[0]["index"]

        input_array = np.asarray(
            tensor,
            dtype=np.float32,
        )

        self.interpreter.set_tensor(
            input_index,
            input_array,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            output_index
        )

        return np.asarray(
            output,
            dtype=np.float32,
        )

    def _run_single(self, tensor):
        """Run one model inference."""

        if self._android_interpreter:
            return self._run_single_android(tensor)

        return self._run_single_desktop(tensor)

    # ============================================================
    # CACHE HELPERS
    # ============================================================

    def _cache_get(self, image_path):
        """
        Safely read cached OCR result.

        Cache errors must never break OCR.
        """

        if not self.cache_enabled or self.cache is None:
            return None

        try:
            cached = self.cache.get(
                image_path
            )

            if cached is None:
                return None

            result = copy.deepcopy(cached)

            result["cache_hit"] = True

            return result

        except Exception:
            return None

    def _cache_set(self, image_path, result):
        """
        Safely save OCR result.

        Failure to write cache must never
        make a successful OCR request fail.
        """

        if not self.cache_enabled or self.cache is None:
            return

        try:
            cache_result = copy.deepcopy(result)

            cache_result.pop(
                "cache_hit",
                None,
            )

            self.cache.set(
                image_path,
                cache_result,
            )

        except Exception:
            pass

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        image_path,
        progress_callback=None,
        use_cache=True,
    ):
        """
        Analyze an image.

        Parameters
        ----------
        image_path:
            Path to image.

        progress_callback:
            Optional callback:
                callback(current, total)

        use_cache:
            Whether cached results may be used.

        Returns
        -------
        dict:
            {
                "text": str,
                "detections": list,
                "lines": list,
                "cache_hit": bool
            }
        """

        with self._lock:

            if self._closed:
                raise RuntimeError(
                    "GothicOCR service is closed."
                )

            image_path = Path(image_path)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            # ----------------------------------------------------
            # CACHE LOOKUP
            # ----------------------------------------------------

            if use_cache:
                cached = self._cache_get(
                    image_path
                )

                if cached is not None:

                    if progress_callback:
                        try:
                            progress_callback(1, 1)
                        except Exception:
                            pass

                    return cached

            # ----------------------------------------------------
            # Load image
            # ----------------------------------------------------

            image = self.image_service.load_rgb(
                image_path
            )

            original_height, original_width = (
                image.shape[:2]
            )

            # ----------------------------------------------------
            # Generate tiles
            # ----------------------------------------------------

            tiles = self.image_service.iter_tiles(
                image,
                overlap=self.TILE_OVERLAP,
                tile_size=self.INPUT_SIZE,
            )

            # Since iter_tiles is a generator, convert only
            # its metadata references into a list so we know
            # the progress total.
            tiles = list(tiles)

            total_tiles = len(tiles)

            if total_tiles == 0:
                result = {
                    "text": "",
                    "detections": [],
                    "lines": [],
                    "cache_hit": False,
                }

                self._cache_set(
                    image_path,
                    result,
                )

                return result

            all_detections = []

            # ----------------------------------------------------
            # TILE INFERENCE
            # ----------------------------------------------------

            for index, tile in enumerate(tiles, start=1):

                tensor = tile["input"]

                meta = tile["meta"]

                offset_x = float(
                    tile.get("offset_x", 0)
                )

                offset_y = float(
                    tile.get("offset_y", 0)
                )

                # ----------------------------------------------
                # Model
                # ----------------------------------------------

                raw_output = self._run_single(
                    tensor
                )

                # ----------------------------------------------
                # Decode
                # ----------------------------------------------

                decoded = self.decoder.decode(
                    raw_output,
                    meta=meta,
                )

                detections = decoded.get(
                    "detections",
                    [],
                )

                # ----------------------------------------------
                # Move boxes from tile coordinates
                # to original image coordinates.
                # ----------------------------------------------

                for detection in detections:

                    item = copy.deepcopy(
                        detection
                    )

                    box = item.get("box")

                    if box is not None:
                        x1, y1, x2, y2 = (
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        )

                        item["box"] = [
                            x1 + offset_x,
                            y1 + offset_y,
                            x2 + offset_x,
                            y2 + offset_y,
                        ]

                    item["tile_index"] = index

                    all_detections.append(
                        item
                    )

                # ----------------------------------------------
                # Progress
                # ----------------------------------------------

                if progress_callback:
                    try:
                        progress_callback(
                            index,
                            total_tiles,
                        )
                    except Exception:
                        pass

            # ----------------------------------------------------
            # GLOBAL NMS
            # ----------------------------------------------------

            merged_detections = (
                self.decoder.merge_detections(
                    all_detections
                )
            )

            # ----------------------------------------------------
            # Final text reconstruction
            # ----------------------------------------------------

            result = (
                self.decoder.compose_result(
                    merged_detections
                )
            )

            # Add useful metadata.
            result["image_width"] = int(
                original_width
            )

            result["image_height"] = int(
                original_height
            )

            result["tiles_processed"] = int(
                total_tiles
            )

            result["cache_hit"] = False

            # ----------------------------------------------------
            # SAVE CACHE
            # ----------------------------------------------------

            self._cache_set(
                image_path,
                result,
            )

            return result

    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================

    def clear_cache(self):
        """Delete all cached OCR results."""

        if self.cache is None:
            return

        try:
            self.cache.clear()
        except Exception:
            pass

    def clean_cache(self):
        """Remove expired cache entries."""

        if self.cache is None:
            return 0

        try:
            return self.cache.clean_old()
        except Exception:
            return 0

    def cache_info(self):
        """
        Return basic cache statistics.
        """

        if self.cache is None:
            return {
                "enabled": False,
                "count": 0,
                "size_bytes": 0,
            }

        try:
            return {
                "enabled": True,
                "count": self.cache.count(),
                "size_bytes": self.cache.size_bytes(),
            }

        except Exception:
            return {
                "enabled": True,
                "count": 0,
                "size_bytes": 0,
            }

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):
        """
        Release interpreter resources.
        """

        with self._lock:

            if self._closed:
                return

            try:
                if self.interpreter is not None:

                    # Desktop TensorFlow Lite
                    close_method = getattr(
                        self.interpreter,
                        "close",
                        None,
                    )

                    if callable(close_method):
                        try:
                            close_method()
                        except Exception:
                            pass

            finally:
                self.interpreter = None
                self._closed = True

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

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
