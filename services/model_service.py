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
    INPUT_SIZE = 1024
    EXPECTED_INPUT_SHAPE = (1, 1024, 1024, 3)
    EXPECTED_OUTPUT_SHAPE = (1, 29, 21504)
    TILE_OVERLAP = 0.15
    CACHE_VERSION = "gothicocr-v2-tile1024-overlap15"

    def __init__(self, model_path=None, labels_path=None, cache_dir=None, cache_enabled=True):
        self._lock = threading.RLock()
        self._closed = False

        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path or (base_dir / "models" / "gothic_ocr.tflite"))
        self.labels_path = Path(labels_path or (base_dir / "data" / "labels.json"))

        if not self.model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.image_service = ImageService(target_size=1024, fill=114)
        self.decoder = TextDecoder(labels_path=self.labels_path)
        self.cache_enabled = bool(cache_enabled)
        self.cache = OCRCache(cache_dir=cache_dir, version=self.CACHE_VERSION) if self.cache_enabled else None

        self.interpreter = None
        self._android_interpreter = False
        self._load_interpreter()

    def _load_interpreter(self):
        try:
            from jnius import autoclass
            Interpreter = autoclass("org.tensorflow.lite.Interpreter")
            FileMapMode = autoclass("java.nio.channels.FileChannel$MapMode")
            RandomAccessFile = autoclass("java.io.RandomAccessFile")
        except Exception:
            Interpreter = None

        if Interpreter is not None:
            model_file = None
            try:
                model_file = RandomAccessFile(str(self.model_path), "r")
                channel = model_file.getChannel()
                mapped = channel.map(FileMapMode.READ_ONLY, 0, self.model_path.stat().st_size)
                self.interpreter = Interpreter(mapped)
                self._android_interpreter = True
                self._validate_android_interpreter()
                return
            except Exception as exc:
                if self.interpreter is not None:
                    try:
                        self.interpreter.close()
                    except Exception:
                        pass
                self.interpreter = None
                raise RuntimeError(f"Android TensorFlow Lite failed: {exc}") from exc
            finally:
                if model_file is not None:
                    try:
                        model_file.close()
                    except Exception:
                        pass

        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self._android_interpreter = False
            self._validate_desktop_interpreter()
        except Exception as exc:
            self.interpreter = None
            raise RuntimeError(f"Failed to load TFLite model: {exc}") from exc

    def _validate_android_interpreter(self):
        inp = self.interpreter.getInputTensor(0)
        out = self.interpreter.getOutputTensor(0)
        input_shape = tuple(int(x) for x in inp.shape())
        output_shape = tuple(int(x) for x in out.shape())
        if input_shape != self.EXPECTED_INPUT_SHAPE:
            raise RuntimeError(f"Unexpected model input shape: {input_shape}; expected {self.EXPECTED_INPUT_SHAPE}")
        if output_shape != self.EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(f"Unexpected model output shape: {output_shape}; expected {self.EXPECTED_OUTPUT_SHAPE}")
        if "FLOAT32" not in str(inp.dataType()):
            raise RuntimeError("Model input must be FLOAT32")
        if "FLOAT32" not in str(out.dataType()):
            raise RuntimeError("Model output must be FLOAT32")

    def _validate_desktop_interpreter(self):
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        input_shape = tuple(int(x) for x in inputs[0]["shape"])
        output_shape = tuple(int(x) for x in outputs[0]["shape"])
        if input_shape != self.EXPECTED_INPUT_SHAPE:
            raise RuntimeError(f"Unexpected model input shape: {input_shape}; expected {self.EXPECTED_INPUT_SHAPE}")
        if output_shape != self.EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(f"Unexpected model output shape: {output_shape}; expected {self.EXPECTED_OUTPUT_SHAPE}")
        if inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32:
            raise RuntimeError("Model input/output dtype must be float32")

    def _run_single_android(self, tensor):
        from jnius import autoclass
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tuple(tensor.shape) != self.EXPECTED_INPUT_SHAPE:
            raise RuntimeError(f"Invalid input tensor shape: {tensor.shape}")

        ByteBuffer = autoclass("java.nio.ByteBuffer")
        ByteOrder = autoclass("java.nio.ByteOrder")
        input_buffer = ByteBuffer.allocateDirect(int(tensor.nbytes))
        output_bytes = int(np.prod(self.EXPECTED_OUTPUT_SHAPE) * 4)
        output_buffer = ByteBuffer.allocateDirect(output_bytes)
        input_buffer.order(ByteOrder.nativeOrder())
        output_buffer.order(ByteOrder.nativeOrder())
        input_buffer.put(tensor.tobytes(order="C"))
        input_buffer.rewind()
        self.interpreter.run(input_buffer, output_buffer)
        output_buffer.rewind()
        raw = bytearray(output_bytes)
        output_buffer.get(raw)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(self.EXPECTED_OUTPUT_SHAPE)

    def _run_single_desktop(self, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        self.interpreter.set_tensor(inputs[0]["index"], tensor)
        self.interpreter.invoke()
        output = np.asarray(self.interpreter.get_tensor(outputs[0]["index"]), dtype=np.float32)
        if tuple(output.shape) != self.EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(f"Invalid model output shape: {output.shape}")
        return output

    def _run_single(self, tensor):
        return self._run_single_android(tensor) if self._android_interpreter else self._run_single_desktop(tensor)

    def _cache_get(self, image_path):
        if not self.cache_enabled or self.cache is None:
            return None
        try:
            cached = self.cache.get(image_path)
            if cached is None:
                return None
            result = copy.deepcopy(cached)
            result["cache_hit"] = True
            return result
        except Exception:
            return None

    def _cache_set(self, image_path, result):
        if not self.cache_enabled or self.cache is None:
            return
        try:
            value = copy.deepcopy(result)
            value.pop("cache_hit", None)
            self.cache.set(image_path, value)
        except Exception:
            pass

    def predict(self, image_path, progress_callback=None, use_cache=True):
        with self._lock:
            if self._closed:
                raise RuntimeError("GothicOCR service is closed.")

            image_path = Path(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            if use_cache:
                cached = self._cache_get(image_path)
                if cached is not None:
                    if progress_callback:
                        try:
                            progress_callback(1, 1)
                        except Exception:
                            pass
                    return cached

            image = self.image_service.load_rgb(image_path)
            original_height, original_width = image.shape[:2]
            tiles = list(self.image_service.iter_tiles(image, overlap=self.TILE_OVERLAP, tile_size=self.INPUT_SIZE))
            total_tiles = len(tiles)
            all_detections = []

            for index, tile in enumerate(tiles, start=1):
                tensor = np.asarray(tile["input"], dtype=np.float32)
                if tuple(tensor.shape) != self.EXPECTED_INPUT_SHAPE:
                    raise RuntimeError(f"ImageService returned {tensor.shape}; expected {self.EXPECTED_INPUT_SHAPE}")

                raw_output = self._run_single(tensor)
                meta = tile["meta"]
                decoded = self.decoder.decode(
                    raw_output,
                    original_width=int(meta["original_width"]),
                    original_height=int(meta["original_height"]),
                    scale=float(meta["scale"]),
                    pad_x=float(meta["pad_x"]),
                    pad_y=float(meta["pad_y"]),
                )

                ox = float(tile.get("offset_x", 0))
                oy = float(tile.get("offset_y", 0))
                for detection in decoded.get("detections", []):
                    item = copy.deepcopy(detection)
                    if item.get("box") is not None:
                        x1, y1, x2, y2 = map(float, item["box"])
                        item["box"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                    item["tile_index"] = index
                    all_detections.append(item)

                if progress_callback:
                    try:
                        progress_callback(index, total_tiles)
                    except Exception:
                        pass

            if all_detections:
                detections = self.decoder.merge_detections(all_detections)
                result = self.decoder.compose_result(detections)
            else:
                result = {"text": "", "detections": [], "lines": []}

            result.update({
                "image_width": int(original_width),
                "image_height": int(original_height),
                "tiles_processed": int(total_tiles),
                "cache_hit": False,
            })
            self._cache_set(image_path, result)
            return result

    def clear_cache(self):
        if self.cache is not None:
            try:
                self.cache.clear()
            except Exception:
                pass

    def cache_info(self):
        if self.cache is None:
            return {"enabled": False, "count": 0, "size_bytes": 0}
        try:
            return {"enabled": True, "count": self.cache.count(), "size_bytes": self.cache.size_bytes()}
        except Exception:
            return {"enabled": True, "count": 0, "size_bytes": 0}

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self.interpreter is not None:
                    close_method = getattr(self.interpreter, "close", None)
                    if callable(close_method):
                        try:
                            close_method()
                        except Exception:
                            pass
            finally:
                self.interpreter = None
                self._closed = True

