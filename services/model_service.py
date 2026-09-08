from pathlib import Path
import numpy as np

from services.image_service import ImageService
from services.text_decoder import TextDecoder


class GothicOCR:
    """YOLO/TFLite inference service for the Gothic OCR model."""

    INPUT_SHAPE = (1, 1024, 1024, 3)
    OUTPUT_SHAPE = (1, 29, 21504)  # 4 box values + 25 classes
    INPUT_SIZE = 1024

    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.input_bytes = int(np.prod(self.INPUT_SHAPE) * np.dtype(np.float32).itemsize)
        self.output_bytes = int(np.prod(self.OUTPUT_SHAPE) * np.dtype(np.float32).itemsize)
        self.image_service = ImageService(target_size=self.INPUT_SIZE)
        self.decoder = TextDecoder()
        self.interpreter = None
        self.bridge = None
        self.runtime = None

        # Android: use the official TensorFlow Lite Java runtime bundled by Gradle.
        try:
            from kivy.utils import platform
            is_android = platform == "android"
        except Exception:
            is_android = False

        if is_android:
            self._init_android_runtime()
        else:
            self._init_desktop_runtime()

    def _init_android_runtime(self):
        try:
            from jnius import autoclass
            Bridge = autoclass("org.gothicocr.TFLiteBridge")
            self.bridge = Bridge(str(self.model_path), self.output_bytes)
            self.runtime = "android-tflite-java"
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Android TensorFlow Lite runtime. "
                "Check android.gradle_dependencies and android.add_src. "
                f"Details: {exc}"
            ) from exc

    def _init_desktop_runtime(self):
        Interpreter = None
        try:
            from tflite_runtime.interpreter import Interpreter as RuntimeInterpreter
            Interpreter = RuntimeInterpreter
            self.runtime = "tflite-runtime"
        except ImportError:
            try:
                from tensorflow.lite import Interpreter as TensorFlowInterpreter
                Interpreter = TensorFlowInterpreter
                self.runtime = "tensorflow-lite"
            except ImportError as exc:
                raise RuntimeError(
                    "Desktop testing requires tflite-runtime or TensorFlow. "
                    "Android does not require either Python package."
                ) from exc

        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as exc:
            self.interpreter = None
            raise RuntimeError(f"Failed to load TFLite model: {exc}") from exc

        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise RuntimeError("The model must have exactly one input and one output tensor.")

        input_shape = tuple(int(v) for v in self.input_details[0]["shape"])
        output_shape = tuple(int(v) for v in self.output_details[0]["shape"])
        if input_shape != self.INPUT_SHAPE:
            raise ValueError(f"Unexpected input shape {input_shape}; expected {self.INPUT_SHAPE}.")
        if output_shape != self.OUTPUT_SHAPE:
            raise ValueError(f"Unexpected output shape {output_shape}; expected {self.OUTPUT_SHAPE}.")
        if np.dtype(self.input_details[0]["dtype"]) != np.dtype(np.float32):
            raise ValueError("Model input must be float32.")
        if np.dtype(self.output_details[0]["dtype"]) != np.dtype(np.float32):
            raise ValueError("Model output must be float32.")

    def _prepare_image(self, image):
        if isinstance(image, (str, Path)):
            prepared, meta = self.image_service.load_and_prepare(image)
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Image must be RGB [H,W,3], got {array.shape}.")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            prepared, meta = self.image_service.prepare(array)

        prepared = np.ascontiguousarray(prepared, dtype=np.float32)
        if prepared.shape != self.INPUT_SHAPE:
            raise ValueError(f"Prepared image shape {prepared.shape}; expected {self.INPUT_SHAPE}.")
        if prepared.nbytes != self.input_bytes:
            raise ValueError("Prepared image byte size is invalid.")
        if not np.isfinite(prepared).all():
            raise ValueError("Prepared image contains NaN or Inf.")
        return prepared, meta

    def _run_inference(self, prepared):
        if self.bridge is not None:
            try:
                raw = self.bridge.run(prepared.tobytes(order="C"))
                output = np.frombuffer(bytes(raw), dtype=np.float32).copy()
                output = output.reshape(self.OUTPUT_SHAPE)
            except Exception as exc:
                raise RuntimeError(f"Android TensorFlow Lite inference failed: {exc}") from exc
        else:
            try:
                input_index = self.input_details[0]["index"]
                output_index = self.output_details[0]["index"]
                self.interpreter.set_tensor(input_index, prepared)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(output_index)
            except Exception as exc:
                raise RuntimeError(f"TensorFlow Lite inference failed: {exc}") from exc
            output = np.asarray(output, dtype=np.float32)

        if output.shape != self.OUTPUT_SHAPE:
            raise ValueError(f"Unexpected output shape {output.shape}; expected {self.OUTPUT_SHAPE}.")
        if not np.isfinite(output).all():
            raise ValueError("Model output contains NaN or Inf.")
        return output

    def predict(self, image):
        prepared, meta = self._prepare_image(image)
        output = self._run_inference(prepared)
        return self.decoder.decode(
            output=output,
            original_width=int(meta["original_width"]),
            original_height=int(meta["original_height"]),
            scale=float(meta["scale"]),
            pad_x=float(meta["pad_x"]),
            pad_y=float(meta["pad_y"]),
        )

    def close(self):
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception:
                pass
            self.bridge = None
        self.interpreter = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
