# ============================================================
# GOTHIC OCR — MODEL INSPECTOR
# ============================================================
#
# هذا الملف للفحص المحلي فقط.
# لا يحتاج Android ولا Buildozer.
#
# يجب تشغيله في بيئة تحتوي على:
#   - tflite-runtime
#   أو
#   - TensorFlow
#
# ============================================================

from pathlib import Path
import sys

import numpy as np


# ============================================================
# MODEL PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "models" / "gothic_ocr.tflite"


# ============================================================
# LOAD TFLITE INTERPRETER
# ============================================================

def get_interpreter():

    try:

        from tflite_runtime.interpreter import Interpreter

        print("Using: tflite-runtime")

        return Interpreter

    except ImportError:

        pass

    try:

        from tensorflow.lite import Interpreter

        print("Using: TensorFlow")

        return Interpreter

    except ImportError as exc:

        raise RuntimeError(
            "ثبتي tflite-runtime أو TensorFlow "
            "لفحص النموذج محليًا."
        ) from exc


# ============================================================
# CHECK MODEL
# ============================================================

def main():

    print("=" * 60)
    print("GOTHIC OCR — MODEL INSPECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    print()
    print("MODEL:")
    print(MODEL)

    if not MODEL.is_file():

        raise FileNotFoundError(
            f"ملف النموذج غير موجود:\n{MODEL}"
        )

    print(
        f"Size: {MODEL.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------------
    # Load interpreter
    # --------------------------------------------------------

    Interpreter = get_interpreter()

    interpreter = Interpreter(
        model_path=str(MODEL)
    )

    # --------------------------------------------------------
    # Allocate tensors
    # --------------------------------------------------------

    interpreter.allocate_tensors()

    # ========================================================
    # INPUT
    # ========================================================

    input_details = (
        interpreter.get_input_details()
    )

    print()
    print("=" * 60)
    print("INPUT TENSORS")
    print("=" * 60)

    for index, tensor in enumerate(
        input_details
    ):

        print()
        print(f"Input #{index}")
        print("  name:        ", tensor["name"])
        print("  shape:       ", tensor["shape"])
        print("  dtype:       ", tensor["dtype"])
        print("  quantization:", tensor["quantization"])
        print("  index:       ", tensor["index"])

    # ========================================================
    # OUTPUT
    # ========================================================

    output_details = (
        interpreter.get_output_details()
    )

    print()
    print("=" * 60)
    print("OUTPUT TENSORS")
    print("=" * 60)

    for index, tensor in enumerate(
        output_details
    ):

        print()
        print(f"Output #{index}")
        print("  name:        ", tensor["name"])
        print("  shape:       ", tensor["shape"])
        print("  dtype:       ", tensor["dtype"])
        print("  quantization:", tensor["quantization"])
        print("  index:       ", tensor["index"])

    # ========================================================
    # EXPECTED MODEL
    # ========================================================

    expected_input_shape = (
        1,
        1024,
        1024,
        3,
    )

    expected_output_shape = (
        1,
        29,
        21504,
    )

    print()
    print("=" * 60)
    print("EXPECTED MODEL")
    print("=" * 60)

    print(
        "Expected input :",
        expected_input_shape
    )

    print(
        "Expected output:",
        expected_output_shape
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    errors = []

    # --------------------------------------------------------
    # Validate number of tensors
    # --------------------------------------------------------

    if len(input_details) != 1:

        errors.append(
            f"Expected 1 input tensor, "
            f"found {len(input_details)}."
        )

    if len(output_details) != 1:

        errors.append(
            f"Expected 1 output tensor, "
            f"found {len(output_details)}."
        )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if len(input_details) == 1:

        actual_input_shape = tuple(
            int(x)
            for x in input_details[0]["shape"]
        )

        actual_input_dtype = (
            input_details[0]["dtype"]
        )

        if actual_input_shape != (
            expected_input_shape
        ):

            errors.append(
                "Input shape mismatch: "
                f"{actual_input_shape} != "
                f"{expected_input_shape}"
            )

        if actual_input_dtype != np.float32:

            errors.append(
                "Input dtype mismatch: "
                f"{actual_input_dtype} != "
                "float32"
            )

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    if len(output_details) == 1:

        actual_output_shape = tuple(
            int(x)
            for x in output_details[0]["shape"]
        )

        actual_output_dtype = (
            output_details[0]["dtype"]
        )

        if actual_output_shape != (
            expected_output_shape
        ):

            errors.append(
                "Output shape mismatch: "
                f"{actual_output_shape} != "
                f"{expected_output_shape}"
            )

        if actual_output_dtype != np.float32:

            errors.append(
                "Output dtype mismatch: "
                f"{actual_output_dtype} != "
                "float32"
            )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 60)

    if errors:

        print("MODEL CHECK: FAILED")

        print("=" * 60)

        for error in errors:

            print(
                "ERROR:",
                error
            )

        sys.exit(1)

    print("MODEL CHECK: PASSED")
    print("=" * 60)

    print()
    print(
        "النموذج متوافق مع الإعدادات الحالية:"
    )

    print(
        "Input : [1, 1024, 1024, 3] FLOAT32"
    )

    print(
        "Output: [1, 29, 21504] FLOAT32"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
