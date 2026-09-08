# ============================================================
# GOTHIC OCR — IMAGE SERVICE
# ============================================================

from pathlib import Path

import numpy as np
from PIL import Image


class ImageService:
    """
    تجهيز الصور لنموذج GothicOCR.

    الناتج:
        dtype: float32
        shape: [1, 1024, 1024, 3]
        range: 0.0 .. 1.0

    يستخدم Letterbox للحفاظ على أبعاد الصورة.
    """

    def __init__(
        self,
        target_size=1024,
        fill=114,
    ):

        self.target_size = int(target_size)
        self.fill = int(fill)

        if self.target_size <= 0:
            raise ValueError(
                "target_size يجب أن يكون أكبر من صفر."
            )

        if not 0 <= self.fill <= 255:
            raise ValueError(
                "fill يجب أن يكون بين 0 و255."
            )

    # ========================================================
    # PREPARE RGB IMAGE
    # ========================================================

    def _prepare_rgb(self, image):

        image = np.asarray(
            image,
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if image.ndim != 3:

            raise ValueError(
                "الصورة يجب أن تكون ثلاثية الأبعاد "
                "[H, W, C]. "
                f"Got: {image.shape}"
            )

        if image.shape[2] != 3:

            raise ValueError(
                "الصورة يجب أن تكون RGB بثلاث قنوات. "
                f"Got: {image.shape}"
            )

        original_h, original_w = image.shape[:2]

        if original_w <= 0 or original_h <= 0:

            raise ValueError(
                "أبعاد الصورة غير صحيحة."
            )

        # ====================================================
        # LETTERBOX SCALE
        # ====================================================

        scale = min(
            self.target_size / float(original_w),
            self.target_size / float(original_h),
        )

        if not np.isfinite(scale) or scale <= 0:

            raise ValueError(
                f"قيمة scale غير صحيحة: {scale}"
            )

        # ----------------------------------------------------
        # New dimensions
        # ----------------------------------------------------

        new_w = max(
            1,
            int(round(original_w * scale)),
        )

        new_h = max(
            1,
            int(round(original_h * scale)),
        )

        # ====================================================
        # RESIZE
        # ====================================================

        pil_image = Image.fromarray(
            image,
            "RGB",
        )

        resized = pil_image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

        # ====================================================
        # CREATE LETTERBOX CANVAS
        # ====================================================

        canvas = Image.new(
            "RGB",
            (
                self.target_size,
                self.target_size,
            ),
            (
                self.fill,
                self.fill,
                self.fill,
            ),
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_x = (
            self.target_size - new_w
        ) // 2

        pad_y = (
            self.target_size - new_h
        ) // 2

        canvas.paste(
            resized,
            (
                pad_x,
                pad_y,
            ),
        )

        # ====================================================
        # NUMPY
        # ====================================================

        array = np.asarray(
            canvas,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Normalize:
        #
        # uint8 0..255
        #        ↓
        # float32 0..1
        # ----------------------------------------------------

        array /= 255.0

        # ----------------------------------------------------
        # Validate normalized values
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(array)
        ):

            raise RuntimeError(
                "الصورة تحتوي على NaN أو Inf."
            )

        if np.min(array) < 0.0 or np.max(array) > 1.0:

            raise RuntimeError(
                "قيم الصورة بعد التطبيع "
                "خرجت عن النطاق 0..1."
            )

        # ====================================================
        # ====================================================
        # ADD BATCH DIMENSION (HWC)
        # ====================================================

        prepared_input = np.expand_dims(
            array,
            axis=0,
        )

        # ----------------------------------------------------
        # Ensure contiguous float32 memory
        # ----------------------------------------------------

        prepared_input = np.ascontiguousarray(
            prepared_input,
            dtype=np.float32,
        )

        # ====================================================
        # STRICT SHAPE VALIDATION
        # ====================================================

        expected_shape = (
            1,
            self.target_size,
            self.target_size,
            3,
        )

        if prepared_input.shape != expected_shape:

            raise RuntimeError(
                "فشل تجهيز الصورة. "
                f"الناتج: {prepared_input.shape}. "
                f"المتوقع: {expected_shape}."
            )

        # ====================================================
        # METADATA
        # ====================================================

        meta = {
            "original_width": int(
                original_w
            ),

            "original_height": int(
                original_h
            ),

            "resized_width": int(
                new_w
            ),

            "resized_height": int(
                new_h
            ),

            "scale": float(
                scale
            ),

            "pad_x": float(
                pad_x
            ),

            "pad_y": float(
                pad_y
            ),

            "input_size": int(
                self.target_size
            ),
        }

        return (
            prepared_input,
            meta,
        )

    # ========================================================
    # LOAD IMAGE FROM PATH
    # ========================================================

    def load_and_prepare(
        self,
        image_path,
    ):

        path = Path(
            image_path
        )

        if not path.is_file():

            raise FileNotFoundError(
                f"الصورة غير موجودة: {path}"
            )

        try:

            with Image.open(path) as source:

                # --------------------------------------------
                # Convert everything to RGB
                # --------------------------------------------

                image = source.convert(
                    "RGB"
                )

                array = np.asarray(
                    image,
                    dtype=np.uint8,
                )

        except Exception as exc:

            raise RuntimeError(
                f"تعذر فتح الصورة: {path}"
            ) from exc

        return self._prepare_rgb(
            array
        )

    # ========================================================
    # PREPARE NUMPY IMAGE
    # ========================================================

    def prepare(
        self,
        image,
    ):

        return self._prepare_rgb(
            image
        )
