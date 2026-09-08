# ============================================================
# GOTHIC OCR — IMAGE SERVICE
# ============================================================

from pathlib import Path

import numpy as np
from PIL import Image


class ImageService:
    """
    تجهيز الصور لنموذج GothicOCR.

    الوظائف:
        1. تجهيز صورة واحدة إلى:
           [1, 1024, 1024, 3]

        2. Letterbox مع الحفاظ على الأبعاد.

        3. تقسيم الصور الكبيرة إلى Tiles متداخلة
           Overlapping Tiles حتى لا تضيع الحروف الصغيرة.

        4. الحفاظ على Metadata اللازمة لإرجاع
           إحداثيات الحروف إلى الصورة الأصلية.
    """

    # ========================================================
    # INIT
    # ========================================================

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
    # VALIDATE RGB IMAGE
    # ========================================================

    @staticmethod
    def _validate_rgb(image):
        image = np.asarray(
            image,
            dtype=np.uint8,
        )

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

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            raise ValueError(
                "أبعاد الصورة غير صحيحة."
            )

        return image

    # ========================================================
    # PREPARE RGB IMAGE
    # ========================================================

    def _prepare_rgb(self, image):
        image = self._validate_rgb(image)

        original_h, original_w = image.shape[:2]

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

        # ====================================================
        # NEW DIMENSIONS
        # ====================================================

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

        # ====================================================
        # PADDING
        # ====================================================

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

        # ====================================================
        # NORMALIZE
        # ====================================================

        array /= 255.0

        # ====================================================
        # VALIDATE NORMALIZED VALUES
        # ====================================================

        if not np.all(
            np.isfinite(array)
        ):
            raise RuntimeError(
                "الصورة تحتوي على NaN أو Inf."
            )

        min_value = float(np.min(array))
        max_value = float(np.max(array))

        if min_value < 0.0 or max_value > 1.0:
            raise RuntimeError(
                "قيم الصورة بعد التطبيع "
                "خرجت عن النطاق 0..1."
            )

        # ====================================================
        # ADD BATCH DIMENSION
        # ====================================================

        prepared_input = np.expand_dims(
            array,
            axis=0,
        )

        # ====================================================
        # CONTIGUOUS FLOAT32
        # ====================================================

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
            "original_width": int(original_w),
            "original_height": int(original_h),

            "resized_width": int(new_w),
            "resized_height": int(new_h),

            "scale": float(scale),

            "pad_x": float(pad_x),
            "pad_y": float(pad_y),

            "input_size": int(
                self.target_size
            ),
        }

        return (
            prepared_input,
            meta,
        )

    # ========================================================
    # LOAD RGB IMAGE ONLY
    # ========================================================

    def load_rgb(
        self,
        image_path,
    ):
        """
        تحميل الصورة الأصلية وتحويلها إلى RGB
        بدون تصغيرها.

        هذا مهم جدًا للـTiling.
        """

        path = Path(
            image_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"الصورة غير موجودة: {path}"
            )

        try:
            with Image.open(path) as source:

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

        return self._validate_rgb(
            array
        )

    # ========================================================
    # GENERATE TILE START POSITIONS
    # ========================================================

    @staticmethod
    def _tile_positions(
        length,
        tile_size,
        stride,
    ):
        """
        إنشاء مواقع Tiles بحيث نضمن تغطية
        كامل الصورة، بما فيها الحافة الأخيرة.
        """

        length = int(length)
        tile_size = int(tile_size)
        stride = int(stride)

        if length <= tile_size:
            return [0]

        positions = []

        position = 0

        while True:

            positions.append(
                int(position)
            )

            if position + tile_size >= length:
                break

            next_position = (
                position + stride
            )

            # ------------------------------------------------
            # إذا كانت القطعة التالية ستتجاوز النهاية،
            # نحركها بحيث تنتهي بالضبط عند حافة الصورة.
            # ------------------------------------------------

            if next_position + tile_size >= length:
                final_position = (
                    length - tile_size
                )

                if (
                    not positions
                    or final_position != positions[-1]
                ):
                    positions.append(
                        int(final_position)
                    )

                break

            position = next_position

        return positions

    # ========================================================
    # PREPARE OVERLAPPING TILES
    # ========================================================

    def iter_tiles(
        self,
        image,
        overlap=0.15,
        tile_size=None,
    ):
        """
        تقسيم الصورة إلى Tiles متداخلة.

        Args:
            image:
                RGB numpy image [H, W, 3]

            overlap:
                نسبة التداخل بين القطع.
                الافتراضي 15%.

            tile_size:
                حجم القطعة الأصلية.
                الافتراضي = target_size.

        Yields:
            {
                "input": prepared_input,
                "meta": metadata,
                "offset_x": x,
                "offset_y": y,
                "tile_width": width,
                "tile_height": height,
            }

        ملاحظة:
            الإحداثيات offset_x / offset_y هي إحداثيات
            القطعة داخل الصورة الأصلية.
        """

        image = self._validate_rgb(
            image
        )

        height, width = image.shape[:2]

        if tile_size is None:
            tile_size = self.target_size

        tile_size = int(tile_size)

        if tile_size <= 0:
            raise ValueError(
                "tile_size يجب أن يكون أكبر من صفر."
            )

        if tile_size > self.target_size:
            raise ValueError(
                "tile_size لا يمكن أن يكون أكبر "
                "من target_size."
            )

        overlap = float(overlap)

        if not 0.0 <= overlap < 0.5:
            raise ValueError(
                "overlap يجب أن يكون بين 0.0 و0.5."
            )

        # ====================================================
        # SMALL IMAGE
        # ====================================================

        if (
            width <= tile_size
            and height <= tile_size
        ):

            prepared, meta = self._prepare_rgb(
                image
            )

            yield {
                "input": prepared,
                "meta": meta,
                "offset_x": 0,
                "offset_y": 0,
                "tile_width": int(width),
                "tile_height": int(height),
            }

            return

        # ====================================================
        # STRIDE
        # ====================================================

        stride = max(
            1,
            int(
                round(
                    tile_size
                    * (1.0 - overlap)
                )
            ),
        )

        x_positions = self._tile_positions(
            width,
            tile_size,
            stride,
        )

        y_positions = self._tile_positions(
            height,
            tile_size,
            stride,
        )

        # ====================================================
        # GENERATE TILES
        # ====================================================

        for offset_y in y_positions:

            for offset_x in x_positions:

                end_x = min(
                    offset_x + tile_size,
                    width,
                )

                end_y = min(
                    offset_y + tile_size,
                    height,
                )

                tile = image[
                    offset_y:end_y,
                    offset_x:end_x,
                    :,
                ]

                tile_height = (
                    end_y - offset_y
                )

                tile_width = (
                    end_x - offset_x
                )

                prepared, meta = (
                    self._prepare_rgb(
                        tile
                    )
                )

                # ------------------------------------------------
                # إضافة معلومات الموقع الأصلي للقطعة.
                # ------------------------------------------------

                meta = dict(meta)

                meta.update(
                    {
                        "offset_x": int(
                            offset_x
                        ),
                        "offset_y": int(
                            offset_y
                        ),
                        "tile_width": int(
                            tile_width
                        ),
                        "tile_height": int(
                            tile_height
                        ),
                        "source_width": int(
                            width
                        ),
                        "source_height": int(
                            height
                        ),
                    }
                )

                yield {
                    "input": prepared,
                    "meta": meta,
                    "offset_x": int(
                        offset_x
                    ),
                    "offset_y": int(
                        offset_y
                    ),
                    "tile_width": int(
                        tile_width
                    ),
                    "tile_height": int(
                        tile_height
                    ),
                }

    # ========================================================
    # LOAD IMAGE FROM PATH + PREPARE
    # ========================================================

    def load_and_prepare(
        self,
        image_path,
    ):
        """
        السلوك القديم:
        تحميل الصورة وتجهيزها مباشرة إلى 1024x1024.

        نحافظ عليه للتوافق مع الكود الحالي.
        """

        image = self.load_rgb(
            image_path
        )

        return self._prepare_rgb(
            image
        )

    # ========================================================
    # PREPARE NUMPY IMAGE
    # ========================================================

    def prepare(
        self,
        image,
    ):
        """
        تجهيز صورة RGB موجودة مسبقًا.
        """

        return self._prepare_rgb(
            image
        )

    # ========================================================
    # PREPARE IMAGE TILES FROM PATH
    # ========================================================

    def iter_tiles_from_path(
        self,
        image_path,
        overlap=0.15,
        tile_size=None,
    ):
        """
        تحميل الصورة الأصلية ثم إرجاع Tiles.

        Generator حتى لا يتم إنشاء كل القطع
        في الذاكرة دفعة واحدة.
        """

        image = self.load_rgb(
            image_path
        )

        yield from self.iter_tiles(
            image=image,
            overlap=overlap,
            tile_size=tile_size,
                )
