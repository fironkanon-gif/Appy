# ============================================================
# GOTHIC OCR — CACHE SERVICE
# ============================================================

from pathlib import Path
import hashlib
import json
import os
import tempfile
import threading
import time


class OCRCache:
    """
    نظام Cache محلي لنتائج Gothic OCR.

    الوظائف:
    1. إنشاء مفتاح فريد لكل صورة.
    2. حفظ نتيجة OCR كـ JSON.
    3. استرجاع النتيجة السابقة.
    4. التأكد أن الصورة لم تتغير.
    5. حذف Cache قديم.
    6. مسح الكاش بالكامل.
    """

    CACHE_VERSION = "1"

    DEFAULT_MAX_AGE = (
        30 * 24 * 60 * 60
    )

    def __init__(
        self,
        cache_dir=None,
        max_age=None,
    ):
        # ====================================================
        # CACHE DIRECTORY
        # ====================================================

        if cache_dir is None:
            cache_dir = (
                Path.home()
                / ".gothicocr"
                / "cache"
            )

        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_age = (
            self.DEFAULT_MAX_AGE
            if max_age is None
            else float(max_age)
        )

        self._lock = threading.Lock()

    # ========================================================
    # IMAGE FINGERPRINT
    # ========================================================

    def _image_fingerprint(
        self,
        image_path,
    ):
        """
        إنشاء بصمة للصورة.

        نعتمد على:
        - المسار
        - حجم الملف
        - وقت آخر تعديل

        وهذا أسرع بكثير من قراءة صورة ضخمة
        وحساب SHA-256 لكل محتواها.
        """

        path = Path(
            image_path
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        stat = path.stat()

        payload = (
            f"{path}|"
            f"{stat.st_size}|"
            f"{stat.st_mtime_ns}|"
            f"{self.CACHE_VERSION}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # CACHE KEY
    # ========================================================

    def make_key(
        self,
        image_path,
    ):
        """
        إنشاء Cache key.
        """

        return self._image_fingerprint(
            image_path
        )

    # ========================================================
    # CACHE FILE
    # ========================================================

    def _cache_file(
        self,
        key,
    ):
        return (
            self.cache_dir
            / f"{key}.json"
        )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        image_path,
    ):
        """
        استرجاع نتيجة محفوظة.

        يرجع:
            dict
        أو:
            None
        """

        with self._lock:

            try:
                key = self.make_key(
                    image_path
                )

            except (
                FileNotFoundError,
                OSError,
            ):
                return None

            cache_file = (
                self._cache_file(
                    key
                )
            )

            if not cache_file.exists():
                return None

            # -----------------------------------------------
            # Check age
            # -----------------------------------------------

            try:
                age = (
                    time.time()
                    - cache_file.stat().st_mtime
                )

                if (
                    self.max_age > 0
                    and age > self.max_age
                ):
                    cache_file.unlink(
                        missing_ok=True
                    )

                    return None

            except OSError:
                return None

            # -----------------------------------------------
            # Read JSON
            # -----------------------------------------------

            try:

                with cache_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    cached = json.load(
                        file
                    )

            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ):
                # Cache تالف → نحذفه
                try:
                    cache_file.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

                return None

            # -----------------------------------------------
            # Validate structure
            # -----------------------------------------------

            if not isinstance(
                cached,
                dict,
            ):
                return None

            if (
                cached.get(
                    "cache_version"
                )
                != self.CACHE_VERSION
            ):
                return None

            if (
                cached.get(
                    "key"
                )
                != key
            ):
                return None

            result = cached.get(
                "result"
            )

            if not isinstance(
                result,
                dict,
            ):
                return None

            return result

    # ========================================================
    # SET
    # ========================================================

    def set(
        self,
        image_path,
        result,
    ):
        """
        حفظ نتيجة OCR.

        الكتابة تتم بشكل Atomic:
        نكتب ملفًا مؤقتًا ثم نستبدله بالملف النهائي.
        """

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "OCR result must be a dict."
            )

        with self._lock:

            key = self.make_key(
                image_path
            )

            cache_file = (
                self._cache_file(
                    key
                )
            )

            payload = {
                "cache_version":
                    self.CACHE_VERSION,

                "key":
                    key,

                "created_at":
                    time.time(),

                "image_path":
                    str(
                        Path(
                            image_path
                        ).expanduser()
                        .resolve()
                    ),

                "result":
                    result,
            }

            # -----------------------------------------------
            # Temporary file
            # -----------------------------------------------

            temp_file = None

            try:

                fd, temp_path = (
                    tempfile.mkstemp(
                        prefix=".gothicocr_",
                        suffix=".tmp",
                        dir=str(
                            self.cache_dir
                        ),
                    )
                )

                temp_file = Path(
                    temp_path
                )

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        payload,
                        file,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )

                    file.flush()

                    try:
                        os.fsync(
                            file.fileno()
                        )
                    except OSError:
                        pass

                # -------------------------------------------
                # Atomic replace
                # -------------------------------------------

                os.replace(
                    str(temp_file),
                    str(cache_file),
                )

                temp_file = None

            finally:

                if (
                    temp_file is not None
                ):
                    try:
                        temp_file.unlink(
                            missing_ok=True
                        )
                    except OSError:
                        pass

            return key

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        image_path,
    ):
        """
        هل توجد نتيجة صالحة محفوظة؟
        """

        return (
            self.get(image_path)
            is not None
        )

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        image_path,
    ):
        """
        حذف Cache لصورة واحدة.
        """

        with self._lock:

            try:
                key = self.make_key(
                    image_path
                )
            except (
                FileNotFoundError,
                OSError,
            ):
                return False

            cache_file = (
                self._cache_file(
                    key
                )
            )

            if not cache_file.exists():
                return False

            try:
                cache_file.unlink()
                return True

            except OSError:
                return False

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):
        """
        حذف جميع نتائج Cache.
        """

        removed = 0

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            for file in (
                self.cache_dir.glob(
                    "*.json"
                )
            ):

                try:
                    file.unlink()
                    removed += 1

                except OSError:
                    pass

        return removed

    # ========================================================
    # CLEAN OLD
    # ========================================================

    def clean_old(
        self,
        max_age=None,
    ):
        """
        حذف ملفات Cache القديمة.
        """

        if max_age is None:
            max_age = self.max_age

        max_age = float(
            max_age
        )

        if max_age <= 0:
            return 0

        removed = 0

        now = time.time()

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            for file in (
                self.cache_dir.glob(
                    "*.json"
                )
            ):

                try:

                    age = (
                        now
                        - file.stat().st_mtime
                    )

                    if age > max_age:
                        file.unlink()
                        removed += 1

                except OSError:
                    pass

        return removed

    # ========================================================
    # SIZE
    # ========================================================

    def size_bytes(self):
        """
        حجم الكاش بالبايت.
        """

        total = 0

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            for file in (
                self.cache_dir.glob(
                    "*.json"
                )
            ):

                try:
                    total += (
                        file.stat().st_size
                    )
                except OSError:
                    pass

        return total

    # ========================================================
    # COUNT
    # ========================================================

    def count(self):
        """
        عدد نتائج Cache.
        """

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            return len(
                list(
                    self.cache_dir.glob(
                        "*.json"
                    )
                )
              )
