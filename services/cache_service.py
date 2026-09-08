# ============================================================
# GOTHIC OCR — CACHE SERVICE
# Persistent OCR Result Cache
# ============================================================

from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import threading


class OCRCache:
    """
    Persistent cache for OCR results.

    Cache key depends on:
    - absolute image path
    - file size
    - file modification time
    - cache version

    This prevents an old OCR result from being returned
    after the source image changes.
    """

    DEFAULT_VERSION = "1"
    DEFAULT_MAX_AGE = 30 * 24 * 60 * 60  # 30 days

    def __init__(
        self,
        cache_dir=None,
        version=None,
        max_age=None,
    ):
        self._lock = threading.RLock()

        self.version = str(
            version or self.DEFAULT_VERSION
        )

        self.max_age = (
            self.DEFAULT_MAX_AGE
            if max_age is None
            else int(max_age)
        )

        # --------------------------------------------------------
        # Cache directory
        # --------------------------------------------------------

        if cache_dir is None:
            cache_dir = (
                Path.home()
                / ".gothicocr"
                / "cache"
            )

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # KEY
    # ============================================================

    def make_key(self, image_path):
        """
        Create a stable cache key for an image.
        """

        path = Path(image_path)

        if not path.exists():
            return None

        try:
            resolved = str(
                path.resolve()
            )

            stat = path.stat()

            payload = (
                f"{self.version}|"
                f"{resolved}|"
                f"{stat.st_size}|"
                f"{stat.st_mtime_ns}"
            )

            return hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

        except OSError:
            return None

    # ============================================================
    # FILE
    # ============================================================

    def _cache_file(self, image_path):
        key = self.make_key(image_path)

        if not key:
            return None

        return self.cache_dir / f"{key}.json"

    # ============================================================
    # GET
    # ============================================================

    def get(self, image_path):
        """
        Return cached OCR result.

        Returns None when:
        - no cache exists
        - cache is expired
        - cache is corrupted
        - image changed
        """

        with self._lock:

            cache_file = self._cache_file(
                image_path
            )

            if cache_file is None:
                return None

            if not cache_file.exists():
                return None

            try:
                stat = cache_file.stat()

                age = (
                    time.time()
                    - stat.st_mtime
                )

                if (
                    self.max_age >= 0
                    and age > self.max_age
                ):
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

                    return None

                with cache_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    payload = json.load(f)

                # ------------------------------------------------
                # Validate payload
                # ------------------------------------------------

                if not isinstance(
                    payload,
                    dict,
                ):
                    return None

                if (
                    payload.get("version")
                    != self.version
                ):
                    return None

                key = self.make_key(
                    image_path
                )

                if (
                    payload.get("key")
                    != key
                ):
                    return None

                result = payload.get(
                    "result"
                )

                if not isinstance(
                    result,
                    dict,
                ):
                    return None

                return result

            except (
                OSError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ):
                # Corrupted cache should never
                # break the OCR application.
                try:
                    cache_file.unlink()
                except OSError:
                    pass

                return None

    # ============================================================
    # SET
    # ============================================================

    def set(self, image_path, result):
        """
        Save OCR result atomically.

        If writing fails, OCR itself is not affected.
        """

        with self._lock:

            cache_file = self._cache_file(
                image_path
            )

            if cache_file is None:
                return False

            key = self.make_key(
                image_path
            )

            if key is None:
                return False

            payload = {
                "version": self.version,
                "key": key,
                "created_at": time.time(),
                "result": result,
            }

            temporary_path = None

            try:

                cache_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ------------------------------------------------
                # Write temporary file
                # ------------------------------------------------

                fd, temporary_path = (
                    tempfile.mkstemp(
                        prefix=".gothicocr-",
                        suffix=".tmp",
                        dir=str(
                            self.cache_dir
                        ),
                    )
                )

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        payload,
                        f,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )

                    f.flush()
                    os.fsync(
                        f.fileno()
                    )

                # ------------------------------------------------
                # Atomic replacement
                # ------------------------------------------------

                os.replace(
                    temporary_path,
                    cache_file,
                )

                temporary_path = None

                return True

            except (
                OSError,
                TypeError,
                ValueError,
            ):
                return False

            finally:

                if (
                    temporary_path
                    and os.path.exists(
                        temporary_path
                    )
                ):
                    try:
                        os.unlink(
                            temporary_path
                        )
                    except OSError:
                        pass

    # ============================================================
    # EXISTS
    # ============================================================

    def exists(self, image_path):
        """
        Check whether a valid cache result exists.
        """

        return (
            self.get(image_path)
            is not None
        )

    # ============================================================
    # DELETE
    # ============================================================

    def delete(self, image_path):
        """
        Delete cached result for one image.
        """

        with self._lock:

            cache_file = self._cache_file(
                image_path
            )

            if (
                cache_file is None
                or not cache_file.exists()
            ):
                return False

            try:
                cache_file.unlink()
                return True
            except OSError:
                return False

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self):
        """
        Delete all cache entries.
        """

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            removed = 0

            for cache_file in self.cache_dir.glob(
                "*.json"
            ):
                try:
                    cache_file.unlink()
                    removed += 1
                except OSError:
                    pass

            return removed

    # ============================================================
    # CLEAN OLD
    # ============================================================

    def clean_old(self):
        """
        Remove expired cache entries.
        """

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            removed = 0
            now = time.time()

            for cache_file in self.cache_dir.glob(
                "*.json"
            ):
                try:

                    age = (
                        now
                        - cache_file.stat().st_mtime
                    )

                    if (
                        self.max_age >= 0
                        and age > self.max_age
                    ):
                        cache_file.unlink()
                        removed += 1

                except OSError:
                    pass

            return removed

    # ============================================================
    # SIZE
    # ============================================================

    def size_bytes(self):
        """
        Return total cache size.
        """

        with self._lock:

            if not self.cache_dir.exists():
                return 0

            total = 0

            for cache_file in self.cache_dir.glob(
                "*.json"
            ):
                try:
                    total += (
                        cache_file.stat().st_size
                    )
                except OSError:
                    pass

            return total

    # ============================================================
    # COUNT
    # ============================================================

    def count(self):
        """
        Return number of cache entries.
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
