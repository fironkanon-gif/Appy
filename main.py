# ============================================================
# GOTHIC OCR — PREMIUM ANDROID APPLICATION
# Gallery + Camera + OCR + Cache + Save + Copy
# ============================================================

from pathlib import Path
import os
import shutil
import threading

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.clipboard import Clipboard
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.utils import platform

# ------------------------------------------------------------
# OPTIONAL ANDROID IMPORTS
# ------------------------------------------------------------
try:
    from android import activity
    from android.permissions import Permission, check_permission, request_permissions
    from android.runnable import run_on_ui_thread
    from jnius import autoclass
    ANDROID_AVAILABLE = platform == "android"
except Exception:
    ANDROID_AVAILABLE = False

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
GOTHIC_FONT_PATH = FONT_DIR / "NotoSansGothic-Regular.ttf"
MODEL_PATH = ROOT / "models" / "gothic_ocr.tflite"

# Android activity request codes
GALLERY_REQUEST = 1001
CAMERA_REQUEST = 1002
SAVE_REQUEST = 1003

# ------------------------------------------------------------
# COLORS
# ------------------------------------------------------------
BACKGROUND = (0.035, 0.025, 0.075, 1)
CARD = (0.10, 0.07, 0.18, 1)
CARD_LIGHT = (0.15, 0.10, 0.25, 1)
PURPLE = (0.55, 0.30, 0.90, 1)
PURPLE_DARK = (0.30, 0.12, 0.55, 1)
GOLD = (0.95, 0.70, 0.25, 1)
TEXT = (0.96, 0.93, 1, 1)
TEXT_DIM = (0.72, 0.65, 0.82, 1)
DARK_TEXT = (0.10, 0.05, 0.18, 1)


def register_fonts():
    if GOTHIC_FONT_PATH.is_file():
        try:
            LabelBase.register(
                name="GothicOCR",
                fn_regular=str(GOTHIC_FONT_PATH),
            )
            return True
        except Exception as exc:
            print("GOTHIC FONT ERROR:", repr(exc))
    return False


GOTHIC_AVAILABLE = register_fonts()


# ============================================================
# PREMIUM WIDGETS
# ============================================================
class PremiumBackground(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*BACKGROUND)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update, size=self._update)

    def _update(self, *_):
        self.rect.pos = self.pos
        self.rect.size = self.size


class PremiumCard(BoxLayout):
    background_color = ListProperty(CARD)

    def __init__(self, radius=22, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        with self.canvas.before:
            Color(*self.background_color)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(radius)],
            )
        self.bind(pos=self._update, size=self._update, background_color=self._update_color)

    def _update(self, *_):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _update_color(self, *_):
        self.rect.rgba = self.background_color


class PremiumButton(Button):
    button_color = ListProperty(PURPLE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = TEXT
        self.font_name = "Roboto"
        self.font_size = "16sp"
        self.size_hint_y = None
        self.height = dp(56)
        with self.canvas.before:
            Color(*self.button_color)
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(18)],
            )
        self.bind(pos=self._update, size=self._update, button_color=self._update_color)

    def _update(self, *_):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _update_color(self, *_):
        self.rect.rgba = self.button_color


class StatCard(PremiumCard):
    def __init__(self, title, value="0", **kwargs):
        super().__init__(orientation="vertical", padding=dp(8), spacing=dp(2), **kwargs)
        self.title_label = Label(
            text=title,
            font_name="Roboto",
            font_size="12sp",
            color=TEXT_DIM,
            halign="center",
            valign="middle",
        )
        self.value_label = Label(
            text=value,
            font_name="Roboto",
            font_size="20sp",
            color=GOLD,
            halign="center",
            valign="middle",
        )
        self.add_widget(self.title_label)
        self.add_widget(self.value_label)


# ============================================================
# MAIN APP
# ============================================================
class GothicOCRApp(App):
    def build(self):
        self.title = "Gothic OCR"
        Window.clearcolor = BACKGROUND

        self.selected_image = None
        self.ocr = None
        self.last_result = None
        self._gallery_bound = False
        self._camera_bound = False
        self._save_bound = False
        self._camera_permission_pending = False

        self.ui_font = "Roboto"
        self.result_font = "GothicOCR" if GOTHIC_AVAILABLE else "Roboto"

        root = FloatLayout()
        root.add_widget(PremiumBackground())

        self.scroll = ScrollView(
            size_hint=(1, 1),
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(4),
        )
        self.root_box = BoxLayout(
            orientation="vertical",
            padding=[dp(18), dp(18), dp(18), dp(28)],
            spacing=dp(12),
            size_hint_y=None,
        )
        self.root_box.bind(minimum_height=self.root_box.setter("height"))
        self.scroll.add_widget(self.root_box)
        root.add_widget(self.scroll)

        # Header
        self.title_label = Label(
            text="✦ GOTHIC OCR ✦",
            font_name=self.ui_font,
            font_size="28sp",
            color=GOLD,
            size_hint_y=None,
            height=dp(52),
            halign="center",
            valign="middle",
        )
        self.subtitle = Label(
            text="Detect and analyze Gothic text using Artificial Intelligence",
            font_name=self.ui_font,
            font_size="15sp",
            color=TEXT_DIM,
            size_hint_y=None,
            height=dp(42),
            halign="center",
            valign="middle",
        )
        self._bind_text(self.subtitle)

        # Status
        self.status = Label(
            text="Ready to select an image",
            font_name=self.ui_font,
            font_size="16sp",
            color=TEXT,
            halign="center",
            valign="middle",
        )
        self._bind_text(self.status)
        self.status_card = PremiumCard(
            size_hint_y=None,
            height=dp(64),
            padding=dp(8),
        )
        self.status_card.add_widget(self.status)

        # Preview
        self.preview_card = PremiumCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(290),
            padding=dp(10),
            background_color=CARD,
        )
        self.preview = Image(allow_stretch=True, keep_ratio=True)
        self.preview_card.add_widget(self.preview)

        # Result
        self.result_title = Label(
            text="✦ EXTRACTED TEXT ✦",
            font_name=self.ui_font,
            font_size="16sp",
            color=GOLD,
            size_hint_y=None,
            height=dp(34),
            halign="center",
            valign="middle",
        )
        self.result = Label(
            text="No image has been analyzed yet",
            font_name=self.result_font,
            font_size="24sp",
            color=GOLD,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=dp(105),
        )
        self._bind_text(self.result)
        self.result_card = PremiumCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(155),
            padding=dp(10),
            spacing=dp(4),
            background_color=CARD_LIGHT,
        )
        self.result_card.add_widget(self.result_title)
        self.result_card.add_widget(self.result)

        # Statistics
        self.stats_layout = BoxLayout(size_hint_y=None, height=dp(82), spacing=dp(8))
        self.char_stat = StatCard("Characters", "0")
        self.conf_stat = StatCard("Confidence", "0%")
        self.line_stat = StatCard("Lines", "0")
        self.cache_stat = StatCard("Cache", "—")
        for stat in (self.char_stat, self.conf_stat, self.line_stat, self.cache_stat):
            self.stats_layout.add_widget(stat)

        # Buttons
        self.gallery_button = PremiumButton(
            text="🖼  Choose Image from Gallery",
            button_color=PURPLE,
        )
        self.gallery_button.bind(on_release=self.choose_gallery)

        self.camera_button = PremiumButton(
            text="📷  Capture Image with Camera",
            button_color=PURPLE_DARK,
        )
        self.camera_button.bind(on_release=self.capture_camera)

        self.analyze_button = PremiumButton(
            text="✦  ANALYZE IMAGE NOW  ✦",
            button_color=GOLD,
        )
        self.analyze_button.color = DARK_TEXT
        self.analyze_button.bind(on_release=self.analyze)

        self.action_row = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(8))
        self.copy_button = PremiumButton(text="Copy Text", button_color=PURPLE)
        self.save_button = PremiumButton(text="Save TXT", button_color=PURPLE_DARK)
        self.copy_button.bind(on_release=self.copy_text)
        self.save_button.bind(on_release=self.save_text)
        self.action_row.add_widget(self.copy_button)
        self.action_row.add_widget(self.save_button)

        self.another_button = PremiumButton(
            text="↻  Analyze Another Image",
            button_color=PURPLE_DARK,
        )
        self.another_button.bind(on_release=self.reset_image)

        self.footer = Label(
            text="✦ Gothic OCR • AI Powered • TFLite ✦",
            font_name=self.ui_font,
            font_size="13sp",
            color=TEXT_DIM,
            size_hint_y=None,
            height=dp(34),
            halign="center",
            valign="middle",
        )

        self._show_main_layout()
        return root

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _bind_text(self, widget):
        widget.bind(size=lambda inst, size: setattr(inst, "text_size", (max(dp(20), size[0] - dp(15)), None)))

    def _set_status(self, text):
        self.status.text = str(text)

    def _reset_stats(self):
        self.char_stat.value_label.text = "0"
        self.conf_stat.value_label.text = "0%"
        self.line_stat.value_label.text = "0"
        self.cache_stat.value_label.text = "—"
        self.last_result = None

    def _show_main_layout(self):
        self._stop_camera()
        self.root_box.clear_widgets()
        for widget in (
            self.title_label,
            self.subtitle,
            self.status_card,
            self.preview_card,
            self.result_card,
            self.stats_layout,
            self.gallery_button,
            self.camera_button,
            self.analyze_button,
            self.action_row,
            self.another_button,
            self.footer,
        ):
            self.root_box.add_widget(widget)

    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------
    def choose_gallery(self, *_):
        self._stop_camera()
        if ANDROID_AVAILABLE:
            self._open_android_gallery()
        else:
            self._open_desktop_gallery()

    def _open_desktop_gallery(self):
        from kivy.uix.filechooser import FileChooserListView

        chooser = FileChooserListView(
            filters=["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.webp", "*.WEBP"],
            multiselect=False,
            path=str(Path.home()),
        )
        chooser.bind(on_selection=self._desktop_selected)
        back = PremiumButton(text="← Back to Gothic OCR", button_color=PURPLE_DARK)
        back.bind(on_release=lambda *_: self._show_main_layout())
        card = PremiumCard(orientation="vertical", padding=dp(10), spacing=dp(10))
        card.add_widget(chooser)
        card.add_widget(back)
        self.root_box.clear_widgets()
        self.root_box.add_widget(self.title_label)
        self.root_box.add_widget(self.subtitle)
        self.root_box.add_widget(card)
        self._set_status("Choose the image you want to analyze")

    def _desktop_selected(self, chooser, selection):
        if selection:
            self._set_selected_image(selection[0])

    def _open_android_gallery(self):
        try:
            Intent = autoclass("android.content.Intent")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            current_activity = PythonActivity.mActivity

            self._unbind_activity("gallery")
            activity.bind(on_activity_result=self.on_gallery_result)
            self._gallery_bound = True

            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.setType("image/*")
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

            # If a device has no document picker, fall back to the classic
            # content picker. Both return a content:// URI that we copy into
            # app-private storage before OCR.
            package_manager = current_activity.getPackageManager()
            if intent.resolveActivity(package_manager) is None:
                intent = Intent(Intent.ACTION_GET_CONTENT)
                intent.addCategory(Intent.CATEGORY_OPENABLE)
                intent.setType("image/*")
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

            @run_on_ui_thread
            def launch():
                current_activity.startActivityForResult(intent, GALLERY_REQUEST)

            self._set_status("Opening gallery...")
            launch()
        except Exception as exc:
            print("GALLERY OPEN ERROR:", repr(exc))
            self._set_status(f"Gallery error: {type(exc).__name__}")

    def _copy_uri_to_private_file(self, uri, filename):
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        resolver = PythonActivity.mActivity.getContentResolver()
        destination = Path(self.user_data_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        # First try a direct file descriptor. This handles local providers well.
        parcel = None
        try:
            parcel = resolver.openFileDescriptor(uri, "r")
            if parcel is not None:
                fd = os.dup(int(parcel.getFd()))
                with os.fdopen(fd, "rb") as source, open(destination, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                parcel.close()
                return str(destination)
        except Exception as exc:
            print("URI FD COPY FALLBACK:", repr(exc))
            try:
                if parcel is not None:
                    parcel.close()
            except Exception:
                pass

        # Generic ContentResolver stream fallback.
        stream = resolver.openInputStream(uri)
        if stream is None:
            raise IOError("Android could not open the selected image")
        try:
            with open(destination, "wb") as target:
                while True:
                    value = stream.read()
                    if value == -1:
                        break
                    target.write(bytes((int(value) & 0xFF,)))
        finally:
            stream.close()
        return str(destination)

    def on_gallery_result(self, request_code, result_code, intent):
        if request_code != GALLERY_REQUEST:
            return
        self._unbind_activity("gallery")
        try:
            Activity = autoclass("android.app.Activity")
            if result_code != Activity.RESULT_OK or intent is None:
                self._set_status("Gallery cancelled")
                return
            uri = intent.getData()
            if uri is None:
                raise IOError("Gallery returned no image URI")
            path = self._copy_uri_to_private_file(uri, "gallery_input.jpg")
            self._normalize_image(path)
            self._set_selected_image(path)
        except Exception as exc:
            print("GALLERY RESULT ERROR:", repr(exc))
            self._set_status(f"Could not load image: {type(exc).__name__}")

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------
    def capture_camera(self, *_):
        if not ANDROID_AVAILABLE:
            self._set_status("Camera is available on Android")
            return
        try:
            if check_permission(Permission.CAMERA):
                self._start_android_camera()
            else:
                self._camera_permission_pending = True
                self._set_status("Allow camera permission...")
                request_permissions([Permission.CAMERA], self._camera_permission_callback)
        except Exception as exc:
            print("CAMERA PERMISSION ERROR:", repr(exc))
            self._set_status(f"Camera permission error: {type(exc).__name__}")

    def _camera_permission_callback(self, permissions, grants):
        if grants and all(bool(value) for value in grants):
            Clock.schedule_once(lambda dt: self._start_android_camera(), 0)
        else:
            self._set_status("Camera permission denied")

    def _start_android_camera(self):
        try:
            Intent = autoclass("android.content.Intent")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            current_activity = PythonActivity.mActivity

            self._unbind_activity("camera")
            activity.bind(on_activity_result=self.on_camera_result)
            self._camera_bound = True

            intent = Intent("android.media.action.IMAGE_CAPTURE")
            if intent.resolveActivity(current_activity.getPackageManager()) is None:
                raise RuntimeError("No camera application is available on this device")

            self._set_status("Opening camera...")

            @run_on_ui_thread
            def launch():
                current_activity.startActivityForResult(intent, CAMERA_REQUEST)

            launch()
        except Exception as exc:
            print("CAMERA START ERROR:", repr(exc))
            self._set_status(f"Camera error: {type(exc).__name__}")

    def on_camera_result(self, request_code, result_code, intent):
        if request_code != CAMERA_REQUEST:
            return
        self._unbind_activity("camera")
        try:
            Activity = autoclass("android.app.Activity")
            if result_code != Activity.RESULT_OK or intent is None:
                self._set_status("Camera cancelled")
                return

            # Camera applications are allowed to return either:
            #   1) a Bitmap in Intent extras (common/default path), or
            #   2) a content URI in Intent.getData().
            # Handle both so the button does not depend on one camera app.
            extras = intent.getExtras()
            bitmap = extras.get("data") if extras is not None else None

            if bitmap is not None:
                path = str(Path(self.user_data_dir) / "camera_input.jpg")
                FileOutputStream = autoclass("java.io.FileOutputStream")
                CompressFormat = autoclass("android.graphics.Bitmap$CompressFormat")
                stream = FileOutputStream(path)
                try:
                    if not bitmap.compress(CompressFormat.JPEG, 95, stream):
                        raise IOError("Camera bitmap compression failed")
                finally:
                    stream.close()
            else:
                uri = intent.getData()
                if uri is None:
                    raise IOError("Camera returned neither image bitmap nor image URI")
                path = self._copy_uri_to_private_file(uri, "camera_input.jpg")

            self._normalize_image(path)
            if not self._set_selected_image(path):
                raise IOError("Captured image could not be loaded")
            self._set_status("✦ Image captured successfully ✦")
        except Exception as exc:
            print("CAMERA RESULT ERROR:", repr(exc))
            self._set_status(f"Camera result error: {type(exc).__name__}")

    # --------------------------------------------------------
    # IMAGE VALIDATION / PREVIEW
    # --------------------------------------------------------
    def _normalize_image(self, path):
        from PIL import Image as PILImage
        image = PILImage.open(path)
        image.load()
        rgb = image.convert("RGB")
        rgb.save(path, "JPEG", quality=95)
        rgb.close()
        image.close()

    def _set_selected_image(self, image_path):
        path = Path(str(image_path))
        if not path.is_file():
            self._set_status("Invalid image")
            return False
        try:
            self._normalize_image(str(path))
        except Exception as exc:
            print("IMAGE VALIDATION ERROR:", repr(exc))
            self._set_status(f"Invalid image: {type(exc).__name__}")
            return False

        self.selected_image = str(path)
        self.preview.source = ""
        self.preview.reload()
        self.preview.source = self.selected_image
        self.preview.reload()
        self.result.text = "Image is ready for analysis"
        self._reset_stats()
        self._show_main_layout()
        self._set_status("✦ Image selected successfully ✦")
        return True

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------
    def analyze(self, *_):
        if not self.selected_image:
            self._set_status("⚠ Please select an image first")
            return
        if self.analyze_button.disabled:
            return

        image_path = Path(self.selected_image)
        if not image_path.is_file():
            self._set_status("Selected image no longer exists")
            return

        self.analyze_button.disabled = True
        self.gallery_button.disabled = True
        self.camera_button.disabled = True
        self.copy_button.disabled = True
        self.save_button.disabled = True
        self.another_button.disabled = True
        self._set_status("✦ Analyzing image... ✦")
        self.result.text = "Extracting Gothic characters..."

        threading.Thread(
            target=self._run_inference_thread,
            args=(str(image_path),),
            daemon=True,
        ).start()

    def _run_inference_thread(self, image_path):
        try:
            if self.ocr is None:
                from services.model_service import GothicOCR
                self.ocr = GothicOCR(MODEL_PATH)

            def progress_callback(current, total):
                self._update_progress(current, total)

            # IMPORTANT: model_service.predict expects the IMAGE PATH,
            # not a NumPy array. It handles RGB conversion, tiling and cache.
            result = self.ocr.predict(
                image_path,
                progress_callback=progress_callback,
                use_cache=True,
            )
            if not isinstance(result, dict):
                raise RuntimeError("Invalid OCR result")

            text = str(result.get("text") or "").strip()
            detections = result.get("detections") or []
            if not text:
                text = "No Gothic text was detected"

            stats = self._calculate_statistics(detections)
            cache_hit = bool(result.get("cache_hit", False))
            width = result.get("image_width", 0)
            height = result.get("image_height", 0)
            tiles = result.get("tiles_processed", 0)

            self._update_ui_success(text, stats, cache_hit, width, height, tiles)
        except Exception as exc:
            print("OCR ERROR:", repr(exc))
            self._update_ui_error(f"{type(exc).__name__}: {exc}")

    @mainthread
    def _update_progress(self, current, total):
        try:
            current = int(current)
            total = int(total)
            if total > 0:
                percent = max(0, min(100, int(current * 100 / total)))
                self._set_status(f"✦ Analyzing... {percent}% ({current}/{total})")
        except Exception:
            pass

    def _calculate_statistics(self, detections):
        confidences = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            value = detection.get("confidence", detection.get("score", 0))
            try:
                value = float(value)
                if value >= 0:
                    confidences.append(value)
            except Exception:
                pass
        average = sum(confidences) / len(confidences) if confidences else 0.0
        return {"count": len(detections), "average_confidence": average}

    @mainthread
    def _update_ui_success(self, text, stats, cache_hit, width, height, tiles):
        self.last_result = {
            "text": text,
            "statistics": stats,
            "cache_hit": cache_hit,
            "image_width": width,
            "image_height": height,
            "tiles_processed": tiles,
        }
        self.result.text = text
        self.char_stat.value_label.text = str(stats.get("count", 0))
        confidence = float(stats.get("average_confidence", 0.0))
        if confidence <= 1:
            confidence *= 100
        self.conf_stat.value_label.text = f"{confidence:.1f}%"
        self.line_stat.value_label.text = str(max(0, len([x for x in text.splitlines() if x.strip()])))
        self.cache_stat.value_label.text = "HIT" if cache_hit else "NEW"
        self._set_status("✦ Analysis completed successfully ✦")
        self._enable_actions()

    @mainthread
    def _update_ui_error(self, message):
        self.result.text = f"Unable to analyze the image:\n{message}"
        self._set_status("✦ An error occurred during analysis ✦")
        self.char_stat.value_label.text = "—"
        self.conf_stat.value_label.text = "—"
        self.line_stat.value_label.text = "—"
        self.cache_stat.value_label.text = "—"
        self._enable_actions()

    @mainthread
    def _enable_actions(self):
        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False
        self.copy_button.disabled = False
        self.save_button.disabled = False
        self.another_button.disabled = False

    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------
    def copy_text(self, *_):
        text = str(self.result.text or "").strip()
        if not text or text in {"—", "No image has been analyzed yet", "Image is ready for analysis"}:
            self._set_status("Nothing to copy yet")
            return
        try:
            Clipboard.copy(text)
            self._set_status("✓ Text copied to clipboard")
        except Exception as exc:
            self._set_status(f"Copy failed: {type(exc).__name__}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    def save_text(self, *_):
        text = str(self.result.text or "").strip()
        if not text or text in {"—", "No image has been analyzed yet", "Image is ready for analysis"}:
            self._set_status("Nothing to save yet")
            return

        if ANDROID_AVAILABLE:
            self._open_android_save_dialog()
        else:
            path = Path.home() / "gothic_ocr_result.txt"
            try:
                path.write_text(text, encoding="utf-8")
                self._set_status(f"✓ Saved: {path.name}")
            except Exception as exc:
                self._set_status(f"Save failed: {type(exc).__name__}")

    def _open_android_save_dialog(self):
        try:
            Intent = autoclass("android.content.Intent")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            current_activity = PythonActivity.mActivity
            self._unbind_activity("save")
            activity.bind(on_activity_result=self.on_save_result)
            self._save_bound = True

            intent = Intent(Intent.ACTION_CREATE_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.setType("text/plain")
            intent.putExtra(Intent.EXTRA_TITLE, "gothic_ocr_result.txt")

            @run_on_ui_thread
            def launch():
                current_activity.startActivityForResult(intent, SAVE_REQUEST)

            self._set_status("Choose where to save the TXT file...")
            launch()
        except Exception as exc:
            print("SAVE DIALOG ERROR:", repr(exc))
            self._set_status(f"Save dialog error: {type(exc).__name__}")

    def on_save_result(self, request_code, result_code, intent):
        if request_code != SAVE_REQUEST:
            return
        self._unbind_activity("save")
        try:
            Activity = autoclass("android.app.Activity")
            if result_code != Activity.RESULT_OK or intent is None:
                self._set_status("Save cancelled")
                return
            uri = intent.getData()
            if uri is None:
                raise IOError("No save location was returned")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            resolver = PythonActivity.mActivity.getContentResolver()
            stream = resolver.openOutputStream(uri)
            if stream is None:
                raise IOError("Could not open save destination")
            writer = None
            try:
                OutputStreamWriter = autoclass("java.io.OutputStreamWriter")
                writer = OutputStreamWriter(stream, "UTF-8")
                writer.write(str(self.result.text))
                writer.flush()
            finally:
                if writer is not None:
                    writer.close()
                else:
                    stream.close()
            self._set_status("✓ TXT saved successfully")
        except Exception as exc:
            print("SAVE RESULT ERROR:", repr(exc))
            self._set_status(f"Save failed: {type(exc).__name__}")

    # --------------------------------------------------------
    # RESET / BACK
    # --------------------------------------------------------
    def reset_image(self, *_):
        self._stop_camera()
        self.selected_image = None
        self.preview.source = ""
        self.preview.reload()
        self.result.text = "No image has been analyzed yet"
        self._reset_stats()
        self._enable_actions()
        self._show_main_layout()
        self._set_status("Ready to select another image")

    def _unbind_activity(self, which):
        if not ANDROID_AVAILABLE:
            return
        try:
            if which == "gallery" and self._gallery_bound:
                activity.unbind(on_activity_result=self.on_gallery_result)
                self._gallery_bound = False
            elif which == "camera" and self._camera_bound:
                activity.unbind(on_activity_result=self.on_camera_result)
                self._camera_bound = False
            elif which == "save" and self._save_bound:
                activity.unbind(on_activity_result=self.on_save_result)
                self._save_bound = False
        except Exception as exc:
            print("ACTIVITY UNBIND WARNING:", repr(exc))

    def _stop_camera(self):
        # Native Android camera is used instead of Kivy Camera, so there is
        # no live Camera widget to stop. Keep this method for screen cleanup.
        if ANDROID_AVAILABLE:
            self._unbind_activity("camera")

    def on_stop(self):
        self._stop_camera()
        self._unbind_activity("gallery")
        self._unbind_activity("save")
        if self.ocr is not None:
            try:
                self.ocr.close()
            except Exception:
                pass
            self.ocr = None


if __name__ == "__main__":
    GothicOCRApp().run()

