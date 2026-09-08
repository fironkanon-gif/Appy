# ============================================================
# GOTHIC OCR — PREMIUM APPLICATION
# Responsive UI + Gallery + Camera + OCR + Cache
# ============================================================

from pathlib import Path
import threading

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.text import LabelBase
from kivy.graphics import (
    Color,
    Rectangle,
    RoundedRectangle,
)
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget


# ============================================================
# OPTIONAL ANDROID PERMISSIONS
# ============================================================

try:
    from android.permissions import (
        Permission,
        check_permission,
        request_permissions,
    )
    ANDROID_AVAILABLE = True
except Exception:
    ANDROID_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

FONT_DIR = ROOT / "fonts"

GOTHIC_FONT_PATH = (
    FONT_DIR / "NotoSansGothic-Regular.ttf"
)

ARABIC_FONT_PATH = (
    FONT_DIR / "Amiri-Regular.ttf"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "gothic_ocr.tflite"
)


# ============================================================
# COLORS
# ============================================================

BACKGROUND = (
    0.035,
    0.025,
    0.075,
    1,
)

CARD = (
    0.10,
    0.07,
    0.18,
    1,
)

CARD_LIGHT = (
    0.15,
    0.10,
    0.25,
    1,
)

PURPLE = (
    0.55,
    0.30,
    0.90,
    1,
)

PURPLE_DARK = (
    0.30,
    0.12,
    0.55,
    1,
)

GOLD = (
    0.95,
    0.70,
    0.25,
    1,
)

GOLD_DARK = (
    0.55,
    0.34,
    0.08,
    1,
)

TEXT = (
    0.96,
    0.93,
    1,
    1,
)

TEXT_DIM = (
    0.72,
    0.65,
    0.82,
    1,
)

SUCCESS = (
    0.35,
    0.90,
    0.60,
    1,
)

ERROR = (
    1.0,
    0.35,
    0.40,
    1,
)


# ============================================================
# PREMIUM BACKGROUND
# ============================================================

class PremiumBackground(Widget):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        with self.canvas.before:

            Color(*BACKGROUND)

            self.rect = Rectangle(
                pos=self.pos,
                size=self.size,
            )

        self.bind(
            pos=self._update,
            size=self._update,
        )

    def _update(self, *_):

        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# PREMIUM CARD
# ============================================================

class PremiumCard(BoxLayout):

    background_color = ListProperty(CARD)

    def __init__(
        self,
        radius=22,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.radius = radius

        with self.canvas.before:

            Color(*self.background_color)

            self.background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[
                    dp(self.radius),
                ],
            )

        self.bind(
            pos=self._update_background,
            size=self._update_background,
            background_color=self._update_color,
        )

    def _update_background(self, *_):

        self.background.pos = self.pos
        self.background.size = self.size

    def _update_color(self, *_):

        self.background.rgba = (
            self.background_color
        )


# ============================================================
# PREMIUM BUTTON
# ============================================================

class PremiumButton(Button):

    button_color = ListProperty(PURPLE)

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.background_normal = ""
        self.background_down = ""

        self.color = TEXT
        self.font_size = "17sp"

        self.size_hint_y = None
        self.height = dp(56)

        with self.canvas.before:

            Color(*self.button_color)

            self.background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[
                    dp(18),
                ],
            )

        self.bind(
            pos=self._update_background,
            size=self._update_background,
            button_color=self._update_color,
        )

    def _update_background(self, *_):

        self.background.pos = self.pos
        self.background.size = self.size

    def _update_color(self, *_):

        self.background.rgba = (
            self.button_color
        )


# ============================================================
# SMALL STAT CARD
# ============================================================

class StatCard(PremiumCard):

    def __init__(
        self,
        title,
        value="0",
        **kwargs,
    ):

        super().__init__(
            orientation="vertical",
            padding=dp(8),
            spacing=dp(2),
            background_color=CARD,
            **kwargs,
        )

        self.title_label = Label(
            text=title,
            font_size="12sp",
            color=TEXT_DIM,
            halign="center",
            valign="middle",
            font_name="Arabic",
        )

        self.value_label = Label(
            text=value,
            font_size="20sp",
            color=GOLD,
            halign="center",
            valign="middle",
            font_name="Arabic",
        )

        self.add_widget(
            self.title_label
        )

        self.add_widget(
            self.value_label
        )

        self.bind(
            size=self._update_labels
        )

    def _update_labels(self, *_):

        for label in (
            self.title_label,
            self.value_label,
        ):

            label.text_size = (
                self.width - dp(10),
                None,
            )


# ============================================================
# APPLICATION
# ============================================================

class GothicOCRApp(App):

    # ========================================================
    # BUILD
    # ========================================================

    def build(self):

        self.title = "Gothic OCR"

        self.selected_image = None
        self.ocr = None

        self.camera_widget = None
        self.camera_running = False

        self.last_result = None

        # ====================================================
        # FONTS
        # ====================================================

        self.gothic_font_available = (
            GOTHIC_FONT_PATH.is_file()
        )

        self.arabic_font_available = (
            ARABIC_FONT_PATH.is_file()
        )

        if self.gothic_font_available:

            LabelBase.register(
                name="Gothic",
                fn_regular=str(
                    GOTHIC_FONT_PATH
                ),
            )

        if self.arabic_font_available:

            LabelBase.register(
                name="Arabic",
                fn_regular=str(
                    ARABIC_FONT_PATH
                ),
            )

        self.ui_font = (
            "Arabic"
            if self.arabic_font_available
            else "Roboto"
        )

        # ====================================================
        # ROOT
        # ====================================================

        root = FloatLayout()

        root.add_widget(
            PremiumBackground()
        )

        self.scroll = ScrollView(
            size_hint=(1, 1),
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(4),
        )

        self.root_box = BoxLayout(
            orientation="vertical",
            padding=[
                dp(18),
                dp(18),
                dp(18),
                dp(24),
            ],
            spacing=dp(12),
            size_hint_y=None,
        )

        self.root_box.bind(
            minimum_height=self.root_box.setter(
                "height"
            )
        )

        self.scroll.add_widget(
            self.root_box
        )

        root.add_widget(
            self.scroll
        )

        # ====================================================
        # HEADER
        # ====================================================

        self.title_label = Label(
            text="✦ GOTHIC OCR ✦",
            font_size="28sp",
            color=GOLD,
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(55),
            halign="center",
            valign="middle",
        )

        self.subtitle = Label(
            text="اكتشاف وتحليل النصوص القوطية بالذكاء الاصطناعي",
            font_size="16sp",
            color=TEXT_DIM,
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(42),
            halign="center",
            valign="middle",
        )

        self.subtitle.bind(
            size=self._update_text_size
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status = Label(
            text="جاهز لاختيار صورة",
            font_size="17sp",
            color=TEXT,
            font_name=self.ui_font,
            halign="center",
            valign="middle",
        )

        self.status.bind(
            size=self._update_text_size
        )

        self.status_card = PremiumCard(
            size_hint_y=None,
            height=dp(65),
            padding=dp(8),
        )

        self.status_card.add_widget(
            self.status
        )

        # ====================================================
        # PREVIEW
        # ====================================================

        self.preview_card = PremiumCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(300),
            padding=dp(10),
            background_color=CARD,
        )

        self.preview = Image(
            allow_stretch=True,
            keep_ratio=True,
        )

        self.preview_card.add_widget(
            self.preview
        )

        # ====================================================
        # RESULT TITLE
        # ====================================================

        self.result_title = Label(
            text="✦ النص المستخرج ✦",
            font_size="16sp",
            color=GOLD,
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(35),
            halign="center",
            valign="middle",
        )

        # ====================================================
        # RESULT TEXT
        # ====================================================

        self.result = Label(
            text="لم يتم تحليل أي صورة بعد",
            font_name=(
                "Gothic"
                if self.gothic_font_available
                else self.ui_font
            ),
            font_size="24sp",
            color=GOLD,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=dp(100),
        )

        self.result.bind(
            size=self._update_text_size
        )

        self.result_card = PremiumCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(155),
            padding=dp(10),
            spacing=dp(5),
            background_color=CARD_LIGHT,
        )

        self.result_card.add_widget(
            self.result_title
        )

        self.result_card.add_widget(
            self.result
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        self.stats_layout = BoxLayout(
            size_hint_y=None,
            height=dp(82),
            spacing=dp(8),
        )

        self.char_stat = StatCard(
            "الحروف",
            "0",
        )

        self.conf_stat = StatCard(
            "متوسط الثقة",
            "0%",
        )

        self.line_stat = StatCard(
            "الأسطر",
            "0",
        )

        self.cache_stat = StatCard(
            "Cache",
            "—",
        )

        self.stats_layout.add_widget(
            self.char_stat
        )

        self.stats_layout.add_widget(
            self.conf_stat
        )

        self.stats_layout.add_widget(
            self.line_stat
        )

        self.stats_layout.add_widget(
            self.cache_stat
        )

        # ====================================================
        # GALLERY
        # ====================================================

        self.gallery_button = PremiumButton(
            text="🖼  اختيار صورة من المعرض",
            font_name=self.ui_font,
            button_color=PURPLE,
        )

        self.gallery_button.bind(
            on_release=self.choose_gallery
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.camera_button = PremiumButton(
            text="📷  التقاط صورة بالكاميرا",
            font_name=self.ui_font,
            button_color=PURPLE_DARK,
        )

        self.camera_button.bind(
            on_release=self.capture_camera
        )

        # ====================================================
        # ANALYZE
        # ====================================================

        self.analyze_button = PremiumButton(
            text="✦  تحليل الصورة الآن  ✦",
            font_name=self.ui_font,
            button_color=GOLD,
        )

        self.analyze_button.color = (
            0.10,
            0.05,
            0.18,
            1,
        )

        self.analyze_button.bind(
            on_release=self.analyze
        )

        # ====================================================
        # FOOTER
        # ====================================================

        self.footer = Label(
            text="✦ Gothic OCR • AI Powered • TFLite ✦",
            font_size="13sp",
            color=TEXT_DIM,
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(35),
            halign="center",
            valign="middle",
        )

        # ====================================================
        # MAIN SCREEN
        # ====================================================

        self._show_main_layout()

        return root

    # ========================================================
    # TEXT SIZE
    # ========================================================

    def _update_text_size(
        self,
        instance,
        size,
    ):

        instance.text_size = (
            max(
                dp(20),
                size[0] - dp(15),
            ),
            None,
        )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def _show_main_layout(self):

        self._stop_camera()

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            self.title_label
        )

        self.root_box.add_widget(
            self.subtitle
        )

        self.root_box.add_widget(
            self.status_card
        )

        self.root_box.add_widget(
            self.preview_card
        )

        self.root_box.add_widget(
            self.result_card
        )

        self.root_box.add_widget(
            self.stats_layout
        )

        self.root_box.add_widget(
            self.gallery_button
        )

        self.root_box.add_widget(
            self.camera_button
        )

        self.root_box.add_widget(
            self.analyze_button
        )

        self.root_box.add_widget(
            self.footer
        )

    # ========================================================
    # GALLERY
    # ========================================================

    def choose_gallery(
        self,
        *_,
    ):

        self._stop_camera()

        chooser = FileChooserListView(
            filters=[
                "*.png",
                "*.PNG",
                "*.jpg",
                "*.JPG",
                "*.jpeg",
                "*.JPEG",
                "*.webp",
                "*.WEBP",
            ],
            multiselect=False,
            path=str(
                Path.home()
            ),
        )

        chooser.bind(
            on_selection=self._selected
        )

        back_button = PremiumButton(
            text="←  رجوع إلى Gothic OCR",
            font_name=self.ui_font,
            button_color=PURPLE_DARK,
        )

        back_button.bind(
            on_release=lambda *_:
            self._show_main_layout()
        )

        chooser_card = PremiumCard(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(10),
        )

        chooser_card.add_widget(
            chooser
        )

        chooser_card.add_widget(
            back_button
        )

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            self.title_label
        )

        self.root_box.add_widget(
            self.subtitle
        )

        self.root_box.add_widget(
            chooser_card
        )

        self.status.text = (
            "اختاري الصورة المراد تحليلها"
        )

    # ========================================================
    # IMAGE SELECTED
    # ========================================================

    def _selected(
        self,
        chooser,
        selection,
    ):

        if not selection:
            return

        selected_path = selection[0]

        path = Path(
            selected_path
        )

        if not path.is_file():

            self.status.text = (
                "تعذر الوصول إلى الصورة"
            )

            return

        self.selected_image = str(
            path
        )

        self.preview.source = (
            self.selected_image
        )

        self.preview.reload()

        self.result.text = (
            "الصورة جاهزة للتحليل"
        )

        self._reset_stats()

        self.status.text = (
            "✦ تم اختيار الصورة بنجاح ✦"
        )

        self._show_main_layout()

    # ========================================================
    # CAMERA
    # ========================================================

    def capture_camera(
        self,
        *_,
    ):

        if ANDROID_AVAILABLE:

            try:

                camera_permission = (
                    Permission.CAMERA
                )

                if not check_permission(
                    camera_permission
                ):

                    request_permissions(
                        [
                            camera_permission
                        ]
                    )

                    self.status.text = (
                        "اسمحي للتطبيق باستخدام الكاميرا ثم اضغطي الكاميرا مرة أخرى 📷"
                    )

                    return

            except Exception:

                pass

        self._open_camera_screen()

    # ========================================================
    # OPEN CAMERA SCREEN
    # ========================================================

    def _open_camera_screen(self):

        self._stop_camera()

        self.camera_widget = Camera(
            play=True,
            resolution=(
                1280,
                720,
            ),
            allow_stretch=True,
            keep_ratio=True,
        )

        self.camera_running = True

        capture_button = PremiumButton(
            text="●  التقاط وتحليل الصورة",
            font_name=self.ui_font,
            button_color=GOLD,
        )

        capture_button.color = (
            0.10,
            0.05,
            0.18,
            1,
        )

        capture_button.bind(
            on_release=self._take_camera_photo
        )

        back_button = PremiumButton(
            text="← رجوع",
            font_name=self.ui_font,
            button_color=PURPLE_DARK,
        )

        back_button.bind(
            on_release=lambda *_:
            self._show_main_layout()
        )

        camera_card = PremiumCard(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(10),
        )

        camera_card.add_widget(
            self.camera_widget
        )

        camera_card.add_widget(
            capture_button
        )

        camera_card.add_widget(
            back_button
        )

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            self.title_label
        )

        self.root_box.add_widget(
            Label(
                text="📷 الكاميرا",
                font_size="22sp",
                color=GOLD,
                font_name=self.ui_font,
                size_hint_y=None,
                height=dp(45),
            )
        )

        self.root_box.add_widget(
            camera_card
        )

        self.status.text = (
            "وجهي الكاميرا نحو النص القوطي"
        )

    # ========================================================
    # TAKE CAMERA PHOTO
    # ========================================================

    def _take_camera_photo(
        self,
        *_,
    ):

        if (
            self.camera_widget is None
            or not self.camera_running
        ):

            self.status.text = (
                "الكاميرا غير جاهزة"
            )

            return

        try:

            camera_dir = (
                Path.home()
                / "GothicOCR"
                / "captures"
            )

            camera_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                camera_dir
                / "gothic_capture.png"
            )

            self.camera_widget.texture.save(
                str(output_path),
                flipped=False,
            )

            self.selected_image = str(
                output_path
            )

            self._stop_camera()

            self.preview.source = (
                self.selected_image
            )

            self.preview.reload()

            self.result.text = (
                "تم التقاط الصورة — جاهزة للتحليل"
            )

            self._reset_stats()

            self.status.text = (
                "✦ تم التقاط الصورة بنجاح ✦"
            )

            self._show_main_layout()

        except Exception as exc:

            self.status.text = (
                "تعذر التقاط الصورة"
            )

            self.result.text = str(
                exc
            )

    # ========================================================
    # STOP CAMERA
    # ========================================================

    def _stop_camera(self):

        if self.camera_widget is not None:

            try:

                self.camera_widget.play = False

            except Exception:

                pass

            self.camera_widget = None

        self.camera_running = False

    # ========================================================
    # RESET STATS
    # ========================================================

    def _reset_stats(self):

        self.char_stat.value_label.text = "0"
        self.conf_stat.value_label.text = "0%"
        self.line_stat.value_label.text = "0"
        self.cache_stat.value_label.text = "—"

        self.last_result = None

    # ========================================================
    # START ANALYSIS
    # ========================================================

    def analyze(
        self,
        *_,
    ):

        if not self.selected_image:

            self.status.text = (
                "⚠ اختاري صورة أولًا"
            )

            return

        if self.analyze_button.disabled:
            return

        image_path = (
            self.selected_image
        )

        if not Path(
            image_path
        ).is_file():

            self.status.text = (
                "الصورة غير موجودة"
            )

            return

        self.status.text = (
            "✦ جاري تحليل الصورة... ✦"
        )

        self.result.text = (
            "⌛ جاري استخراج الحروف..."
        )

        self.analyze_button.disabled = True
        self.gallery_button.disabled = True
        self.camera_button.disabled = True

        threading.Thread(
            target=self._run_inference_thread,
            args=(image_path,),
            daemon=True,
        ).start()

    # ========================================================
    # BACKGROUND INFERENCE
    # ========================================================

    def _run_inference_thread(
        self,
        image_path,
    ):

        try:

            if self.ocr is None:

                from services.model_service import (
                    GothicOCR
                )

                self.ocr = GothicOCR(
                    MODEL_PATH
                )

            def progress_callback(
                current,
                total,
            ):

                self._update_progress(
                    current,
                    total,
                )

            result = self.ocr.predict(
                image_path,
                progress_callback=progress_callback,
                use_cache=True,
            )

            if not isinstance(
                result,
                dict,
            ):

                raise RuntimeError(
                    "ناتج OCR غير صالح"
                )

            recognized_text = (
                result.get(
                    "text",
                    "",
                )
                or ""
            )

            detections = (
                result.get(
                    "detections",
                    [],
                )
                or []
            )

            lines = (
                result.get(
                    "lines",
                    [],
                )
                or []
            )

            cache_hit = bool(
                result.get(
                    "cache_hit",
                    False,
                )
            )

            image_width = result.get(
                "image_width",
                0,
            )

            image_height = result.get(
                "image_height",
                0,
            )

            tiles_processed = result.get(
                "tiles_processed",
                0,
            )

            if not recognized_text.strip():

                recognized_text = (
                    "لم يتم العثور على نص قوطي"
                )

            statistics = (
                self._calculate_statistics(
                    detections
                )
            )

            self._update_ui_success(
                recognized_text,
                statistics,
                cache_hit,
                image_width,
                image_height,
                tiles_processed,
            )

        except Exception as exc:

            self._update_ui_error(
                str(exc)
            )

    # ========================================================
    # PROGRESS
    # ========================================================

    @mainthread
    def _update_progress(
        self,
        current,
        total,
    ):

        try:

            current = int(current)
            total = int(total)

            if total > 0:

                percent = int(
                    (current / total) * 100
                )

                self.status.text = (
                    f"✦ جاري التحليل... "
                    f"{percent}% "
                    f"({current}/{total})"
                )

        except Exception:

            pass

    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    def _calculate_statistics(
        self,
        detections,
    ):

        count = len(
            detections
        )

        confidences = []

        for detection in detections:

            if not isinstance(
                detection,
                dict,
            ):

                continue

            confidence = (
                detection.get(
                    "confidence",
                    detection.get(
                        "score",
                        0,
                    ),
                )
            )

            try:

                confidence = float(
                    confidence
                )

                if confidence >= 0:

                    confidences.append(
                        confidence
                    )

            except Exception:

                continue

        if confidences:

            average = (
                sum(confidences)
                / len(confidences)
            )

        else:

            average = 0.0

        return {
            "count": count,
            "average_confidence": average,
        }

    # ========================================================
    # SUCCESS
    # ========================================================

    @mainthread
    def _update_ui_success(
        self,
        recognized_text,
        statistics,
        cache_hit,
        image_width,
        image_height,
        tiles_processed,
    ):

        self.last_result = {
            "text": recognized_text,
            "statistics": statistics,
            "cache_hit": cache_hit,
        }

        self.status.text = (
            "✦ تم التحليل بنجاح ✦"
        )

        self.result.text = (
            recognized_text
        )

        self.char_stat.value_label.text = (
            str(
                statistics.get(
                    "count",
                    0,
                )
            )
        )

        average_confidence = (
            statistics.get(
                "average_confidence",
                0,
            )
        )

        if average_confidence <= 1:

            average_confidence *= 100

        self.conf_stat.value_label.text = (
            f"{average_confidence:.1f}%"
        )

        lines = (
            self._estimate_line_count(
                recognized_text
            )
        )

        self.line_stat.value_label.text = (
            str(lines)
        )

        if cache_hit:

            self.cache_stat.value_label.text = (
                "HIT"
            )

        else:

            self.cache_stat.value_label.text = (
                "NEW"
            )

        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False

    # ========================================================
    # ESTIMATE LINES
    # ========================================================

    def _estimate_line_count(
        self,
        text,
    ):

        if not text:
            return 0

        lines = [
            line
            for line in str(text).splitlines()
            if line.strip()
        ]

        return max(
            1,
            len(lines),
        )

    # ========================================================
    # ERROR
    # ========================================================

    @mainthread
    def _update_ui_error(
        self,
        error_msg,
    ):

        self.status.text = (
            "✦ حدث خطأ أثناء التحليل ✦"
        )

        self.result.text = (
            f"تعذر تحليل الصورة:\n{error_msg}"
        )

        self.char_stat.value_label.text = "—"
        self.conf_stat.value_label.text = "—"
        self.line_stat.value_label.text = "—"
        self.cache_stat.value_label.text = "—"

        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False

    # ========================================================
    # APP SHUTDOWN
    # ========================================================

    def on_stop(self):

        self._stop_camera()

        if self.ocr is not None:

            try:

                self.ocr.close()

            except Exception:

                pass

            finally:

                self.ocr = None


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    GothicOCRApp().run()
