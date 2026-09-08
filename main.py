# ============================================================
# GOTHIC OCR — PREMIUM APPLICATION
# ============================================================

from pathlib import Path
import threading

from kivy.app import App
from kivy.clock import mainthread
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
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget


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


# ============================================================
# BACKGROUND WIDGET
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

    def _update(
        self,
        *_,
    ):

        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# CARD LAYOUT
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

    def _update_background(
        self,
        *_,
    ):

        self.background.pos = self.pos
        self.background.size = self.size

    def _update_color(
        self,
        *_,
    ):

        self.canvas.before.children[-1].rgba = (
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

        self.font_size = "18sp"

        self.size_hint_y = None
        self.height = dp(55)

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

    def _update_background(
        self,
        *_,
    ):

        self.background.pos = self.pos
        self.background.size = self.size

    def _update_color(
        self,
        *_,
    ):

        self.background.rgba = (
            self.button_color
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

        # ====================================================
        # REGISTER FONTS
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
        # MAIN ROOT
        # ====================================================

        root = BoxLayout()

        root.add_widget(
            PremiumBackground()
        )

        self.root_box = BoxLayout(
            orientation="vertical",
            padding=[
                dp(18),
                dp(18),
                dp(18),
                dp(18),
            ],
            spacing=dp(12),
        )

        root.add_widget(
            self.root_box
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
        )

        self.subtitle = Label(
            text="اكتشاف وتحليل النصوص القوطية",
            font_size="18sp",
            color=TEXT_DIM,
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(35),
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status = Label(
            text="جاهز لاختيار صورة",
            font_size="19sp",
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
        # IMAGE PREVIEW CARD
        # ====================================================

        self.preview_card = PremiumCard(
            orientation="vertical",
            padding=dp(10),
        )

        self.preview = Image(
            allow_stretch=True,
            keep_ratio=True,
        )

        self.preview_card.add_widget(
            self.preview
        )

        # ====================================================
        # RESULT
        # ====================================================

        self.result = Label(
            text="",
            font_name=(
                "Gothic"
                if self.gothic_font_available
                else self.ui_font
            ),
            font_size="22sp",
            color=GOLD,
            halign="center",
            valign="middle",
        )

        self.result.bind(
            size=self._update_text_size
        )

        self.result_card = PremiumCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(115),
            padding=dp(10),
            background_color=CARD_LIGHT,
        )

        self.result_card.add_widget(
            self.result
        )

        # ====================================================
        # GALLERY BUTTON
        # ====================================================

        self.gallery_button = PremiumButton(
            text="🖼   اختيار صورة من المعرض",
            font_name=self.ui_font,
            button_color=PURPLE,
        )

        self.gallery_button.bind(
            on_release=self.choose_gallery
        )

        # ====================================================
        # CAMERA BUTTON
        # ====================================================

        self.camera_button = PremiumButton(
            text="📷   التقاط صورة",
            font_name=self.ui_font,
            button_color=PURPLE_DARK,
        )

        self.camera_button.bind(
            on_release=self.capture_camera
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        self.analyze_button = PremiumButton(
            text="✦   تحليل الصورة الآن   ✦",
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
            text="✦ Gothic OCR • AI Powered ✦",
            font_size="14sp",
            color=TEXT_DIM,
            size_hint_y=None,
            height=dp(30),
        )

        # ====================================================
        # SHOW MAIN SCREEN
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
            size[0] - dp(15),
            None,
        )


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def _show_main_layout(self):

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
            self.gallery_button
        )

        self.root_box.add_widget(
            self.camera_button
        )

        self.root_box.add_widget(
            self.analyze_button
        )

        self.root_box.add_widget(
            self.result_card
        )

        self.root_box.add_widget(
            self.footer
        )


    # ========================================================
    # CHOOSE IMAGE
    # ========================================================

    def choose_gallery(
        self,
        *_,
    ):

        chooser = FileChooserListView(
            filters=[
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.webp",
            ],
            multiselect=False,
        )

        chooser.bind(
            on_selection=self._selected
        )

        self.root_box.clear_widgets()

        chooser_card = PremiumCard(
            orientation="vertical",
            padding=dp(10),
        )

        back_button = PremiumButton(
            text="← رجوع",
            font_name=self.ui_font,
            size_hint_y=None,
            height=dp(50),
        )

        back_button.bind(
            on_release=lambda *_:
            self._show_main_layout()
        )

        chooser_card.add_widget(
            chooser
        )

        chooser_card.add_widget(
            back_button
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

        if not Path(
            selected_path
        ).is_file():

            self.status.text = (
                "تعذر الوصول إلى الصورة"
            )

            return

        self.selected_image = (
            selected_path
        )

        self.preview.source = (
            self.selected_image
        )

        self.preview.reload()

        self.result.text = ""

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

        self.status.text = (
            "ميزة الكاميرا قادمة قريبًا 📷"
        )


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

        self.status.text = (
            "✦ جاري تحليل الصورة... ✦"
        )

        self.result.text = (
            "⌛ جاري استخراج النص..."
        )

        self.analyze_button.disabled = True
        self.gallery_button.disabled = True
        self.camera_button.disabled = True

        image_path = (
            self.selected_image
        )

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

            result = self.ocr.predict(
                image_path
            )

            recognized_text = result.get(
                "text",
                "",
            )

            if not recognized_text:

                recognized_text = (
                    "لم يتم العثور على نص"
                )

            self._update_ui_success(
                recognized_text
            )

        except Exception as exc:

            self._update_ui_error(
                str(exc)
            )


    # ========================================================
    # SUCCESS
    # ========================================================

    @mainthread
    def _update_ui_success(
        self,
        recognized_text,
    ):

        self.status.text = (
            "✦ تم التحليل بنجاح ✦"
        )

        self.result.text = (
            recognized_text
        )

        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False


    # ========================================================
    # ERROR
    # ========================================================

    @mainthread
    def _update_ui_error(
        self,
        error_msg,
    ):

        self.status.text = (
            "حدث خطأ أثناء التحليل"
        )

        self.result.text = (
            error_msg
        )

        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False


    # ========================================================
    # APP SHUTDOWN
    # ========================================================

    def on_stop(self):

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
