# ============================================================
# GOTHIC OCR — MAIN APPLICATION
# ============================================================

from pathlib import Path
import threading

from kivy.app import App
from kivy.clock import mainthread
from kivy.core.text import LabelBase
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.label import Label


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parent

FONT_PATH = (
    ROOT
    / "fonts"
    / "NotoSansGothic-Regular.ttf"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "gothic_ocr.tflite"
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

        # ----------------------------------------------------
        # OCR service
        #
        # يتم إنشاؤه عند أول تحليل حتى لا نثقل
        # تشغيل الواجهة مباشرة.
        # ----------------------------------------------------

        self.ocr = None

        # ====================================================
        # FONT
        # ====================================================

        self.gothic_font_available = (
            FONT_PATH.is_file()
        )

        if self.gothic_font_available:

            LabelBase.register(
                name="Gothic",
                fn_regular=str(FONT_PATH),
            )

        # ====================================================
        # ROOT LAYOUT
        # ====================================================

        self.root_box = BoxLayout(
            orientation="vertical",
            padding=12,
            spacing=8,
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status = Label(
            text="جاهز لاختيار صورة",
            size_hint_y=None,
            height=45,
        )

        # ====================================================
        # IMAGE PREVIEW
        # ====================================================

        self.preview = Image(
            allow_stretch=True,
            keep_ratio=True,
        )

        # ====================================================
        # RESULT
        # ====================================================

        self.result = Label(
            text="",
            font_name=(
                "Gothic"
                if self.gothic_font_available
                else "Roboto"
            ),
            font_size="20sp",
            halign="center",
            valign="middle",
        )

        self.result.bind(
            size=self._update_result_text_size
        )

        # ====================================================
        # GALLERY BUTTON
        # ====================================================

        self.gallery_button = Button(
            text="🖼 اختيار صورة",
            size_hint_y=None,
            height=52,
        )

        self.gallery_button.bind(
            on_release=self.choose_gallery
        )

        # ====================================================
        # CAMERA BUTTON
        # ====================================================

        self.camera_button = Button(
            text="📷 التقاط صورة",
            size_hint_y=None,
            height=52,
        )

        self.camera_button.bind(
            on_release=self.capture_camera
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================

        self.analyze_button = Button(
            text="🤖 تحليل الصورة",
            size_hint_y=None,
            height=52,
        )

        self.analyze_button.bind(
            on_release=self.analyze
        )

        # ====================================================
        # SHOW MAIN SCREEN
        # ====================================================

        self._show_main_layout()

        return self.root_box

    # ========================================================
    # RESULT TEXT SIZE
    # ========================================================

    def _update_result_text_size(
        self,
        instance,
        size,
    ):

        instance.text_size = (
            size[0] - 20,
            None,
        )

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def _show_main_layout(self):

        self.root_box.clear_widgets()

        self.root_box.add_widget(
            self.status
        )

        self.root_box.add_widget(
            self.preview
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
            self.result
        )

    # ========================================================
    # CHOOSE IMAGE FROM GALLERY
    # ========================================================

    def choose_gallery(self, *_):

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

        self.root_box.add_widget(
            chooser
        )

        self.status.text = (
            "اختاري صورة..."
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
                "تعذر الوصول إلى الصورة."
            )

            return

        # ----------------------------------------------------
        # Save selected image
        # ----------------------------------------------------

        self.selected_image = (
            selected_path
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        self.preview.source = (
            self.selected_image
        )

        self.preview.reload()

        # ----------------------------------------------------
        # Reset previous result
        # ----------------------------------------------------

        self.result.text = ""

        self.status.text = (
            "تم اختيار الصورة بنجاح"
        )

        self._show_main_layout()

    # ========================================================
    # CAMERA
    # ========================================================

    def capture_camera(self, *_):

        self.status.text = (
            "الكاميرا سيتم ربطها "
            "في مرحلة Android Native."
        )

    # ========================================================
    # START ANALYSIS
    # ========================================================

    def analyze(self, *_):

        if not self.selected_image:

            self.status.text = (
                "اختاري صورة أولًا."
            )

            return

        # ----------------------------------------------------
        # Prevent multiple simultaneous analyses
        # ----------------------------------------------------

        if self.analyze_button.disabled:

            return

        self.status.text = (
            "جاري التحليل واستخراج النص..."
        )

        self.result.text = ""

        self.analyze_button.disabled = True
        self.gallery_button.disabled = True
        self.camera_button.disabled = True

        image_path = (
            self.selected_image
        )

        # ----------------------------------------------------
        # Background thread
        # ----------------------------------------------------

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

            # =================================================
            # LOAD OCR SERVICE
            # =================================================

            if self.ocr is None:

                from services.model_service import (
                    GothicOCR
                )

                self.ocr = GothicOCR(
                    MODEL_PATH
                )

            # =================================================
            # RUN COMPLETE OCR PIPELINE
            # =================================================

            result = self.ocr.predict(
                image_path
            )

            # -------------------------------------------------
            # Extract text
            # -------------------------------------------------

            recognized_text = result.get(
                "text",
                "",
            )

            if not recognized_text:

                recognized_text = (
                    "لم يتم العثور على نص."
                )

            self._update_ui_success(
                recognized_text
            )

        except Exception as exc:

            self._update_ui_error(
                str(exc)
            )

    # ========================================================
    # SUCCESS UI UPDATE
    # ========================================================

    @mainthread
    def _update_ui_success(
        self,
        recognized_text,
    ):

        self.status.text = (
            "تم التحليل بنجاح!"
        )

        self.result.text = (
            recognized_text
        )

        self.analyze_button.disabled = False
        self.gallery_button.disabled = False
        self.camera_button.disabled = False

    # ========================================================
    # ERROR UI UPDATE
    # ========================================================

    @mainthread
    def _update_ui_error(
        self,
        error_msg,
    ):

        self.status.text = (
            f"خطأ أثناء التحليل: "
            f"{error_msg}"
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
