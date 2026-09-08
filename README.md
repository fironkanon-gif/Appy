# GothicOCR

تطبيق Android مبني باستخدام Python + Kivy للتعرّف على الحروف القوطية (Gothic) من الصور باستخدام نموذج TensorFlow Lite.

## بنية المشروع

- `main.py` — واجهة التطبيق وتشغيل التحليل.
- `services/image_service.py` — تجهيز الصورة إلى NHWC بحجم 1024×1024 مع letterbox.
- `services/model_service.py` — تحميل وتشغيل نموذج TFLite.
- `services/text_decoder.py` — فك مخرجات YOLO وترتيب الحروف.
- `models/gothic_ocr.tflite` — نموذج OCR.
- `data/labels.json` — 25 حرفًا قوطيًا.
- `fonts/NotoSansGothic-Regular.ttf` — خط عرض الحروف.
- `buildozer.spec` — إعداد بناء Android.
- `.circleci/config.yml` — بناء APK عبر CircleCI مع موافقة يدوية.

## مواصفات النموذج

- Input: `[1, 1024, 1024, 3]` — `float32`
- Output: `[1, 29, 21504]` — `float32`
- Classes: `25`

## البناء

المشروع يستخدم `tflite-runtime` من خلال recipe الخاص بـ python-for-android. يوجد في CircleCI زر موافقة يدوي قبل خطوة البناء حتى لا يبدأ Build بالخطأ.

> لا يتم تشغيل CircleCI تلقائيًا من ملفات المشروع هنا؛ يجب تنفيذ البناء فقط بعد الموافقة اليدوية.
