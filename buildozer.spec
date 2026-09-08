[app]

title = GothicOCR
package.name = gothicocr
package.domain = org.gothicocr
version = 1.0.0

source.dir = .
source.include_exts = py,png,jpg,jpeg,webp,ttf,tflite,json
source.exclude_dirs = tests,__pycache__,.buildozer,bin,.git

requirements = python3,kivy,numpy,pillow,pyjnius

orientation = portrait
fullscreen = 0

android.permissions = READ_EXTERNAL_STORAGE,READ_MEDIA_IMAGES

android.api = 35
android.minapi = 23

android.archs = arm64-v8a

android.ndk = 25b

android.enable_androidx = True

# TensorFlow Lite is provided as an Android/Java dependency.
android.gradle_dependencies = org.tensorflow:tensorflow-lite:2.17.0
android.add_src = java
android.private_storage = True

p4a.bootstrap = sdl2

android.debug_artifact = apk
android.release_artifact = aab

android.logcat_filters = *:S python:D

android.allow_backup = False

p4a.branch = master
