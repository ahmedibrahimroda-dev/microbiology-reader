# Microbiology Lecture Notes Builder

أداة لتحويل فيديوهات أو تسجيلات المحاضرات إلى مذكرات Word منظمة ومناسبة للطلبة.

## المزايا
- استخراج الصوت من الفيديو عبر ffmpeg
- تفريغ المحاضرة صوتيًا عبر faster-whisper
- استخراج النص الظاهر على الشرائح عبر OCR
- دمج النص المسموع مع النص الظاهر
- إعادة صياغة المحتوى إلى ملاحظات دراسية منظمة
- تصدير Word و JSON
- واجهة Streamlit بسيطة
- جاهز للنشر على Streamlit Community Cloud

## الملفات
- `microbio_notes_tool.py` : المحرك الأساسي
- `app.py` : واجهة Streamlit
- `requirements.txt` : المتطلبات
- `packages.txt` : الحزم الخارجية للنشر السحابي
- `.python-version` : إصدار بايثون المطلوب
- `.streamlit/config.toml` : إعدادات Streamlit
- `.streamlit/secrets.toml.example` : نموذج الأسرار
- `.gitignore` : تجاهل الملفات الحساسة

## التشغيل المحلي
```bash
pip install -r requirements.txt
streamlit run app.py
```

لازم يكون `ffmpeg` مثبت على الجهاز.
ولو هتستخدم OCR محليًا فثبّت `tesseract`.

## التشغيل من سطر الأوامر
```bash
python microbio_notes_tool.py lecture.mp4 --title "History and Nature of Virology"
```

## تجهيز الأسرار محليًا
انسخ `.streamlit/secrets.toml.example` إلى `.streamlit/secrets.toml` ثم ضع:
```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```

## النشر على Streamlit Community Cloud
1. ارفع المشروع إلى GitHub.
2. ادخل إلى Streamlit Community Cloud.
3. اختر الريبو وحدد الملف الرئيسي `app.py`.
4. من إعدادات التطبيق أضف Secrets بنفس محتوى ملف المثال:
```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-4o-mini"
```
5. اضغط Deploy.

## ملاحظات
- لو OpenAI API غير متاح، الأداة تستخدم Rule-based writer.
- لو faster-whisper غير متاح، يمكن التحويل إلى dummy mode للتجربة.
- في النشر السحابي، `packages.txt` يساعد على تثبيت `ffmpeg` و `tesseract-ocr`.