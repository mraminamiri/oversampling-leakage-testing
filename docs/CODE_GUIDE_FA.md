# راهنمای کوتاه کد

## سؤال پروژه

اگر سامپلر عمداً دادهٔ Test را هم ببیند، کدام بررسی این خطا را پیدا می‌کند؟ هدف اصلی گرفتن بالاترین F1 نیست؛ می‌خواهیم رفتار بررسی‌ها و محدودیت آن‌ها را بفهمیم.

## مسیر اجرا

از پوشهٔ اصلی `python run_project.py` را اجرا می‌کنیم. این فایل داده را بررسی می‌کند، تست‌ها را اجرا می‌کند، جدول تشخیص را می‌سازد، هفت مدل را ارزیابی می‌کند و شکل‌ها را ذخیره می‌کند. برای اجرای کامل به نوت‌بوک نیاز نداریم.

1. `pipelines.py`: خواندن داده و ساخت نسخه‌های درست و معیوب.
   تابع‌های قابل استفادهٔ مجدد اراکل‌ها در `oracles.py` قرار دارند.
2. `test_*.py`: بررسی رفتار مورد انتظار، درست بودن ردیابی و چند حالت دشوار.
3. `detection_matrix.py`: اجرای سه اراکل روی شش حالت نمونه‌گیری.
4. `evaluate_models.py`: محاسبهٔ معیارهای هفت مدل روی همان Test اصلی.
5. `plot_results.py`: تبدیل جدول‌های ذخیره‌شده به نمودار.

## تابع‌های اصلی

| تابع | کار آن |
|---|---|
| `load_data` | فایل CSV را می‌خواند و شمارهٔ ردیف اصلی را به‌عنوان شناسه نگه می‌دارد. |
| `original_split` | تقسیم ثابت و stratified با ۶۱۴ Train و ۱۵۴ Test می‌سازد. |
| `make_preprocessor` | صفرهای مشخص را missing می‌کند؛ imputer و scaler جدید می‌سازد. |
| `make_reference_pipeline` | مدل بدون sampling یا یکی از سه پایپلاین سالم را می‌سازد. |
| `make_buggy_ros/smote/adasyn` | سامپلر را به کل داده دسترسی می‌دهد و فقط ردیف‌های اضافه‌شده را به Train اصلی می‌افزاید. |
| `TracedSMOTE` | والدهای واقعی نمونه‌های مصنوعی SMOTE را ثبت می‌کند. |
| `trace_adasyn_generation` | تولید ADASYN را بازسازی و با خروجی واقعی آن مقایسه می‌کند. |
| `run_oracles` | شواهد هر سناریو را به سه بررسی می‌دهد. |

شناسه‌ها ویژگی ورودی مدل نیستند. آن‌ها فقط کمک می‌کنند منشأ رکوردها را بدانیم. شمارهٔ خاص شناسه مهم نیست؛ نگاشت درست آن به رکورد اصلی مهم است.

## mutation دقیقاً چیست؟

در نسخهٔ سالم، سامپلر فقط Train و برچسب‌های آن را می‌گیرد. در نسخهٔ معیوب، تمام ۷۶۸ رکورد و برچسب‌ها را می‌گیرد. سپس فقط خروجی‌های اضافه‌شده به Train اصلی افزوده می‌شوند. Test اصلی برای ارزیابی ثابت می‌ماند.

این یک خانوادهٔ خطاست که روی سه روش اجرا شده است. این آزمایش دقیقاً «اول sampling، بعد split» نیست. imputer و scaler در هر دو نسخه فقط روی Train یاد می‌گیرند.

## معنی نتایج

- **Source Isolation:** آیا در شناسه‌های ثبت‌شدهٔ ورودی سامپلر، Test وجود دارد؟
- **Exact Duplicate:** آیا یک ردیف اضافه‌شده از نظر مقدار ویژگی‌ها دقیقاً با Test برابر است؟
- **Generated Provenance:** آیا ردیف اضافه‌شده والد مستقیم از Test دارد؟

عددهای ۵۴، ۷۸ و ۸۸ تعداد ردیف‌های اضافه‌شده با والد مستقیم از Test هستند؛ تعداد افراد متفاوت Test نیستند. نتیجهٔ `pytest PASS` یعنی انتظار تست درست بوده؛ ممکن است انتظار این بوده باشد که خطای عمدی شناسایی شود.

در حالت حذف کپی‌های Test، اراکل والدها قاعدهٔ محدود خودش را قبول می‌کند، اما دسترسی قبلی سامپلر همچنان غلط بوده است. شباهت دقیق دو ردیف مستقل نیز می‌تواند هشدار اشتباه دربارهٔ نشت بدهد.

## پنج جواب کوتاه برای جلسه

**What is the mutation?**

> I change the sampler's input. The faulty version sees both Train and Test. I apply this change to three samplers.

**Where do the checks come from?**

> The basic rule is to keep Test information out of training. I turn this rule into checks on input IDs, exact matches, and direct parents.

**Are the checks dependent on source IDs?**

> Yes, two checks need reliable source information. Exact Duplicate only compares feature values. This is a limitation of the prototype.

**Does a higher F1 prove leakage?**

> No. A score alone cannot tell us whether the data were used correctly. The faulty versions also have different training sizes and class balances.

**What is the next step?**

> Some input metadata is supplied by the experiment code. My next step is to record actual sampler inputs independently and test different fault types.

## چیزی که هنوز انجام نشده

ثبت مستقل و عمومیِ ورودی واقعی سامپلر، نتیجهٔ سه‌حالته برای شواهد ناقص، و ارزیابی روی چند دیتاست و چند seed هنوز پیاده‌سازی نشده‌اند. این نسخه یک آزمایش کنترل‌شده است؛ نتیجهٔ آن را به همهٔ پایپلاین‌ها تعمیم نمی‌دهیم.

در این نسخه، ساختارهای ناقصِ مشخص مثل نبود والد برای یک ردیف اضافه‌شده خطا می‌دهند؛ این بررسی ساختار، صحت واقعیِ همهٔ اطلاعات ثبت‌شده را تضمین نمی‌کند.
