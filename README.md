# 🎯 GPU Optimizasyonlu Sihirli Kare Çözücü

Bu proje, **MagicSquareGenerator** ile otomatik sihirli kare haritaları üretip, bu haritaları kullanarak listedeki ifadelerin ebced değerlerini yerleştirip sihirli kare olup olmadığını test eden yüksek performanslı bir Python uygulamasıdır.

https://github.com/metatronslove/square-checker/blob/5755e63dfcf47bf3c409a6e37b6e105af8e1779d/docs/Ekran_G%C3%B6r%C3%BCnt%C3%BCs%C3%BC_20251104_050723.png

## 🚀 Özellikler

- **MagicSquareGenerator Entegrasyonu**: Otomatik sihirli kare haritası üretimi
- **GPU Hızlandırma**: CUDA/ROCm desteği ile yüksek performans
- **Çok Dilli Ebced**: 5 farklı dil ve çoklu ebced tabloları
- **Akıllı Devam Etme**: Kesilen işlemler checkpoint'ten devam
- **Detaylı Raporlama**: Zengin terminal arayüzü ve loglama
- **Gerçek Zamanlı İzleme**: Sistem ve GPU performans takibi

## 📋 Kurulum

### 1. Hızlı Kurulum

```bash
# Tüm bağımlılıklarla birlikte
pip install -r requirements.txt

# Sadece temel bağımlılıklar (GPU olmadan)
pip install numpy scipy tqdm colorama pandas
```

### 2. Platforma Özel Kurulum

**NVIDIA GPU ile (CUDA):**
```bash
# CUDA 11.x için
pip install cupy-cuda11x

# CUDA 12.x için
pip install cupy-cuda12x

# GPU izleme araçları
pip install GPUtil pycuda
```

**AMD GPU ile (ROCm):**
```bash
pip install cupy-rocm-5-3
pip install pyopencl
```

**CPU Modu (GPU olmadan):**
```bash
pip install numpy scipy tqdm colorama pandas psutil joblib
```

### 3. Geliştirici Kurulumu

```bash
# Geliştirme araçlarıyla birlikte
pip install -r requirements.txt
pip install black flake8 pytest

# Gelişmiş özellikler
pip install rich loguru matplotlib dask
```

## 🛠️ Sistem Gereksinimleri

### Minimum Sistem
- **Python**: 3.8+
- **RAM**: 4GB
- **Depolama**: 1GB boş alan
- **İşletim Sistemi**: Windows 10+, Linux, macOS

### Önerilen Sistem (GPU ile)
- **Python**: 3.9+
- **RAM**: 16GB+
- **GPU**: NVIDIA (CUDA 11.x/12.x) veya AMD (ROCm 5.3+)
- **VRAM**: 8GB+
- **Depolama**: NVMe SSD, 5GB+ boş alan

### Bağımlılık Matrisi

| Kategori | Paket | Versiyon | Gerekli | Açıklama |
|----------|-------|----------|---------|-----------|
| **Temel** | `numpy` | ≥1.21.0 | ✅ | Matematik işlemleri |
| | `scipy` | ≥1.7.0 | ✅ | Bilimsel hesaplamalar |
| **GPU** | `cupy-cuda11x` | ≥9.0.0 | ❌ | NVIDIA CUDA 11.x |
| | `cupy-cuda12x` | ≥9.0.0 | ❌ | NVIDIA CUDA 12.x |
| | `cupy-rocm-5-3` | ≥9.0.0 | ❌ | AMD ROCm |
| **İlerleme** | `tqdm` | ≥4.62.0 | ✅ | İlerleme çubuğu |
| | `colorama` | ≥0.4.4 | ✅ | Renkli çıktı |
| **Veri** | `pandas` | ≥1.3.0 | ❌ | Veri analizi |
| **Görsel** | `matplotlib` | ≥3.4.0 | ❌ | Grafikler |
| **Sistem** | `psutil` | ≥5.8.0 | ✅ | Sistem izleme |
| | `GPUtil` | ≥1.4.0 | ❌ | GPU izleme |
| **Geliştirme** | `pytest` | ≥6.2.0 | ❌ | Testler |
| | `black` | ≥21.0.0 | ❌ | Kod formatlama |
| | `flake8` | ≥3.9.0 | ❌ | Kod kalitesi |
| **Paralel** | `joblib` | ≥1.0.0 | ❌ | Paralel işleme |
| | `dask` | ≥2021.10.0 | ❌ | Dağıtık hesaplama |
| **Loglama** | `loguru` | ≥0.5.0 | ❌ | Gelişmiş loglama |
| **UI** | `rich` | ≥10.0.0 | ❌ | Zengin terminal |

## 🎯 Kullanım

### 1. Temel Kullanım

```bash
# 3x3 sihirli kareler için, 5 harita ile
python checksquares.py phrases.txt --size 3 --map-count 5

# 4x4 sihirli kareler için, 10 harita ile
python checksquares.py phrases.txt --size 4 --map-count 10

# Özel çıktı dizini ile
python checksquares.py phrases.txt --size 3 --output my_results
```

### 2. Girdi Dosyası Formatı

**Basit format** (`phrases.txt`):
```الله 
الرحمن 
الرحيم 
العدل 
العفو 
الآخر 
العليم 
العلي 
العظيم 
العزيز 
الباعث 
الباقي 
البارئ 
البصير 
الباسط 
الباطن 
البديع 
البَرّ 
الجامع 
الجبّار 
الجليل 
الضار 
الأوّل 
الفتّاح 
الغفّار 
الغفور 
الغني 
الخبير 
الهادي 
الخافض 
الحفيظ 
الحكم 
الحكيم 
الحقّ 
الخالق 
الحليم 
الحميد 
الحسيب 
الحيّ 
القابض 
القادر 
القهّار 
القويّ 
القيّوم 
الكبير 
الكريم 
القدّوس 
اللطيف 
الماجد 
مالك الملك 
المانع 
المجيد 
الملك 
المتين 
المؤخّر 
المجيب 
المغني 
المحسي 
المحيي 
المعيد 
المعز 
المقدّم 
المقيت 
المقسط 
المقتدر 
المصور 
المبدىء 
المهيْمن 
المؤمن 
المميت 
المنتقم 
المتعالِ 
المتكبّر 
المذل 
النافع 
النور 
الرافع 
الرقيب 
الرؤوف 
الرشيد 
الرزّاق 
الصبور 
الصمد 
الشهيد 
الشكور 
السلام 
السميع 
التوّاب 
الواجد 
الواحد 
الوالي 
الوارث 
الواسع 
الودود 
الوهّاب 
الوكيل 
الولي 
الظاهر 
ذو الجلال والإكرام 

```

**Gelişmiş format**:
```python
phrases = [
الله
الرحمن
الرحيم
الملك
القدّوس
السلام
المومن
المهيمن
العزيز
الجبار
المتكبّر
الخالق
البارئ
المصور
الغفار
القهّار
الوهّاب
الرزاق
الفتاح
العليم
القابض
الباسط
الخافض
الرافع
المعز
المذل
السميع
البصير
الحكم
العدل
اللطيف
الخبير
الحليم
العظيم
الغفور
الشكور
العلي
الكبير
الحفيظ
المُقيت
الحسيب
الجليل
الكريم
الرقيب
المجيب
الواسع
الحكيم
الودود
المجيد
الباعث
الشهيد
الحق
الوكيل
القوي
المتين
الولي
الحميد
المحصي
المبدئ
المعيد
المحيي
المميت
الحي
القيوم
الواجد
الماجد
الواحد
الصمد
القادر
المقتدر
المقدم
المؤخر
الأول
الآخر
الظاهر
الباطن
الوالي
المتعالِ
البر
التواب
المنتقم
العفو
الرؤوف
الاجود
الفرد
المقسط
الجامع
الغني
المغني
المانع
الضار
النافع
النور
الهادي
البديع
الباقي
الوارث
الرشيد
الصبور
الجميل
القاهر
القريب
الراشد
الرب
المبين
البرهان
الشديد
الواقي
البار
ذو القوة
القائم
الدائم
الحافظ
الفاطر
السامع
المعطي
الكافي
الأبد
العالم
الصادق
المنير
التام
القديم
الوتر
الأحد
المالك
المليك
الجواد
الخلاق
الدافع
الديان
الرازق
الرفيق
السيد
السبوح
السريع
الستار
الشافي
الشاهد
الشاكر
الصاحب
الطيب
الطهر
الأعلى
العلام
الغافر
الفاتح
القدير
القيام
القاضي
الكفيل
المقدر
المعين
المنان
المفضل
الموسع
المنعم
المفرج
المعافي
المُطعم
الناصر
النذير
الوافي
البادئ
الذارئ
الصانع
المحيط
الحنان
الأكرم
الحييّ
الطالب
الأعز
المحسان
المسعَّر
الدهر
الكائن
القيم
الطبيب
المريد
المحب
المبغض
الرضا
السخط
الحفي
الغيور
المُبرم
المنذر
المدبر
الممتحن
البالي
المُبلي
المبتلي
الفاتن
النصير
المستعان
المعبود
الحاكم
الأحكم
الرفيع
الأقوى
الأقرب
الفاعل
الأعظم
المستمع
الكاشف
عالم الغيب والشهادة
علام الغيوب
ذو الفضل العظيم
ذو العرش المجيد
ذو الطول والإحسان
الغياث
ذو الرحمة الواسعة
ذو المعارج
ذو الإنتقام
ذو الجبروت والملكوت
منزل الكتاب
كاشف الكرب
الأكبر
أسرع الحاسبين
ولي المؤمنين
جاعل الليل سكناً
فالق الإصباح
بالغ أمره
المحسن
مخرج الحي من الميت
مخرج الميت من الحي
فعال لما يريد
كاشف الضر
فالق الحب والنوى
واسع المغفرة
الإله
غافر الذنب
مقلب القلوب
غالب على أمره
خير الفاصلين
خير الناصرين
الأعلم
خير الفاتحين
خير الراحمين
خير الغافرين
أرحم الراحمين
خير المنزلين
خير الماكرين
المرسِل
خير الحاكمين
خير الرازقين
خير الحافظين
خير الوارثين
الرفيع الدرجات
المغيث
شديد المحال
أهل التقوى
أهل المغفرة
فارج الهم
قابل التوب
السريع الحساب
المولى
عدو الكافرين
المخزي الكافرين
أحكم الحاكمين
أحسن الخالقين
نعم القادر
المذكور
نعم المولى
نعم القاهر
نعم الماهد
فاطر السموات والأرض
نور السماوات والأرض
متم نوره
الحاسب
المنشئ
المُنزل
الكامل
مالك الملك
ذو الجلال والإكرام
الكاتب
]
```

### 3. Gelişmiş Seçenekler

```bash
# GPU ile yüksek performans
python checksquares.py phrases.txt --size 4 --batch-size 50000 --map-count 8

# Çoklu ebced tabloları
python checksquares.py phrases.txt --size 3 --tables 1 7 12 --lang arabic

# Performans izleme ile
python checksquares.py phrases.txt --size 3 --map-count 5 --monitor-gpu

# Checkpoint yönetimi
python checksquares.py phrases.txt --size 3 --no-resume  # Yeniden başlat
```

## ⚙️ Parametreler

| Parametre | Kısa | Açıklama | Varsayılan |
|-----------|------|-----------|------------|
| `input_file` | - | İfade listesi dosyası | **Zorunlu** |
| `--size` | `-s` | Kare boyutu (3-8) | `3` |
| `--tables` | `-t` | Ebced tablo kodları | Dil varsayılanı |
| `--shadda` | `-S` | Şedde hesabı (Arapça) | `1` |
| `--limit` | `-l` | Maksimum kombinasyon | `100000` |
| `--batch-size` | `-b` | GPU batch boyutu | `10000` |
| `--lang` | `-L` | Dil seçeneği | `arabic` |
| `--map-count` | `-m` | Harita sayısı | `5` |
| `--output` | `-o` | Çıktı dizini | Otomatik |
| `--monitor-gpu` | `-g` | GPU izleme | `False` |
| `--no-resume` | - | Checkpoint devam etme | `False` |

## 🌍 Desteklenen Diller

| Dil | Kod | Tablolar | Özellikler |
|-----|-----|----------|------------|
| Arapça | `arabic` | 1, 7, 12, 17, 22, 27, 32 | Şedde desteği |
| İbranice | `hebrew` | 1 | Temel ebced |
| Türkçe | `turkish` | 1 | Türk alfabesi |
| İngilizce | `english` | 1 | Latin alfabesi |
| Latince | `latin` | 1 | Klasik Latin |

## 🔧 Performans Optimizasyonu

### GPU Ayarları
```bash
# Yüksek bellekli GPU'lar için
python checksquares.py phrases.txt --batch-size 100000 --size 4

# Düşük bellekli sistemler için
python checksquares.py phrases.txt --batch-size 5000 --size 3

# GPU izleme ile
python checksquares.py phrases.txt --monitor-gpu --batch-size 20000
```

### CPU Optimizasyonu
```bash
# Paralel işleme (joblib gerektirir)
python checksquares.py phrases.txt --size 3 --parallel

# Bellek optimizasyonu
python checksquares.py phrases.txt --size 4 --low-memory
```

## 📊 Çıktı ve Raporlama

### Temel Çıktılar
- `results_{input}_{size}/` - Çözüm dizini
- `solution_XXX.txt` - Detaylı çözüm raporları
- `square_checker.log` - Sistem logları

### Gelişmiş Raporlama (Rich kuruluysa)
```bash
# Zengin terminal çıktısı
python checksquares.py phrases.txt --size 3 --rich-output

# Detaylı istatistikler
python checksquares.py phrases.txt --size 4 --verbose
```

### Performans Metrikleri
```bash
# GPU performans izleme
python checksquares.py phrases.txt --monitor-gpu --size 4

# Sistem kaynak izleme
python checksquares.py phrases.txt --size 3 --system-stats
```

## 🐛 Sorun Giderme

### GPU Sorunları
```bash
# CUDA sürüm kontrolü
nvidia-smi
python -c "import cupy; print(cupy.__version__)"

# GPU bellek kontrolü
python -c "import GPUtil; GPUtil.showUtilization()"
```

### Performans Sorunları
```bash
# Bellek kullanımı
python checksquares.py phrases.txt --batch-size 5000  # Küçült

# CPU kullanımı
python checksquares.py phrases.txt --size 3 --map-count 2  # Sadeleştir
```

### Bağımlılık Sorunları
```bash
# Bağımlılık kontrolü
python -c "import numpy, scipy, tqdm; print('Temel bağımlılıklar tamam')"

# GPU bağımlılıkları
python -c "import cupy; print('GPU desteği aktif')"
```

## 🚀 Hızlı Başlangıç

### 1. Hızlı Test
```bash
# Minimum bağımlılıklarla
pip install numpy scipy tqdm colorama
python checksquares.py phrases.txt --size 3 --map-count 2 --limit 5000
```

### 2. GPU ile Test
```bash
# GPU desteği ile
pip install cupy-cuda11x GPUtil psutil
python checksquares.py phrases.txt --size 4 --map-count 5 --monitor-gpu
```

### 3. Tam Özellikli Kullanım
```bash
# Tüm özelliklerle
pip install -r requirements.txt
python checksquares.py phrases.txt --size 4 --map-count 8 --tables 1 7 12 --rich-output --monitor-gpu
```

## 📞 Destek

Sorunlar için:
1. Sistem bilgilerinizi paylaşın
2. Kullanılan parametreleri belirtin
3. Hata loglarını ekleyin
4. GPU durumunu kontrol edin

```bash
# Sistem bilgisi
python --version
pip list | grep -E "(numpy|cupy|tqdm)"
nvidia-smi  # GPU varsa
```

---

**Not**: GPU desteği opsiyonel olup yüksek performans sağlar. CPU modu da tüm temel özelliklerle çalışır.
