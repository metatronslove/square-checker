#!/bin/bash

# GPU Optimizasyonlu Sihirli Kare Çözücü Kurulum Scripti
# Conda ai-env ortamında çalıştırılacak

set -e  # Hata durumunda dur

echo "=================================================="
echo "🚀 GPU Optimizasyonlu Sihirli Kare Çözücü Kurulumu"
echo "=================================================="

# Conda ortamını aktif et
echo "🔧 Conda ortamı kontrol ediliyor..."
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "ai-env" ]]; then
    echo "❌ Lütfen önce conda ai-env ortamını aktif edin:"
    echo "   conda activate ai-env"
    exit 1
fi

echo "✅ Conda ai-env ortamı aktif: $CONDA_DEFAULT_ENV"

# Conda kanallarını ekle
echo "📦 Conda kanalları ekleniyor..."
conda config --add channels conda-forge
conda config --add channels nvidia

# Temel paketleri conda ile kur
echo "📦 Temel paketler conda ile kuruluyor..."
conda install -y numpy scipy pandas matplotlib tqdm psutil joblib python-dateutil typing-extensions pathlib2

# GPU paketlerini conda ile kur
echo "🚀 GPU paketleri conda ile kuruluyor..."
conda install -y cudatoolkit  # CUDA Toolkit

# CuPy'yi conda-forge'dan kur
conda install -y -c conda-forge cupy

# PyCUDA'yı conda-forge'dan kur
conda install -y -c conda-forge pycuda

# Geliştirme araçları
echo "🛠️ Geliştirme araçları kuruluyor..."
conda install -y pytest black flake8

# Pip ile kurulması gereken paketler
echo "📚 Pip ile kurulacak paketler..."
pip install loguru rich gputil

# Proje dosyalarını kontrol et
echo "📁 Proje dosyaları kontrol ediliyor..."

REQUIRED_FILES=("Abjad.py" "MagicSquare.py" "checksquares.py")

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file bulundu"
    else
        echo "❌ $file bulunamadı!"
        echo "📥 Lütfen gerekli dosyaları indirin:"
        echo "   - Abjad.py"
        echo "   - MagicSquare.py"
        echo "   - checksquares.py"
        exit 1
    fi
done

# Test çalıştırması
echo "🧪 Test çalıştırması yapılıyor..."

python -c "
try:
    from Abjad import Abjad
    print('✅ Abjad.py başarıyla yüklendi')
except Exception as e:
    print(f'❌ Abjad.py yüklenemedi: {e}')

try:
    from MagicSquare import MagicSquareGenerator
    print('✅ MagicSquare.py başarıyla yüklendi')
except Exception as e:
    print(f'❌ MagicSquare.py yüklenemedi: {e}')

try:
    import cupy as cp
    print('✅ CuPy başarıyla yüklendi')
    if cp.cuda.is_available():
        print('🚀 CUDA GPU desteği aktif')
        print(f'   CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}')
    else:
        print('⚠️ CuPy kurulu ama CUDA kullanılamıyor')
except ImportError as e:
    print(f'⚠️ CuPy kurulu değil: {e}')

try:
    import pycuda.driver as cuda
    cuda.init()
    print(f'✅ PyCUDA başarıyla yüklendi - {cuda.Device.count()} GPU bulundu')
except ImportError as e:
    print(f'⚠️ PyCUDA kurulu değil: {e}')

print('✅ Tüm testler tamamlandı')
"

# Örnek kullanım bilgisi
echo ""
echo "=================================================="
echo "🎉 KURULUM TAMAMLANDI!"
echo "=================================================="
echo ""
echo "🚀 KULLANIM ÖRNEKLERİ:"
echo ""
echo "1. Temel kullanım:"
echo "   python checksquares.py input.txt"
echo ""
echo "2. 4x4 kare ile:"
echo "   python checksquares.py input.txt --size 4"
echo ""
echo "3. GPU batch boyutu ile:"
echo "   python checksquares.py input.txt --batch-size 50000"
echo ""
echo "🔧 Sistem bilgisi:"
python -c "
import sys, numpy, scipy
print(f'Python: {sys.version}')
print(f'NumPy: {numpy.__version__}')
print(f'SciPy: {scipy.__version__}')
try:
    import cupy
    print(f'CuPy: {cupy.__version__}')
except:
    print('CuPy: Kurulu değil')
try:
    import pycuda
    print('PyCUDA: Kurulu')
except:
    print('PyCUDA: Kurulu değil')
"

echo "=================================================="
