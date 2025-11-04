#!/usr/bin/env python3
"""
SİHİRLİ KARE ÇÖZÜCÜ - DÜZELTİLMİŞ TAM VERSİYON
"""

import numpy as np
import math
import time
import json
import os
import sys
import pickle
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Abjad sınıfını import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from Abjad import Abjad
    ABJAD_AVAILABLE = True
    print("✅ Abjad.py sınıfı başarıyla yüklendi")
except ImportError as e:
    ABJAD_AVAILABLE = False
    print(f"❌ Abjad.py yüklenemedi: {e}")
    sys.exit(1)

# MagicSquareGenerator'ı import et
try:
    from MagicSquare import MagicSquareGenerator
    MAGIC_SQUARE_AVAILABLE = True
    print("✅ MagicSquare.py sınıfı başarıyla yüklendi")
except ImportError as e:
    MAGIC_SQUARE_AVAILABLE = False
    print(f"⚠️ MagicSquare.py yüklenemedi: {e}")

# =============================================================================
# BASİT GPU KONTROLÜ
# =============================================================================

GPU_AVAILABLE = False
cp = None
GPU_DEVICES = []

try:
    import cupy as cp
    print("✅ CuPy başarıyla import edildi")

    try:
        # Basit GPU testi
        test_array = cp.array([1, 2, 3])
        result = test_array * 2
        cp.cuda.Stream.null.synchronize()

        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ {device_count} adet CUDA cihazı tespit edildi")

        for i in range(device_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                device_name = props['name'].decode('utf-8')
                total_memory = props['totalGlobalMem'] / (1024**3)
                print(f"   📍 GPU {i}: {device_name} - {total_memory:.1f} GB")
                GPU_DEVICES.append(i)
            except Exception as e:
                print(f"   ⚠️ GPU {i} bilgisi alınamadı: {e}")
                GPU_DEVICES.append(i)

        if GPU_DEVICES:
            GPU_AVAILABLE = True
            cp.cuda.Device(GPU_DEVICES[0]).use()
            print(f"🚀 GPU modu AKTİF - {len(GPU_DEVICES)} GPU kullanılıyor")
        else:
            print("❌ Hiçbir CUDA cihazı kullanılamıyor")
            cp = None
            GPU_AVAILABLE = False

    except Exception as e:
        print(f"❌ CUDA testi başarısız: {e}")
        cp = None
        GPU_AVAILABLE = False

except ImportError as e:
    print(f"❌ CuPy import edilemedi: {e}")
    cp = None
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    import numpy
    cp = numpy
    print("⚠️ CuPy kullanılamıyor, NumPy ile CPU modunda çalışılıyor")

# =============================================================================
# VERİ YAPILARI
# =============================================================================

@dataclass
class EbcedResult:
    phrase: str
    value: int
    table_code: int
    language: str

@dataclass
class Checkpoint:
    processed_combinations: int
    current_table_code: int
    current_language: str
    current_batch_index: int
    found_solutions: List[Dict[str, Any]]
    value_groups: Dict[int, List[str]]
    unique_values: List[int]
    start_time: float
    input_file: str
    square_size: int
    magic_square_map: List[int]

@dataclass
class MagicSquareSolution:
    search_id: int
    table_code: int
    language: str
    magic_constant: int
    combination_index: int
    square_values: List[List[int]]
    square_phrases: List[List[str]]
    alternative_phrases: Dict[str, List[str]]
    timestamp: str
    input_file: str
    map_square: List[List[int]]

# =============================================================================
# KONFİGÜRASYON
# =============================================================================

SUPPORTED_LANGUAGES = {
    "arabic": {
        "name": "Arapça",
        "tables": [1, 7, 12, 17, 22, 27, 32],
        "default_table": 1
    },
    "hebrew": {
        "name": "İbranice",
        "tables": [1],
        "default_table": 1
    },
    "turkish": {
        "name": "Türkçe",
        "tables": [1],
        "default_table": 1
    },
    "english": {
        "name": "İngilizce",
        "tables": [1],
        "default_table": 1
    },
    "latin": {
        "name": "Latince",
        "tables": [1],
        "default_table": 1
    }
}

# =============================================================================
# GPU VALIDATOR
# =============================================================================

class GPUMagicSquareValidator:
    def __init__(self, square_size: int, batch_size: int = 10000):
        self.square_size = square_size
        self.batch_size = batch_size
        self.n = square_size
        self.gpu_available = GPU_AVAILABLE

        if self.gpu_available:
            try:
                self._compile_gpu_kernels()
            except Exception as e:
                print(f"❌ GPU kernel derleme hatası: {e}. CPU moduna geçiliyor.")
                self.gpu_available = False

    def _compile_gpu_kernels(self):
        global cp
        if not self.gpu_available:
            return

        self.validate_magic_squares_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void validate_magic_squares(
            const long long* combinations,
            const int n,
            const int batch_size,
            const int combo_length,
            bool* results,
            long long* magic_constants
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size) return;

            const long long* combo = &combinations[idx * combo_length];

            long long row_sums[12];
            for (int i = 0; i < n; i++) {
                row_sums[i] = 0;
                for (int j = 0; j < n; j++) {
                    row_sums[i] += combo[i * n + j];
                }
            }

            long long col_sums[12];
            for (int j = 0; j < n; j++) {
                col_sums[j] = 0;
                for (int i = 0; i < n; i++) {
                    col_sums[j] += combo[i * n + j];
                }
            }

            long long diag1 = 0, diag2 = 0;
            for (int i = 0; i < n; i++) {
                diag1 += combo[i * n + i];
                diag2 += combo[i * n + (n - 1 - i)];
            }

            bool is_magic = true;
            long long target = row_sums[0];

            for (int i = 1; i < n; i++) {
                if (row_sums[i] != target) {
                    is_magic = false;
                    break;
                }
            }

            if (is_magic) {
                for (int j = 0; j < n; j++) {
                    if (col_sums[j] != target) {
                        is_magic = false;
                        break;
                    }
                }
            }

            if (is_magic && (diag1 != target || diag2 != target)) {
                is_magic = false;
            }

            results[idx] = is_magic;
            magic_constants[idx] = is_magic ? target : -1;
        }
        ''', 'validate_magic_squares')

    def validate_batch_gpu(self, combinations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = combinations.shape[0]
        n_squared = self.n * self.n

        if self.gpu_available:
            try:
                combinations_gpu = cp.asarray(combinations, dtype=cp.int64)
                results_gpu = cp.zeros(batch_size, dtype=cp.bool_)
                constants_gpu = cp.full(batch_size, -1, dtype=cp.int64)

                threads_per_block = 256
                blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block

                self.validate_magic_squares_kernel(
                    (blocks_per_grid,), (threads_per_block,),
                    (combinations_gpu, self.n, batch_size, n_squared, results_gpu, constants_gpu)
                )

                results = cp.asnumpy(results_gpu)
                constants = cp.asnumpy(constants_gpu)

                del combinations_gpu, results_gpu, constants_gpu
                if hasattr(cp, 'get_default_memory_pool'):
                    cp.get_default_memory_pool().free_all_blocks()

                return results, constants

            except Exception as e:
                print(f"❌ GPU doğrulama hatası: {e}")
                self.gpu_available = False
                return self._validate_batch_cpu(combinations)
        else:
            return self._validate_batch_cpu(combinations)

    def _validate_batch_cpu(self, combinations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = combinations.shape[0]
        results = np.zeros(batch_size, dtype=bool)
        constants = np.full(batch_size, -1, dtype=np.int64)

        for i in range(batch_size):
            square = combinations[i].reshape(self.n, self.n)
            if self._validate_single_cpu(square):
                results[i] = True
                constants[i] = np.sum(square[0])

        return results, constants

    def _validate_single_cpu(self, square: np.ndarray) -> bool:
        n = square.shape[0]
        target = np.sum(square[0])

        row_sums = np.sum(square, axis=1)
        if not np.all(row_sums == target):
            return False

        col_sums = np.sum(square, axis=0)
        if not np.all(col_sums == target):
            return False

        diag1 = np.trace(square)
        diag2 = np.trace(np.fliplr(square))

        return diag1 == target and diag2 == target

# =============================================================================
# KOMBİNASYON ÜRETİCİ
# =============================================================================

class GPUCombinationGenerator:
    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size

    def generate_combination_batches(self, unique_values: np.ndarray, n_squared: int,
                                total_combinations: int, start_index: int = 0):
        from itertools import combinations, islice

        batch_size = min(self.batch_size, 10000)

        for batch_start in range(start_index, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)
            current_batch_size = batch_end - batch_start

            if current_batch_size <= 0:
                break

            batch = np.zeros((current_batch_size, n_squared), dtype=np.int64)
            combo_iter = combinations(range(len(unique_values)), n_squared)

            if batch_start > 0:
                combo_iter = islice(combo_iter, batch_start, None)

            for i in range(current_batch_size):
                try:
                    indices = next(combo_iter)
                    batch[i] = unique_values[list(indices)]
                except StopIteration:
                    batch = batch[:i]
                    break

            if len(batch) > 0:
                yield batch, batch_start, batch_start + len(batch)

# =============================================================================
# KUTULU KARE FORMATLAMA
# =============================================================================

class BoxedSquareFormatter:
    def __init__(self):
        pass

    def box_the_square(self, matrix, border_style=2, cell_width=6):
        box = [
            ["─", "│", "┌", "┐", "└", "┘", "├", "┼", "┤", "┬", "┴"],
            ["┄", "┆", "┌", "┐", "└", "┘", "├", "┼", "┤", "┬", "┴"],
            ["┅", "┇", "┏", "┓", "┗", "┛", "┣", "╋", "┫", "┳", "┻"],
        ]
        border_style = max(0, min(border_style, len(box) - 1))

        if matrix is None:
            return "Boş matris"

        if hasattr(matrix, 'tolist'):
            matrix = matrix.tolist()

        if not matrix or len(matrix) == 0:
            return "Boş matris"

        n = len(matrix)
        longest_length = 0

        for row in matrix:
            for cell in row:
                value = str(cell) if cell is not None else "?"
                length = len(value)
                longest_length = max(longest_length, length)

        cell_width = max(cell_width, longest_length)
        border_length = cell_width
        boxed = []

        for r in range(n):
            if r == 0:
                line = (box[border_style][2] +
                        ''.join(box[border_style][0] * border_length + box[border_style][9] for _ in range(n - 1)) +
                        box[border_style][0] * border_length + box[border_style][3])
                boxed.append(line)

            line = box[border_style][1]
            for c in range(n):
                cell_value = matrix[r][c]
                display_value = str(cell_value) if cell_value is not None else "?"
                value_length = len(display_value)
                padding = border_length - value_length
                left_pad = padding // 2
                right_pad = padding - left_pad
                line += " " * left_pad + display_value + " " * right_pad + box[border_style][1]
            boxed.append(line)

            if r < n - 1:
                line = (box[border_style][6] +
                        ''.join(box[border_style][0] * border_length + box[border_style][7] for _ in range(n - 1)) +
                        box[border_style][0] * border_length + box[border_style][8])
                boxed.append(line)
            else:
                line = (box[border_style][4] +
                        ''.join(box[border_style][0] * border_length + box[border_style][10] for _ in range(n - 1)) +
                        box[border_style][0] * border_length + box[border_style][5])
                boxed.append(line)

        return "\n".join(boxed)

# =============================================================================
# İLERLEME GÖRSELLEŞTİRME
# =============================================================================

def format_seconds_to_eta(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds == float('inf'):
        return "∞"

    parts = []
    total_seconds = int(seconds)

    intervals = [
        (365 * 24 * 3600, " yıl"),
        (30 * 24 * 3600, " ay"),
        (24 * 3600, " gün"),
        (3600, " saat"),
        (60, " dakika"),
        (1, " saniye"),
    ]

    for interval_seconds, label in intervals:
        count = total_seconds // interval_seconds
        if count > 0 or len(parts) > 0:
            parts.append(f"{count}{label}")
            total_seconds %= interval_seconds
            if len(parts) >= 3:
                break

    if not parts:
        return "0sn"

    return " ".join(parts)

class ProgressVisualizer:
    def __init__(self, total_combinations: int, square_size: int):
        self.total = total_combinations
        self.current = 0
        self.start_time = time.time()
        self.found_squares = 0
        self.square_size = square_size
        self.formatter = BoxedSquareFormatter()
        self.current_magic_map = None

    def update(self, current: int, found_squares: int = None, current_square=None, magic_map=None, value_combo=None):
        self.current = current
        if found_squares is not None:
            self.found_squares = found_squares

        if magic_map is not None:
            self.current_magic_map = magic_map

        sys.stdout.write('\033[2J\033[H')
        print("🔍 SİHİRLİ KARE ARAŞTIRMASI")
        print("═" * 60)
        print(f"📋 AMAÇ: MagicSquareGenerator ile üretilen sihirli kareyi HARİTA olarak kullanıp")
        print(f"         listedeki ifadelerin ebced değerlerini harita sırasına göre yerleştirip")
        print(f"         sihirli kare olup olmadığını test etmek")
        print("═" * 60)

        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '▒' * (bar_length - filled_length)

        elapsed_time = time.time() - self.start_time

        if self.current > 0 and elapsed_time > 0:
            items_per_second = self.current / elapsed_time
            remaining_time = (self.total - self.current) / items_per_second if items_per_second > 0 else 0
            eta = format_seconds_to_eta(remaining_time)
            speed_info = f"│ Hız: {items_per_second:.0f} kombinasyon/sn"
        else:
            eta = "Hesaplanıyor..."
            speed_info = ""

        print(f"│ Test: {self.current:,}/{self.total:,} │ Bulunan: {self.found_squares} │ Kalan: {eta} {speed_info}")
        print(f"│{bar}│ {percentage:5.1f}%")

        global GPU_AVAILABLE, GPU_DEVICES
        if GPU_AVAILABLE:
            print(f"│ 🚀 GPU MODU: AKTİF ({len(GPU_DEVICES)} GPU - Yüksek Hız)")
        else:
            print("│ ⚠️ GPU MODU: PASİF (NumPy/CPU)")
        print("═" * 60)

        # HARİTAYI GÖSTER
        if self.current_magic_map is not None:
            print("🗺️  KULLANILAN HARİTA (Sıralama referansı):")
            try:
                map_boxed = self.formatter.box_the_square(self.current_magic_map, border_style=1, cell_width=6)
                print(map_boxed)

                # Harita değerlerini sıralı göster
                map_flat = np.array(self.current_magic_map).flatten()
                sorted_indices = sorted(range(len(map_flat)), key=lambda i: map_flat[i])
                sorted_values = [map_flat[i] for i in sorted_indices]
                # print(f"📊 Harita sıralaması (küçükten büyüğe): {sorted_values}")

            except Exception as e:
                print(f"⚠️ Harita gösterim hatası: {e}")

        # TEST EDİLEN DEĞER DAĞILIMINI GÖSTER - HARITA SIRALAMASIYLA UYUMLU
        if current_square is not None and value_combo is not None and self.current_magic_map is not None:
            print("\n🔄 Şu an test edilen EBCED DEĞERLERİ (Harita sıralamasına göre):")
            try:
                # Mevcut kombinasyonu harita sıralamasına göre düzenle
                map_flat = np.array(self.current_magic_map).flatten()
                sorted_positions = sorted(range(len(map_flat)), key=lambda i: map_flat[i])
                sorted_ebced = sorted(value_combo)

                # Harita sıralamasına göre kare oluştur
                display_square = np.zeros((self.square_size, self.square_size), dtype=int)
                for pos_idx, original_pos in enumerate(sorted_positions):
                    if pos_idx < len(sorted_ebced):
                        row = original_pos // self.square_size
                        col = original_pos % self.square_size
                        display_square[row, col] = sorted_ebced[pos_idx]

                boxed = self.formatter.box_the_square(display_square.tolist(), border_style=1, cell_width=8)
                print(boxed)

                # Sıralı değerleri göster
                # print(f"📈 Ebced değerleri (küçükten büyüğe): {sorted_ebced}")

            except Exception as e:
                print(f"⚠️ Değer gösterim hatası: {e}")
                # Fallback: orijinal kareyi göster
                try:
                    if hasattr(current_square, 'tolist'):
                        current_square = current_square.tolist()
                    boxed = self.formatter.box_the_square(current_square, border_style=1, cell_width=8)
                    print(boxed)
                except:
                    print("❌ Görselleştirme başarısız")

        print("═" * 60)
        sys.stdout.flush()

    def show_solution(self, solution):
        print("\n🎉 YENİ SİHİRLİ KARE BULUNDU!")
        print("═" * 60)
        print(f"🔢 Çözüm #{solution.search_id} | Dil: {solution.language} | Tablo: {solution.table_code}")
        print(f"📐 Sihirli Sabit: {solution.magic_constant}")
        print("═" * 60)

        print("🗺️  KULLANILAN HARİTA (MagicSquareGenerator ile üretildi):")
        map_boxed = self.formatter.box_the_square(solution.map_square, border_style=2, cell_width=6)
        print(map_boxed)

        print("\n🔢 OLUŞAN SAYISAL KARE:")
        values_boxed = self.formatter.box_the_square(solution.square_values, border_style=2, cell_width=8)
        print(values_boxed)

        print("\n📝 OLUŞAN İFADELER KARESİ:")
        phrases_boxed = self.formatter.box_the_square(solution.square_phrases, border_style=1, cell_width=15)
        print(phrases_boxed)

        if solution.alternative_phrases:
            print("\n🔄 EŞ EBCED DEĞERLİ ALTERNATİF İFADELER:")
            for pos_key, alternatives in solution.alternative_phrases.items():
                row, col = map(int, pos_key.split('_'))
                main_phrase = solution.square_phrases[row][col]
                main_value = solution.square_values[row][col]
                print(f"  📍 Pozisyon ({row+1},{col+1}): {main_phrase} = {main_value}")
                if alternatives:
                    print(f"     🔄 Alternatifler: {', '.join(alternatives[:3])}")

        print("⏳ 5 saniye sonra devam ediliyor...")
        time.sleep(5)

    def complete(self):
        print(f"\n✅ TAMAMLANDI! Toplam {self.found_squares} sihirli kare bulundu.")

# =============================================================================
# ANA ÇÖZÜCÜ - DÜZELTİLMİŞ
# =============================================================================

class GPUOptimizedSquareChecker:
    def __init__(self, log_file: str = "square_checker.log", gpu_batch_size: int = 10000):
        self.abjad_calculator = Abjad()
        self.log_file = log_file
        self.checkpoint_file = "checkpoint.pkl"
        self.solution_counter = 0
        self.formatter = BoxedSquareFormatter()
        self.gpu_batch_size = gpu_batch_size
        self._should_save_checkpoint = True

        self.gpu_available = GPU_AVAILABLE
        self.gpu_devices = GPU_DEVICES

        print(f"🎯 Çözücü Başlatılıyor - GPU: {self.gpu_available}")

        if self.gpu_available:
            self.log(f"🚀 GPU modu ETKİN - {len(self.gpu_devices)} GPU kullanılıyor")
        else:
            self.log("⚠️ GPU modu PASİF, CPU kullanılıyor")

        if MAGIC_SQUARE_AVAILABLE:
            self.magic_generator = MagicSquareGenerator()
            print("✅ MagicSquareGenerator başlatıldı")
        else:
            self.magic_generator = None
            print("❌ MagicSquareGenerator kullanılamıyor!")

        self._setup_logging()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def signal_handler(sig, frame):
            print(f"\n⏸️ İşlem kesildi! Checkpoint kaydediliyor...")
            self._should_save_checkpoint = False
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    def _setup_logging(self):
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Oturum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def save_checkpoint(self, checkpoint: Checkpoint, output_dir: str):
        if not self._should_save_checkpoint:
            return
        try:
            checkpoint_path = os.path.join(output_dir, self.checkpoint_file)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            self.log(f"Checkpoint kaydedildi: {checkpoint_path}")
        except Exception as e:
            self.log(f"Checkpoint kaydetme hatası: {str(e)}", "WARNING")

    def load_checkpoint(self, output_dir: str) -> Optional[Checkpoint]:
        checkpoint_path = os.path.join(output_dir, self.checkpoint_file)
        if not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.log(f"Checkpoint yüklendi: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.log(f"Checkpoint yükleme hatası: {str(e)}", "WARNING")
            return None

    def cleanup_checkpoint(self, output_dir: str):
        checkpoint_path = os.path.join(output_dir, self.checkpoint_file)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            self.log(f"Checkpoint temizlendi: {checkpoint_path}")

    def validate_language_tables(self, language: str, tables: List[int]) -> Tuple[bool, List[int]]:
        if language not in SUPPORTED_LANGUAGES:
            return False, []

        supported_tables = SUPPORTED_LANGUAGES[language]["tables"]
        valid_tables = [t for t in tables if t in supported_tables]

        if not valid_tables:
            default_table = SUPPORTED_LANGUAGES[language]["default_table"]
            valid_tables = [default_table]
            self.log(f"⚠️ {language} için geçersiz tablolar, varsayılan tablo {default_table} kullanılıyor", "WARNING")

        return True, valid_tables

    def abjad(self, text: str, table_code: int = 1, shadda: int = 1, lang: str = "arabic") -> int:
        try:
            result = self.abjad_calculator.abjad(text, table_code, shadda, 0, lang)
            if isinstance(result, dict):
                return result["sum"]
            return result
        except Exception as e:
            self.log(f"Ebced hesaplama hatası: '{text}', Hata: {str(e)}", "WARNING")
            return 0

    def generate_unique_magic_square_maps(self, size: int, count: int = 5) -> List[np.ndarray]:
        if not MAGIC_SQUARE_AVAILABLE:
            self.log("❌ MagicSquareGenerator kullanılamıyor!", "ERROR")
            return []

        try:
            raw_squares = self.magic_generator.generate_multiple_squares(size, count * 3, variations=False)
            unique_squares = self._filter_unique_squares(raw_squares)
            final_squares = unique_squares[:count]

            self.log(f"✅ {len(final_squares)} adet benzersiz {size}x{size} sihirli kare haritası üretildi")
            return final_squares
        except Exception as e:
            self.log(f"❌ Sihirli kare üretme hatası: {str(e)}", "ERROR")
            return []

    def _filter_unique_squares(self, squares: List[np.ndarray]) -> List[np.ndarray]:
        unique_squares = []
        seen_patterns = set()

        for square in squares:
            normalized = self._get_canonical_form(square)
            pattern_str = ','.join(map(str, normalized.flatten()))

            if pattern_str not in seen_patterns:
                seen_patterns.add(pattern_str)
                unique_squares.append(square)

        return unique_squares

    def _get_canonical_form(self, square: np.ndarray) -> np.ndarray:
        n = square.shape[0]
        variations = []

        for rotation in range(4):
            rotated = np.rot90(square, rotation)
            variations.append(rotated)
            variations.append(np.fliplr(rotated))
            variations.append(np.flipud(rotated))

        min_variation = None
        for var in variations:
            flat = var.flatten()
            if min_variation is None or tuple(flat) < tuple(min_variation.flatten()):
                min_variation = var

        return min_variation

    def parse_input_file(self, file_path: str) -> Dict[str, Any]:
        self.log(f"Girdi dosyası okunuyor: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if 'phrases = [' in content:
            phrases_section = content.split('phrases = [')[1].rstrip(']').strip()
            phrases = [p.strip() for p in phrases_section.split('\n') if p.strip()]
        else:
            phrases = [p.strip() for p in content.split('\n') if p.strip()]

        return {'phrases': phrases}

    def calculate_ebced_values(self, phrases: List[str], table_codes: List[int], shadda: int = 1, lang: str = "arabic") -> List[EbcedResult]:
        results = []
        for phrase in phrases:
            for table_code in table_codes:
                value = self.abjad(phrase, table_code, shadda, lang)
                results.append(EbcedResult(phrase, value, table_code, lang))
        return results

    def _get_current_square_for_display(self, value_combo: np.ndarray, magic_square_map: np.ndarray, n: int):
        """Görselleştirme için HARITA SIRALAMASIYLA UYUMLU kare oluştur"""

        # Aynı mantıkla harita sıralamasına göre yerleştirme
        map_flat = magic_square_map.flatten()
        sorted_positions = sorted(range(len(map_flat)), key=lambda i: map_flat[i])
        sorted_ebced = sorted(value_combo)

        position_to_value = {}
        for pos_idx, original_pos in enumerate(sorted_positions):
            if pos_idx < len(sorted_ebced):
                position_to_value[original_pos] = sorted_ebced[pos_idx]

        # Görselleştirme karesi oluştur
        display_square = np.zeros((n, n), dtype=int)

        for original_idx, ebced_value in position_to_value.items():
            row = original_idx // n
            col = original_idx % n
            display_square[row, col] = ebced_value

        return display_square.tolist()

    def create_magic_square_solution(self, value_combo: np.ndarray, value_groups: Dict[int, List[str]],
                                   magic_square_map: np.ndarray, table_code: int, language: str,
                                   combination_index: int, n: int, input_file: str) -> MagicSquareSolution:
        """Sihirli kare çözümü oluştur - HARITA SIRALAMASIYLA UYUMLU"""

        # 1. Haritayı düzleştir ve sırala
        map_flat = magic_square_map.flatten()

        # 2. DOĞRU SIRALAMA: Harita pozisyonlarını DEĞERLERİNE göre sırala
        sorted_positions = sorted(range(len(map_flat)), key=lambda i: map_flat[i])

        # 3. Ebced değerlerini küçükten büyüğe sırala
        sorted_ebced = sorted(value_combo)

        # 4. DOĞRU EŞLEME: En küçük harita değeri -> en küçük ebced değeri
        position_to_value = {}
        for pos_idx, original_pos in enumerate(sorted_positions):
            if pos_idx < len(sorted_ebced):
                position_to_value[original_pos] = sorted_ebced[pos_idx]

        # 5. Kareleri oluştur - HARITA SIRALAMASINA GÖRE
        values_square = np.zeros((n, n), dtype=int)
        phrases_square = np.empty((n, n), dtype=object)
        alternative_phrases = {}

        # Değer gruplarının kopyasını al
        remaining_groups = {k: v.copy() for k, v in value_groups.items()}

        # 6. ORİJİNAL HARITA POZİSYONLARINA değerleri yerleştir
        for original_idx, ebced_value in position_to_value.items():
            row = original_idx // n
            col = original_idx % n

            if ebced_value in remaining_groups and remaining_groups[ebced_value]:
                main_phrase = remaining_groups[ebced_value].pop(0)
                values_square[row, col] = ebced_value
                phrases_square[row, col] = main_phrase

                # Alternatif ifadeleri kaydet
                if len(remaining_groups[ebced_value]) > 0:
                    alt_key = f"{row}_{col}"
                    alternative_phrases[alt_key] = remaining_groups[ebced_value].copy()

        magic_constant = int(np.sum(values_square[0]))

        self.solution_counter += 1

        return MagicSquareSolution(
            search_id=self.solution_counter,
            table_code=table_code,
            language=language,
            magic_constant=magic_constant,
            combination_index=combination_index,
            square_values=values_square.tolist(),
            square_phrases=phrases_square.tolist(),
            alternative_phrases=alternative_phrases,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            input_file=input_file,
            map_square=magic_square_map.tolist()
        )

    def save_solution_to_file(self, solution: MagicSquareSolution, output_dir: str):
        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M')
        filename = f"solution_{solution.search_id:03d}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SİHİRLİ KARE ÇÖZÜMÜ #{solution.search_id}\n")
            f.write("=" * 50 + "\n")
            f.write(f"📅 Oluşturulma: {solution.timestamp}\n")
            f.write(f"🌍 Dil: {solution.language}\n")
            f.write(f"📊 Tablo: {solution.table_code}\n")
            f.write(f"📐 Sihirli Sabit: {solution.magic_constant}\n")
            f.write(f"🔢 Kombinasyon No: {solution.combination_index}\n")

            global GPU_AVAILABLE, GPU_DEVICES
            f.write(f"⚡ GPU: {'EVET' if GPU_AVAILABLE else 'HAYIR'}\n")
            if GPU_AVAILABLE:
                f.write(f"🎯 GPU Sayısı: {len(GPU_DEVICES)}\n")
            f.write(f"🗺️  Harita: MagicSquareGenerator ile üretildi\n")
            f.write("\n")

            f.write("🗺️  KULLANILAN HARİTA:\n")
            f.write("-" * 30 + "\n")
            map_boxed = self.formatter.box_the_square(solution.map_square, border_style=2, cell_width=6)
            f.write(map_boxed)
            f.write("\n\n")

            f.write("🔢 OLUŞAN SAYISAL SİHİRLİ KARE:\n")
            f.write("-" * 30 + "\n")
            values_boxed = self.formatter.box_the_square(solution.square_values, border_style=2, cell_width=8)
            f.write(values_boxed)
            f.write("\n\n")

            f.write("📝 OLUŞAN İFADELER SİHİRLİ KARESİ:\n")
            f.write("-" * 30 + "\n")
            phrases_boxed = self.formatter.box_the_square(solution.square_phrases, border_style=1, cell_width=20)
            f.write(phrases_boxed)
            f.write("\n\n")

            if solution.alternative_phrases:
                f.write("🔄 EŞ EBCED DEĞERLİ ALTERNATİF İFADELER:\n")
                f.write("-" * 40 + "\n")
                for pos_key, alternatives in solution.alternative_phrases.items():
                    row, col = map(int, pos_key.split('_'))
                    main_phrase = solution.square_phrases[row][col]
                    main_value = solution.square_values[row][col]
                    f.write(f"📍 Pozisyon ({row+1},{col+1}): {main_phrase} = {main_value}\n")
                    if alternatives:
                        f.write(f"   🔄 Alternatifler: {', '.join(alternatives)}\n")
                    f.write("\n")

        self.log(f"✅ Çözüm kaydedildi: {filename}")
        return filepath

    def solve_magic_squares(self, input_file: str, table_codes: List[int] = None,
                          square_size: int = 3, shadda: int = 1, max_combinations: int = 100000,
                          output_dir: str = None, resume: bool = True, language: str = "arabic",
                          map_count: int = 5):
        """Ana çözüm fonksiyonu - DÜZELTİLMİŞ"""

        if not MAGIC_SQUARE_AVAILABLE:
            self.log("❌ MagicSquareGenerator gerekli!", "ERROR")
            return []

        is_valid, valid_tables = self.validate_language_tables(language, table_codes or [])
        if not is_valid:
            self.log(f"❌ Geçersiz dil: {language}", "ERROR")
            return []

        table_codes = valid_tables

        if output_dir is None:
            input_stem = Path(input_file).stem
            output_dir = f"results_{input_stem}_size{square_size}"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            input_data = self.parse_input_file(input_file)
            phrases = input_data['phrases']
            n = square_size
            n_squared = n * n

            self.log(f"🔍 SİHİRLİ KARE ARAŞTIRMASI BAŞLATILDI")
            self.log(f"📐 Kare Boyutu: {n}x{n} | 📝 İfadeler: {len(phrases)} | 🌍 Dil: {language}")
            self.log(f"📊 Tablolar: {table_codes} | 🗺️  Harita Sayısı: {map_count}")
            self.log(f"🎯 GPU Durumu: {'AKTİF' if self.gpu_available else 'PASİF'}")

            magic_square_maps = self.generate_unique_magic_square_maps(n, map_count)
            if not magic_square_maps:
                self.log("❌ Hiç harita üretilemedi!", "ERROR")
                return []

            ebced_results = self.calculate_ebced_values(phrases, table_codes, shadda, language)
            all_solutions = []

            # GPU VALIDATOR VE COMBO GENERATOR TANIMLAMA - DÜZELTME BURADA
            try:
                gpu_validator = GPUMagicSquareValidator(n, self.gpu_batch_size)
                combo_generator = GPUCombinationGenerator(self.gpu_batch_size)  # ✅ TANIMLANDI
                self.log(f"✅ GPU validator başlatıldı - Durum: {'AKTİF' if gpu_validator.gpu_available else 'PASİF'}")
            except Exception as e:
                self.log(f"❌ GPU validator başlatma hatası: {e}, CPU moduna geçiliyor", "WARNING")
                gpu_validator = GPUMagicSquareValidator(n, self.gpu_batch_size)
                gpu_validator.gpu_available = False
                combo_generator = GPUCombinationGenerator(self.gpu_batch_size)  # ✅ TANIMLANDI

            for map_index, magic_square_map in enumerate(magic_square_maps):
                self.log(f"🗺️  Benzersiz Harita #{map_index+1} işleniyor...")

                map_boxed = self.formatter.box_the_square(magic_square_map.tolist(), border_style=1, cell_width=6)
                self.log(f"Kullanılan harita:\n{map_boxed}")

                for table_code in table_codes:
                    self.log(f"🔧 {language} - Tablo {table_code} işleniyor...")

                    table_results = [r for r in ebced_results if r.table_code == table_code]
                    value_groups = {}
                    for result in table_results:
                        if result.value not in value_groups:
                            value_groups[result.value] = []
                        value_groups[result.value].append(result.phrase)

                    unique_values = sorted(value_groups.keys())
                    unique_array = np.array(unique_values, dtype=np.int64)

                    if len(unique_values) < n_squared:
                        self.log(f"⚠️ Yetersiz benzersiz değer: {len(unique_values)} < {n_squared}", "WARNING")
                        continue

                    total_combinations = math.comb(len(unique_values), n_squared)
                    #if max_combinations and total_combinations > max_combinations:
                    #    total_combinations = max_combinations
                    #self.log(f"📈 Toplam kombinasyon: {total_combinations:,}")

                    checkpoint = None
                    start_index = 0
                    found_solutions = []

                    if resume:
                        checkpoint = self.load_checkpoint(output_dir)
                        if (checkpoint and checkpoint.current_table_code == table_code and
                            checkpoint.current_language == language and
                            np.array_equal(checkpoint.magic_square_map, magic_square_map.flatten().tolist())):
                            start_index = checkpoint.current_batch_index
                            found_solutions = checkpoint.found_solutions
                            self.log(f"🔄 Checkpoint'ten devam: {start_index:,}. batch")

                    visual = ProgressVisualizer(total_combinations, n)
                    found_count = len(found_solutions)
                    processed_count = start_index * self.gpu_batch_size

                    # ✅ COMBO_GENERATOR ARTIK TANIMLI - HATA DÜZELDİ
                    for batch, batch_start, batch_end in combo_generator.generate_combination_batches(
                        unique_array, n_squared, total_combinations, start_index):

                        batch_size = batch.shape[0]
                        gpu_time = 0

                        # GPU DOĞRULAMA
                        start_gpu_time = time.time()
                        valid_mask, magic_constants = gpu_validator.validate_batch_gpu(batch)
                        gpu_time = time.time() - start_gpu_time

                        valid_indices = np.where(valid_mask)[0]

                        # HARITA SIRALAMASIYLA UYUMLU GÖRSELLEŞTİRME
                        current_square_display = None
                        current_value_combo = None
                        if batch_size > 0:
                            # İlk kombinasyonu al ve harita sıralamasına göre düzenle
                            current_value_combo = batch[0]
                            current_square_display = self._get_current_square_for_display(current_value_combo, magic_square_map, n)

                        # Güncelle - value_combo parametresini ekle
                        visual.update(processed_count, found_count, current_square_display,
                                     magic_square_map.tolist(), current_value_combo)

                        # GEÇERLİ ÇÖZÜMLERİ İŞLE
                        for idx in valid_indices:
                            combo_index = batch_start + idx
                            value_combo = batch[idx]

                            solution = self.create_magic_square_solution(
                                value_combo, value_groups, magic_square_map,
                                table_code, language, combo_index, n, input_file
                            )

                            found_count += 1
                            found_solutions.append(asdict(solution))
                            all_solutions.append(solution)

                            self.save_solution_to_file(solution, output_dir)
                            self.log(f"✅ Harita#{map_index+1} - {language} - Sihirli kare #{solution.search_id} bulundu!")

                            visual.show_solution(solution)

                        processed_count += batch_size
                        items_per_second = batch_size / gpu_time if gpu_time > 0 else 0

                        # Sonraki iterasyon için görselleştirme
                        current_square_display = None
                        current_value_combo = None
                        if batch_size > 0:
                            current_value_combo = batch[0]
                            current_square_display = self._get_current_square_for_display(current_value_combo, magic_square_map, n)

                        visual.update(processed_count, found_count, current_square_display,
                                     magic_square_map.tolist(), current_value_combo)

                        if gpu_time > 0:
                            self.log(f"⚡ Batch: {batch_size} combo, {gpu_time:.3f}s, {items_per_second:.0f} combo/sn")

                        # CHECKPOINT KAYDET
                        checkpoint = Checkpoint(
                            processed_combinations=processed_count,
                            current_table_code=table_code,
                            current_language=language,
                            current_batch_index=batch_start + 1,
                            found_solutions=found_solutions,
                            value_groups=value_groups,
                            unique_values=unique_values,
                            start_time=start_time,
                            input_file=input_file,
                            square_size=n,
                            magic_square_map=magic_square_map.flatten().tolist()
                        )
                        self.save_checkpoint(checkpoint, output_dir)

                    visual.complete()

            self.cleanup_checkpoint(output_dir)

            execution_time = time.time() - start_time
            self.log(f"✅ TAMAMLANDI: {len(all_solutions)} sihirli kare bulundu")
            self.log(f"⏱️ Toplam süre: {execution_time:.2f}s")
            if execution_time > 0:
                self.log(f"📊 Ortalama hız: {processed_count/execution_time:.0f} kombinasyon/sn")

            return all_solutions

        except Exception as e:
            self.log(f"❌ Çözüm hatası: {str(e)}", "ERROR")
            import traceback
            self.log(f"❌ Traceback: {traceback.format_exc()}", "ERROR")
            raise

# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sihirli Kare Çözücü - MagicSquareGenerator ile harita üretip ebced değerlerini yerleştirir')
    parser.add_argument('input_file', help='Girdi dosyası yolu (ifade listesi içeren)')
    parser.add_argument('--size', '-s', type=int, default=3, choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       help='Sihirli kare boyutu (3-12)')
    parser.add_argument('--tables', '-t', nargs='+', type=int, default=None,
                       help='Ebced tablo kodları')
    parser.add_argument('--shadda', '-S', type=int, default=1, choices=[1, 2],
                       help='Şedde hesabı (Arapça için)')
    parser.add_argument('--limit', '-l', type=int, default=100000,
                       help='Maksimum test kombinasyonu')
    parser.add_argument('--batch-size', '-b', type=int, default=80000,
                       help='GPU batch boyutu')
    parser.add_argument('--lang', '-L', default='arabic',
                       choices=['arabic', 'hebrew', 'turkish', 'english', 'latin'],
                       help='Dil seçeneği')
    parser.add_argument('--map-count', '-m', type=int, default=5,
                       help='Üretilecek sihirli kare harita sayısı')
    parser.add_argument('--output', '-o', help='Çıktı dizini')
    parser.add_argument('--no-resume', action='store_true',
                       help='Checkpoint\'ten devam etme')

    args = parser.parse_args()

    if not ABJAD_AVAILABLE:
        print("❌ Abjad.py sınıfı yüklenemedi!")
        sys.exit(1)

    if not MAGIC_SQUARE_AVAILABLE:
        print("❌ MagicSquare.py sınıfı yüklenemedi!")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"❌ Girdi dosyası bulunamadı: {args.input_file}")
        sys.exit(1)

    if args.tables is None:
        if args.lang in SUPPORTED_LANGUAGES:
             args.tables = [SUPPORTED_LANGUAGES[args.lang]["default_table"]]
        else:
             args.tables = [1]

    print("🚀 SİHİRLİ KARE ÇÖZÜCÜ")
    print("═" * 60)
    print(f"🎯 AMAÇ: MagicSquareGenerator ile {args.map_count} adet {args.size}x{args.size} FARKLI sihirli kare HARİTASI üretip")
    print(f"         bu haritalara göre ebced değerlerini yerleştirip sihirli kare test etmek")
    print("═" * 60)
    print(f"📁 Girdi: {args.input_file}")
    print(f"📐 Boyut: {args.size}x{args.size}")
    print(f"🌍 Dil: {args.lang}")
    print(f"📊 Tablolar: {args.tables}")
    print(f"🔤 Şedde: {args.shadda}")
    print(f"🗺️  Harita Sayısı: {args.map_count}")
    print(f"⏱️ Limit: {args.limit:,} kombinasyon")
    print(f"🎯 Batch: {args.batch_size}")

    global GPU_AVAILABLE, GPU_DEVICES
    if GPU_AVAILABLE:
        print(f"🚀 GPU MODU: AKTİF ({len(GPU_DEVICES)} GPU)")
    else:
        print("⚠️ GPU MODU: PASİF (CPU)")
    print("═" * 60)

    checker = GPUOptimizedSquareChecker(gpu_batch_size=args.batch_size)
    checker.solve_magic_squares(
        input_file=args.input_file,
        square_size=args.size,
        table_codes=args.tables,
        shadda=args.shadda,
        max_combinations=args.limit,
        output_dir=args.output,
        resume=not args.no_resume,
        language=args.lang,
        map_count=args.map_count
    )

if __name__ == "__main__":
    main()
