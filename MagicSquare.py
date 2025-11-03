import math
import numpy as np

class MagicSquareGenerator:
    def __init__(self):
        pass

    def generate_magic_square(self, n, row_sum=None, rotation=0, mirror=False):
        """
        Sihirli kare üretir ve numpy array olarak döndürür

        Args:
            n: Kare boyutu (3x3, 4x4, vb.)
            row_sum: Satır toplamı (opsiyonel)
            rotation: Döndürme açısı (0, 90, 180, 270)
            mirror: Ayna efekti uygula

        Returns:
            numpy.ndarray: Sihirli kare matrisi
        """
        if n < 3:
            raise ValueError("Size must be at least 3")

        magic_constant = (n * (n * n + 1)) / 2
        if row_sum is None:
            row_sum = magic_constant
        if row_sum < magic_constant:
            raise ValueError(f"Row sum cannot be less than the magic constant ({magic_constant})")

        magic_square = self.create_magic_square(n)

        if row_sum > magic_constant:
            if n % 2 == 1:
                magic_square = self.incremented_magic_square(magic_square, row_sum)
            elif n % 4 == 0:
                magic_square = self.increment_matrix(magic_square, row_sum)

        if rotation > 0:
            magic_square = self.rotate_matrix(magic_square, rotation // 90)

        if mirror:
            magic_square = self.mirror_flip(magic_square)

        # Numpy array'e çevir
        magic_array = np.array(magic_square, dtype=np.int64)

        if not self.check_magic_square(magic_array, row_sum):
            # Eğer doğrulama başarısız olursa bir sonraki boyutu dene
            return self.generate_magic_square(n + 1, row_sum, rotation, mirror)

        return magic_array

    def create_magic_square(self, n):
        """Temel sihirli kare oluşturma"""
        if n % 2 == 1:
            return self.siamese_method(n)
        elif n % 4 == 0:
            return self.strachey_method(n)
        else:
            return self.strachey_singly_even_method(n)

    def siamese_method(self, n):
        """Tek boyutlu kareler için Siamese method"""
        magic_square = [[0] * n for _ in range(n)]
        row, col = 0, n // 2

        for num in range(1, n * n + 1):
            magic_square[row][col] = int(num)
            next_row = (row - 1 + n) % n
            next_col = (col + 1) % n

            if magic_square[next_row][next_col] != 0:
                row = (row + 1) % n
            else:
                row, col = next_row, next_col

        return magic_square

    def strachey_method(self, n):
        """4'ün katları için Strachey method"""
        magic_square = [[0] * n for _ in range(n)]
        count = 1

        for i in range(n):
            for j in range(n):
                if i % 4 == j % 4 or (i + j) % 4 == 3:
                    magic_square[i][j] = int(n * n - count + 1)
                else:
                    magic_square[i][j] = int(count)
                count += 1

        return magic_square

    def strachey_singly_even_method(self, n):
        """Tek-çift boyutlar için Strachey method"""
        magic_square = [[0] * n for _ in range(n)]
        k = n // 2
        mini_magic = self.siamese_method(k)

        for i in range(k):
            for j in range(k):
                magic_square[i][j] = int(mini_magic[i][j])
                magic_square[i + k][j + k] = int(mini_magic[i][j] + k * k)
                magic_square[i][j + k] = int(mini_magic[i][j] + 2 * k * k)
                magic_square[i + k][j] = int(mini_magic[i][j] + 3 * k * k)

        swap_col = list(range((k - 1) // 2)) + list(range(n - (k - 1) // 2 + 1, n))

        for i in range(k):
            for col in swap_col:
                magic_square[i][col], magic_square[i + k][col] = (
                    magic_square[i + k][col], magic_square[i][col]
                )

        half_k = k // 2
        magic_square[half_k][0], magic_square[half_k + k][0] = (
            magic_square[half_k + k][0], magic_square[half_k][0]
        )
        magic_square[half_k + k][half_k], magic_square[half_k][half_k] = (
            magic_square[half_k][half_k], magic_square[half_k + k][half_k]
        )

        return magic_square

    def incremented_magic_square(self, magic_square, row_sum):
        """Tek boyutlu kareler için artırılmış sihirli kare"""
        n = len(magic_square)
        magic_constant = (n * (n * n + 1)) / 2
        incremention = (row_sum - magic_constant) / n

        for r in range(n):
            for c in range(n):
                magic_square[r][c] += self.incremention_for_cell(
                    n, row_sum, incremention, magic_square[r][c]
                )

        return magic_square

    def incremention_for_cell(self, n, row_sum, incremention, cell_value):
        """Hücre başına artış miktarını hesapla"""
        magic_constant = (n * (n * n + 1)) / 2
        threshold = n * n - n * (row_sum % n)
        return math.ceil(incremention) if cell_value > threshold else math.floor(incremention)

    def increment_matrix(self, magic_square, row_sum):
        """Matris artırma"""
        n = len(magic_square)
        magic_constant = (n * (n * n + 1)) / 2
        z = (row_sum - magic_constant) % n
        incremention = (row_sum - magic_constant - z) / n

        for k in range(int(z)):
            for i in range(n):
                row = (k + i) % n
                col = i
                magic_square[row][col] += 1

        for r in range(n):
            for c in range(n):
                magic_square[r][c] += incremention

        return magic_square

    def mirror_flip(self, magic_square):
        """Ayna efekti uygula"""
        n = len(magic_square)
        mirror_flipped = [[0] * n for _ in range(n)]

        for a in range(n):
            for b in range(n):
                m = n - 1 - a
                n_idx = n - 1 - b
                mirror_flipped[a][b] = magic_square[m][n_idx]

        return mirror_flipped

    def rotate_matrix(self, matrix, repeat):
        """Matrisi döndür"""
        n = len(matrix)
        rotated = [[matrix[i][j] for j in range(n)] for i in range(n)]

        for _ in range(repeat % 4):  # Normalize rotations
            temp = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    temp[i][j] = rotated[i][j]
            for i in range(n):
                for j in range(n):
                    rotated[j][n - 1 - i] = temp[i][j]

        return rotated

    def check_magic_square(self, magic_square, expected_sum):
        """Sihirli kareyi doğrula"""
        n = len(magic_square)
        expected_sum = int(expected_sum)

        # Satır toplamlarını kontrol et
        for i in range(n):
            row_sum = np.sum(magic_square[i])
            if row_sum != expected_sum:
                return False

        # Sütun toplamlarını kontrol et
        for j in range(n):
            col_sum = np.sum(magic_square[:, j])
            if col_sum != expected_sum:
                return False

        # Köşegen toplamlarını kontrol et
        diag_sum1 = np.trace(magic_square)
        diag_sum2 = np.trace(np.fliplr(magic_square))

        if diag_sum1 != expected_sum or diag_sum2 != expected_sum:
            return False

        return True

    def generate_multiple_squares(self, n, count=10, variations=True):
        """
        Birden fazla sihirli kare üretir

        Args:
            n: Kare boyutu
            count: Üretilecek kare sayısı
            variations: Farklı varyasyonlar üret

        Returns:
            List[numpy.ndarray]: Sihirli kare listesi
        """
        squares = []
        base_square = self.generate_magic_square(n)
        squares.append(base_square)

        if variations and count > 1:
            # Farklı rotasyon ve ayna kombinasyonları
            rotations = [0, 90, 180, 270]
            mirrors = [False, True]

            for rotation in rotations:
                for mirror in mirrors:
                    if len(squares) >= count:
                        break
                    try:
                        variant = self.generate_magic_square(n, rotation=rotation, mirror=mirror)
                        # Benzersiz kareleri kontrol et
                        if not any(np.array_equal(variant, sq) for sq in squares):
                            squares.append(variant)
                    except:
                        continue

        return squares[:count]

    def get_magic_constant(self, n):
        """n boyutlu kare için sihirli sabiti hesapla"""
        return n * (n * n + 1) // 2

    def is_magic_square(self, square):
        """Verilen karenin sihirli kare olup olmadığını kontrol et"""
        if not isinstance(square, np.ndarray):
            square = np.array(square)

        n = square.shape[0]
        magic_constant = self.get_magic_constant(n)

        return self.check_magic_square(square, magic_constant)
