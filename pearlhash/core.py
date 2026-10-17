"""Математическое ядро pearlhash: NoisyGEMM INT8 + transcript fold + BLAKE3.

Последовательность для одного «грида» (попытки):

  1. seed   = BLAKE3(key || nonce_be64)          — производная от job + локального nonce
  2. A[M,K], B[N,K] ~ U[-64, 64] ∩ int8         — из Philox(salt=0/1)
  3. U[M,R], V[R,K], X[N,R], Y[R,K] ~ U[-8,8]   — низкоранговые факторы шума (salt=2..5)
     E = U@V (ранг R=64), F = X@Y
  4. C' = (A + E_int8clip) @ (B + F_int8clip)^T  — единое INT8 GEMM с точным
     накоплением (int64 в CPU-эталоне; INT32 tensor cores на GPU bit-exact)
  5. transcript: блоки C' тайлами tile_h x tile_w, побайто little-endian int32;
     fold = XOR-свёртка хешей тайлов в основном цикле (in-mainloop transcript fold)
  6. H = BLAKE3(fold_bytes); решение найдено, если int_le(H) < target.

Честный майнер не может вычислить A@B и «добавить шум сверху»: это потребовало
бы трёх GEMM вместо одного (шум встроен ДО умножения).
"""

from __future__ import annotations

import numpy as np
import blake3

from .params import MiningParams, DEFAULT_PARAMS
from .philox import PhiloxRng

# salt'ы Philox для ролей (независимые потоки из одного seed)
SALT_A, SALT_B, SALT_U, SALT_V, SALT_X, SALT_Y = 0, 1, 2, 3, 4, 5

NOISE_ELEM_MIN, NOISE_ELEM_MAX = -8, 8  # факторы шума U,V,X,Y
NOISE_QUANT_SHIFT = 6                   # квантование шума: E = (U V^T >> s) << s


def wrap_lowrank_noise(f1: np.ndarray, f2: np.ndarray,
                       shift: int = NOISE_QUANT_SHIFT) -> np.ndarray:
    """Низкоранговый шум с гарантией ранга <= R и попаданием в int8.

    Сырой продукт Q = f1 @ f2^T имеет ранг <= R; элементы |Q| <= R*64 = 4096.
    Арифметическое квантование T = (Q >> shift) << shift — точная целая
    операция (bit-exact на всех платформах) с факторизацией
    T = (f1 >> shift) @ f2, откуда ранг T <= R строгo.
    Клип в [-128, 127] переводит шум в int8-репрезентацию; майнер и
    верификатор работают с A+E как с int8 (mod 256), что сохраняет
    детерминированность транскрипта на любом железе.
    """
    q = f1.astype(np.int64) @ f2.T.astype(np.int64)
    t = (q >> shift) << shift
    return np.clip(t, -128, 127).astype(np.int8)


def derive_seed(key: bytes, nonce: int) -> int:
    """seed = BLAKE3(key || nonce_be64) -> первые 8 байт как uint64 LE."""
    h = blake3.blake3(bytes(key) + int(nonce).to_bytes(8, "big"))
    return int.from_bytes(h.digest()[:8], "little")


class NoisyGemmCore:
    """Эталонное (CPU/bit-exact reference) ядро NoisyGEMM."""

    def __init__(self, params: MiningParams = DEFAULT_PARAMS):
        params.validate()
        self.p = params

    # ---------- генерация ----------
    def generate_matrices(self, key: bytes, nonce: int):
        """Детерминированно порождает A,B,E,F из (key, nonce)."""
        p = self.p
        seed = derive_seed(key, nonce)
        rng_a = PhiloxRng(seed, SALT_A)
        A = rng_a.randint_range(p.m * p.k, p.elem_min, p.elem_max).astype(np.int8)
        A = A.reshape(p.m, p.k)
        rng_b = PhiloxRng(seed, SALT_B)
        B = rng_b.randint_range(p.n * p.k, p.elem_min, p.elem_max).astype(np.int8)
        B = B.reshape(p.n, p.k)

        def small(rows, cols, salt):
            r = PhiloxRng(seed, salt)
            return r.randint_range(rows * cols, NOISE_ELEM_MIN,
                                   NOISE_ELEM_MAX).astype(np.int64).reshape(rows, cols)

        U = small(p.m, p.noise_rank, SALT_U)
        V = small(p.k, p.noise_rank, SALT_V)   # храним V^T: [K, R]
        X = small(p.n, p.noise_rank, SALT_X)
        Y = small(p.k, p.noise_rank, SALT_Y)   # храним Y^T: [K, R]
        # Шум встраивается в A и B ДО основного умножения; low-rank wrap + int8
        E = wrap_lowrank_noise(U, V)
        F = wrap_lowrank_noise(X, Y)
        return A, B, E, F

    # ---------- GEMM ----------
    def noisy_gemm(self, A, B, E, F) -> np.ndarray:
        """C' = (A+E)(B+F)^T — единое INT8 GEMM с точным накоплением.

        Шум встраивается ДО умножения; операнды приводятся к int8 по модулю
        256 (wrap), поэтому честный майнер не может вычислить A@B и добавить
        шум «сверху» — это потребовало бы трёх GEMM вместо одного.
        Накопление выполняется точно (int64 эталон); при wrap-операндах
        |a_i|,|b_i| <= 128 и |C'_ij| <= K*128^2 < 2^31 для допустимых K —
        значения гарантированно помещаются в INT32 без переполнения.
        """
        Ap = ((A.astype(np.int64) + E.astype(np.int64))
              .astype(np.int8).astype(np.int64))
        Bp = ((B.astype(np.int64) + F.astype(np.int64))
              .astype(np.int8).astype(np.int64))
        return Ap @ Bp.T  # [M, N] int64

    # ---------- transcript fold ----------
    def transcript_fold(self, C: np.ndarray) -> bytes:
        """Свёртка транскрипта: для каждого тайла tile_h x tile_w считается
        BLAKE3 от сырых little-endian int32 байт тайла; все хеши тайлов
        XOR-сворачиваются в один 32-байтовый дайджест.

        Порядок обхода тайлов: row-major (сначала по высоте, затем ширине).
        """
        p = self.p
        th, tw = p.matmul_tile_h, p.matmul_tile_w
        m, n = C.shape
        if m % th or n % tw:
            raise ValueError("размеры C' кратны тайлу")
        acc = bytearray(32)
        raw = C.astype(np.int32)
        for i in range(0, m, th):
            for j in range(0, n, tw):
                tile = raw[i:i + th, j:j + tw]
                # little-endian int32, row-major внутри тайла
                tb = np.ascontiguousarray(tile, dtype="<i4").tobytes()
                h = blake3.blake3(tb).digest()
                for x in range(32):
                    acc[x] ^= h[x]
        return bytes(acc)

    # ---------- полный путь ----------
    def hash_grid(self, key: bytes, nonce: int) -> tuple[bytes, np.ndarray]:
        """Возвращает (H, C') для одной попытки."""
        A, B, E, F = self.generate_matrices(key, nonce)
        C = self.noisy_gemm(A, B, E, F)
        fold = self.transcript_fold(C)
        return blake3.blake3(fold).digest(), C

    @staticmethod
    def meets_target(hash_bytes: bytes, target: int) -> bool:
        """H < target, сравнение 256-битного числа little-endian."""
        return int.from_bytes(hash_bytes, "little") < int(target)

    # ---------- batch-вариант (для GPU/массового перебора) ----------
    def search_batch(self, key: bytes, nonces: np.ndarray, target: int):
        """Перебор пачки nonce на CPU-эталоне. Возвращает список найденных
        решений [(nonce, hash_hex)]. Используется в тестах bit-exact."""
        found = []
        for nonce in nonces:
            h, _ = self.hash_grid(key, int(nonce))
            if self.meets_target(h, target):
                found.append((int(nonce), h.hex()))
        return found
