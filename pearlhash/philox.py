"""Philox-4x32-10 CSPRNG — чистая реализация на NumPy.

Используется для детерминированной генерации матриц A, B и шумовых
факторов U, V, X, Y из seed = BLAKE3(key || nonce).

Ключевые свойства:
  * counter-based (CTR режим): выход = E_key(counter), state не мутируется;
  * 10 раундов Philox (Philox-4x32-10);
  * побитово воспроизводим на любой платформе (все операции uint32);
  * seed 64 бита -> ключ {seed_lo, seed_hi}; независимые потоки для разных
    ролей (A/B/U/V/X/Y) получаются через salt (сдвиг старшего слова ключа).

Константы — Random123 (Salmon et al., SC '11) / curand Philox-4x32-10:
  M0 = 0xD251_1F53, M1 = 0xCD9E_8D57, W0 = 0x9E37_79B9, W1 = 0xBB67_AE85
Раунд (каноническая раскладка curand/Random123):
  (hi0, lo0) = mulu64(x0, M0);  (hi1, lo1) = mulu64(x2, M1)
  x0' = hi1 ^ x1 ^ k0
  x1' = lo1 ^ x3 ^ k1
  x2' = hi0 ^ x0 ^ W0
  x3' = lo0 ^ x0 ^ W1   (во всех выражениях справа — СТАРЫЕ значения)
"""

from __future__ import annotations

import numpy as np

# --- константы Philox-4x32-10 (Random123 / CUDA curand) ---
M0 = np.uint32(0xD2511F53)   # multiplier для пары (x0, x1)
M1 = np.uint32(0xCD9E8D57)   # multiplier для пары (x2, x3)
W0 = np.uint32(0x9E3779B9)   # round constant (golden ratio 32-bit)
W1 = np.uint32(0xBB67AE85)   # round constant (sqrt(2)-1 32-bit)

ROUNDS = 10
U32 = np.uint32
MASK32 = np.uint64(0xFFFFFFFF)


def _mul_hi_lo(x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """32x32->64 умножение; возвращает (hi, lo) как uint32 массивы."""
    p = x.astype(np.uint64) * m.astype(np.uint64)
    lo = (p & MASK32).astype(U32)
    hi = (p >> np.uint64(32)).astype(U32)
    return hi, lo


def philox_round(x0, x1, x2, x3, k0, k1):
    """Один раунд Philox-4x32 (все аргументы — uint32 массивы одной формы)."""
    hi0, lo0 = _mul_hi_lo(x0, M0)
    hi1, lo1 = _mul_hi_lo(x2, M1)
    n0 = (hi1 ^ x1 ^ k0).astype(U32)
    n1 = (lo1 ^ x3 ^ k1).astype(U32)
    n2 = (hi0 ^ x0 ^ W0).astype(U32)
    n3 = (lo0 ^ x0 ^ W1).astype(U32)
    return n0, n1, n2, n3


def philox10(ctr, key):
    """Philox-4x32-10: ctr=(c0..c3), key=(k0,k1) -> (r0..r3), uint32 массивы."""
    x0, x1, x2, x3 = (np.asarray(c, dtype=U32) for c in ctr)
    k0, k1 = (np.asarray(k, dtype=U32) for k in key)
    for _ in range(ROUNDS):
        x0, x1, x2, x3 = philox_round(x0, x1, x2, x3, k0, k1)
    return x0, x1, x2, x3


class PhiloxRng:
    """Counter-based RNG поверх Philox-4x32-10.

    seed: int (<=64 бит) или bytes (первые 8 байт, little-endian).
    salt: смещение старшего слова ключа — независимый поток для каждой
          роли (A, B, U, V, X, Y), аналог subsequence в torch Philox.
    """

    def __init__(self, seed: int | bytes, salt: int = 0):
        if isinstance(seed, (bytes, bytearray)):
            b = bytes(seed[:8]).ljust(8, b"\x00")
            seed = int.from_bytes(b, "little")
        seed &= (1 << 64) - 1
        self.k0 = U32(seed & 0xFFFFFFFF)
        self.k1 = U32(((seed >> 32) + salt) & 0xFFFFFFFF)
        self._counter = 0  # счётчик 4-словных блоков

    def next_words(self, count: int) -> np.ndarray:
        """Сгенерировать `count` слов uint32, продвинув счётчик блоков."""
        out = np.empty(count, dtype=U32)
        pos = 0
        while pos < count:
            n_blocks = min((count - pos + 3) // 4, 1 << 20)
            idx = np.arange(self._counter, self._counter + n_blocks,
                            dtype=np.uint64)
            c0 = (idx & MASK32).astype(U32)
            c1 = (idx >> np.uint64(32)).astype(U32)
            z = np.zeros_like(c0)
            r0, r1, r2, r3 = philox10((c0, c1, z, z),
                                      (np.full_like(c0, self.k0),
                                       np.full_like(c0, self.k1)))
            words = np.stack([r0, r1, r2, r3], axis=1).reshape(-1)
            take = min(count - pos, n_blocks * 4)
            out[pos:pos + take] = words[:take]
            pos += take
            self._counter += n_blocks
        return out

    def randint_range(self, count: int, lo: int, hi: int) -> np.ndarray:
        """Равномерные int64 в [lo, hi]; rejection sampling против modulo bias."""
        span = hi - lo + 1
        if span < 1:
            raise ValueError("пустой диапазон")
        limit = (1 << 32) - ((1 << 32) % span)
        vals = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            w = self.next_words(count - filled)
            keep = w[w < limit]
            vals[filled:filled + keep.size] = \
                keep.astype(np.int64) % span + lo
            filled += keep.size
        return vals

    def randbytes(self, nbytes: int) -> bytes:
        w = self.next_words((nbytes + 3) // 4).astype("<u4").tobytes()
        return w[:nbytes]
