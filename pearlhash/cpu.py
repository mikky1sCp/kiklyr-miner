"""CPU-эталонный backend майнера (analog Pascal DP4A reference path).

Использует NumPy int64 matmul — точное целое накопление, бит-эквивалентно
корректным INT8/INT32 путям на GPU. Считается эталоном bit-exact для
валидации остальных ядер (как DP4A в референсных реализациях).
"""

from __future__ import annotations

import time
import numpy as np

from .core import NoisyGemmCore
from .params import MiningParams, DEFAULT_PARAMS


class CpuBackend:
    name = "cpu-reference"

    def __init__(self, params: MiningParams = DEFAULT_PARAMS):
        self.core = NoisyGemmCore(params)

    def hash_one(self, key: bytes, nonce: int) -> bytes:
        h, _ = self.core.hash_grid(key, nonce)
        return h

    # алиас для единообразного интерфейса backend'ов (см. miner.verify_bit_exact)
    hash_grid = hash_one

    def search(self, key: bytes, nonce_start: int, count: int, target: int):
        """Перебор `count` nonce начиная с nonce_start.

        Возвращает (found, stats), found = [(nonce, hash_hex)],
        stats = {"hashes": n, "elapsed": sec}.
        """
        t0 = time.perf_counter()
        found = []
        for n in range(count):
            nonce = nonce_start + n
            h, _ = self.core.hash_grid(key, nonce)
            if NoisyGemmCore.meets_target(h, target):
                found.append((nonce, h.hex()))
        return found, {"hashes": count, "elapsed": time.perf_counter() - t0}
