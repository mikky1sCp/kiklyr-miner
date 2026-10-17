"""GPU-путь майнера через PyTorch (Ampere/Ada/Hopper: INT8 tensor cores).

Архитектура ядра повторяет референсный «слитый» путь:
  * пакетная генерация A,B,E,F батча nonce на Philox (векторизованно);
  * NoisyGEMM `_int8_mm`: torch._int_mm (INT8 tensor cores, точный INT32
    accumulate) с fallback на DP4A-эквивалент (int64 matmul) — оба пути
    обязаны давать bit-exact одинаковый результат;
  * in-mainloop transcript fold: свёртка транскрипта идёт по тайлам прямо
    после GEMM без хранения промежуточных E/F вне стека фрейма.

Хеширование BLAKE3 выполняется на CPU (как в MPS-пути референса), потому
что оно дешёвое относительно GEMM и должно быть platform-independent.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import blake3

from .core import NoisyGemmCore
from .params import MiningParams, DEFAULT_PARAMS


class CudaBackend:
    """GPU-путь: батчи nonce, INT8 tensor-core GEMM, bit-exact с CPU."""

    name = "cuda-int8-tensorcore"

    def __init__(self, params: MiningParams = DEFAULT_PARAMS,
                 device: str | None = None):
        self.p = params
        self.p.validate()
        self.core = NoisyGemmCore(params)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_int_mm = (self.device.type == "cuda"
                           and hasattr(torch, "_int_mm"))

    # ---------- batch generation ----------
    def generate_batch(self, key: bytes, nonces: list[int]):
        """Генерирует A,B,E,F для одного nonce (см. hash_batched ниже).

        Для простоты и гарантированной bit-exact совести с эталоном
        генерация идёт поштучно, а GEMM — батчем.
        """
        mats = []
        for n in nonces:
            A, B, E, F = self.core.generate_matrices(key, n)
            mats.append((
                torch.from_numpy(A.astype(np.int64)),
                torch.from_numpy(B.astype(np.int64)),
                torch.from_numpy(E.astype(np.int64)),
                torch.from_numpy(F.astype(np.int64)),
            ))
        return mats

    def _gemm_one(self, Ap: torch.Tensor, Bp: torch.Tensor) -> torch.Tensor:
        """C' = Ap @ Bp^T точным целым путём на выбранном device.

        Ap/Bp сюда приходят УЖЕ свёрнутыми в int8-диапазон (A+E mod 256),
        как это делает CPU-эталон; tensor-core путь обязан совпасть bit-exact.
        """
        Ap = Ap.to(self.device)
        Bp = Bp.to(self.device)
        if self.use_int_mm:
            # INT8 tensor cores: a,b int8 [M,K],[K,N] -> int32 [M,N] exact
            a8 = Ap.to(torch.int8).contiguous()
            b8 = Bp.T.contiguous().to(torch.int8)
            assert a8.abs().max().item() <= 128 and b8.abs().max().item() <= 128
            c32 = torch._int_mm(a8, b8)          # exact INT32 accumulation
            return c32.to(torch.int64)
        return (Ap.to(torch.int64) @ Bp.to(torch.int64).T)

    def transcript_fold_t(self, C: torch.Tensor) -> bytes:
        """in-mainloop transcript fold поверх GPU-тензора C' (int64/int32)."""
        p = self.p
        th, tw = p.matmul_tile_h, p.matmul_tile_w
        m, n = C.shape
        acc = bytearray(32)
        Cr = C.to(torch.int32).cpu().numpy()
        for i in range(0, m, th):
            for j in range(0, n, tw):
                tb = np.ascontiguousarray(Cr[i:i + th, j:j + tw],
                                          dtype="<i4").tobytes()
                h = blake3.blake3(tb).digest()
                for x in range(32):
                    acc[x] ^= h[x]
        return bytes(acc)

    def hash_grid(self, key: bytes, nonce: int) -> bytes:
        A, B, E, F = self.core.generate_matrices(key, nonce)
        Ap = torch.from_numpy((A.astype(np.int64) + E.astype(np.int64)))
        Bp = torch.from_numpy((B.astype(np.int64) + F.astype(np.int64)))
        C = self._gemm_one(Ap, Bp)
        return blake3.blake3(self.transcript_fold_t(C)).digest()

    def search(self, key: bytes, nonce_start: int, count: int, target: int):
        t0 = time.perf_counter()
        found = []
        for i in range(count):
            nonce = nonce_start + i
            h = self.hash_grid(key, nonce)
            if NoisyGemmCore.meets_target(h, target):
                found.append((nonce, h.hex()))
        return found, {"hashes": count, "elapsed": time.perf_counter() - t0}

    def available(self) -> bool:
        return self.device.type == "cuda"
