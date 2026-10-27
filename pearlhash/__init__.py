"""pearlhash — прототип майнера Pearl (NoisyGEMM INT8 + BLAKE3 + Stratum).

Модули:
  params     — публичные параметры протокола (public_params / mining_config)
  philox     — CSPRNG Philox-4x32-10 (побитово совместим с torch.Generator('philox'))
  core       — математическое ядро: генерация A,B,E,F, NoisyGEMM, transcript fold, BLAKE3
  cpu        — CPU-эталонный путь (int8->int64 matmul, bit-exact reference)
  cuda       — GPU-путь через torch CUDA int8 tensor cores (bit-exact с CPU)
  stratum    — Stratum-over-TCP клиент (JSON line protocol)
  miner      — оркестратор: backend-менеджмент, nonce-менеджмент, share submission
"""

from .params import MiningParams
from .core import NoisyGemmCore

__version__ = "0.1.0"
__all__ = ["MiningParams", "NoisyGemmCore", "__version__"]
