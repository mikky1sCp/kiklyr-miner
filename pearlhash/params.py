"""Параметры референсной реализации Pearl (public_params / mining_config).

Значения соответствуют открытой конфигурации монорепозитория
`pearl-research-labs/pearl` и поведению открытых пуловых майнеров
(SRBMiner / Neil Pearl Miner):

| Параметр         | Значение | Примечание                          |
|------------------|----------|-------------------------------------|
| noise_range      | 128      | Диапазон шума                       |
| noise_rank (R)   | 64       | Ранг низкорангового шума            |
| matmul_tile_h    | 64       | Размер тайла по высоте (CUDA)       |
| matmul_tile_w    | 64       | Размер тайла по ширине (CUDA)       |
| Точность входа   | INT8     | элементы в [-64, 64]                |
| Накопление       | INT32    | точное целое, без переполнения      |
| Хеш              | BLAKE3   | свёртка транскрипта                 |
| Время блока      | ~112.8 c | эмиссия E(t)=S*H/((t+H)(t+H-1))     |
| Макс. эмиссия    | 2.1 млрд | PRL, без халвинга                   |

Размеры матриц M/N/K не фиксированы протоколом жёстко (в открытых
реализациях варьируются); здесь выбраны кратными размеру тайла 64.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MiningParams:
    # --- геометрия NoisyGEMM ---
    m: int = 256          # C' имеет размер M x N; кратно tile
    n: int = 256
    k: int = 256          # внутреннее измерение (накопление по K)
    noise_rank: int = 64  # R — ранг структурированного шума E = U@V, F = X@Y

    # --- тайлинг CUDA-ядра (матрица C' разбивается на tile_h x tile_w) ---
    matmul_tile_h: int = 64
    matmul_tile_w: int = 64

    # --- диапазон элементов ---
    elem_min: int = -64
    elem_max: int = 64    # 129 значений, помещаются в int8

    noise_range: int = 128

    # --- точность ---
    input_dtype_bits: int = 8     # INT8
    accum_dtype_bits: int = 32    # INT32

    # --- хеш ---
    hash_algo: str = "BLAKE3"
    hash_len_bytes: int = 32

    # --- сеть / экономика (для расчётов и отображения) ---
    block_time_sec: float = 112.8
    max_supply_prl: int = 2_100_000_000
    default_block_reward_prl: int = 2688

    # --- Philox ---
    philox_rounds: int = 10       # Philox-4x32-10
    seed_bits: int = 64

    def validate(self) -> None:
        if self.elem_min != -self.elem_max:
            raise ValueError("несимметричный диапазон элементов")
        for name, dim in (("m", self.m), ("n", self.n), ("k", self.k)):
            if dim % self.matmul_tile_h != 0 or dim % self.matmul_tile_w != 0:
                raise ValueError(f"{name}={dim} должно кратно размеру тайла")
        if self.noise_rank > min(self.m, self.n, self.k):
            raise ValueError("noise_rank превышает размеры матриц")
        # Контроль переполнения INT32: максимум |sum a_i*b_i| <= K * 64*64
        overflow_bound = self.k * self.elem_max * self.elem_max
        if overflow_bound > 2 ** (self.accum_dtype_bits - 1) - 1:
            raise ValueError(
                f"возможное переполнение INT32: bound={overflow_bound}")

    @property
    def tiles(self) -> tuple[int, int]:
        """Количество тайлов по высоте/ширине матрицы C'."""
        return (self.m // self.matmul_tile_h, self.n // self.matmul_tile_w)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MiningParams":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# Дефолтный набор — «public_params» референса.
DEFAULT_PARAMS = MiningParams()
