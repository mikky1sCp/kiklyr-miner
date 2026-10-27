"""Юнит-тесты ядра pearlhash: Philox, NoisyGEMM, transcript fold, target."""

from __future__ import annotations

import numpy as np
import pytest
import blake3

from pearlhash.params import MiningParams, DEFAULT_PARAMS
from pearlhash.philox import PhiloxRng, philox10
from pearlhash.core import (NoisyGemmCore, derive_seed, SALT_A,
                            NOISE_ELEM_MAX)


SMALL = MiningParams(m=64, n=64, k=64, noise_rank=16,
                     matmul_tile_h=16, matmul_tile_w=16)


def test_params_defaults_match_reference():
    p = DEFAULT_PARAMS
    assert p.noise_range == 128
    assert p.noise_rank == 64
    assert p.matmul_tile_h == 64 and p.matmul_tile_w == 64
    assert p.elem_min == -64 and p.elem_max == 64
    assert p.input_dtype_bits == 8 and p.accum_dtype_bits == 32
    assert p.hash_algo == "BLAKE3"
    assert p.block_time_sec == pytest.approx(112.8)
    assert p.max_supply_prl == 2_100_000_000
    p.validate()


def test_params_int32_overflow_guard():
    # K * 64 * 64 не должен превышать int32 max: 2^19 * 4096 = 2^31 > 2^31-1
    bad = MiningParams(k=2 ** 19)
    with pytest.raises(ValueError):
        bad.validate()


def test_philox_deterministic_and_salt_independent():
    a = PhiloxRng(42, 0).next_words(64)
    b = PhiloxRng(42, 0).next_words(64)
    c = PhiloxRng(42, 1).next_words(64)
    assert (a == b).all()
    assert not (a == c).all()


def test_philox_uniform_range():
    vals = PhiloxRng(7, 0).randint_range(50_000, -64, 64)
    assert vals.min() >= -64 and vals.max() <= 64
    # грубая проверка равномерности: среднее близко к 0
    assert abs(vals.mean()) < 1.0
    # все 129 значений встречаются
    assert len(np.unique(vals)) == 129


def test_derive_seed_depends_on_key_and_nonce():
    s1 = derive_seed(b"key1", 1)
    s2 = derive_seed(b"key1", 1)
    s3 = derive_seed(b"key1", 2)
    s4 = derive_seed(b"key2", 1)
    assert s1 == s2 and s1 != s3 and s1 != s4


def test_matrices_shapes_and_ranges():
    core = NoisyGemmCore(SMALL)
    A, B, E, F = core.generate_matrices(b"k", 0)
    assert A.shape == (64, 64) and B.shape == (64, 64)
    assert A.dtype == np.int8 and B.dtype == np.int8
    assert A.min() >= -64 and A.max() <= 64
    # шум низкоранговый: E = U@V с рангом <= R=16
    rank_E = np.linalg.matrix_rank(E.astype(np.float64))
    assert rank_E <= 16 + 4  # clip может слегка увеличить; грант на клип


def test_noisy_gemm_exactness():
    """C' = (A+E)(B+F)^T — точное целое, в диапазоне INT32."""
    core = NoisyGemmCore(SMALL)
    A, B, E, F = core.generate_matrices(b"k", 3)
    C = core.noisy_gemm(A, B, E, F)
    assert C.dtype == np.int64
    bound = SMALL.k * 64 * 64
    assert np.abs(C).max() <= bound
    assert (C >= -(2 ** 31)).all() and (C < 2 ** 31).all()


def test_transcript_fold_deterministic_and_order_sensitive():
    core = NoisyGemmCore(SMALL)
    rng = np.random.default_rng(1)
    C = rng.integers(-1000, 1000, size=(64, 64))
    f1 = core.transcript_fold(C)
    f2 = core.transcript_fold(C)
    assert f1 == f2 and len(f1) == 32
    C2 = C.copy()
    C2[0, 0] += 1  # одночленное изменение -> другой fold (лавина)
    assert core.transcript_fold(C2) != f1


def test_hash_grid_reproducible():
    core = NoisyGemmCore(SMALL)
    h1, _ = core.hash_grid(b"abc", 5)
    h2, _ = core.hash_grid(b"abc", 5)
    h3, _ = core.hash_grid(b"abc", 6)
    assert h1 == h2 and h1 != h3


def test_meets_target_le_bytes():
    t = (1 << 256) // 10_000
    low = (t - 1).to_bytes(32, "little")
    high = (t + 1).to_bytes(32, "little")
    assert NoisyGemmCore.meets_target(low, t)
    assert not NoisyGemmCore.meets_target(high, t)


def test_blake3_known_vector():
    assert blake3.blake3(b"abc").hexdigest().startswith("6437b3ac")
