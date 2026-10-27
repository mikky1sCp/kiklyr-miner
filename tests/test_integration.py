"""Интеграционные тесты: bit-exact CPU vs CUDA/torch путь, stratum-протокол,
e2e майнинг против локального фейкового пула."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import torch

from pearlhash.params import MiningParams
from pearlhash.cpu import CpuBackend
from pearlhash.cuda import CudaBackend
from pearlhash.core import NoisyGemmCore
from pearlhash.miner import PearlMiner, difficulty_to_target
from pearlhash.stratum import StratumClient, Job, parse_target_hex

SMALL = MiningParams(m=64, n=64, k=64, noise_rank=16,
                     matmul_tile_h=16, matmul_tile_w=16)


# ---------- backends ----------
@pytest.mark.parametrize("nonce", [0, 1, 12345])
def test_cpu_cuda_bit_exact(nonce):
    ref = CpuBackend(SMALL)
    gpu = CudaBackend(SMALL, device="cpu")  # torch-путь без tensor cores
    assert ref.hash_one(b"key", nonce) == gpu.hash_grid(b"key", nonce)


def test_torch_int_mm_matches_numpy():
    """INT8 tensor-core путь (_int_mm) == точный int64 matmul."""
    if not (torch.cuda.is_available() and hasattr(torch, "_int_mm")):
        pytest.skip("no CUDA / _int_mm")
    rng = np.random.default_rng(0)
    a = rng.integers(-64, 65, size=(64, 64)).astype(np.int8)
    b = rng.integers(-64, 65, size=(64, 64)).astype(np.int8)
    exact = a.astype(np.int64) @ b.astype(np.int64).T
    got = torch._int_mm(torch.from_numpy(a).cuda(),
                        torch.from_numpy(b.T.copy()).cuda())
    assert (exact == got.cpu().numpy()).all()


def test_backend_autoselect_and_verify():
    miner = PearlMiner(params=SMALL, backend="auto")
    assert miner.backend.name in ("cpu-reference", "cuda-int8-tensorcore")
    assert PearlMiner.verify_bit_exact(CpuBackend(SMALL), miner.backend, n=2)


# ---------- stratum parsing ----------
def test_parse_notify_prl_style():
    key = bytes(range(16))
    target = difficulty_to_target(1000)
    job = StratumClient._parse_notify(
        ["7", key.hex(), f"{target:064x}", True])
    assert job.job_id == "7" and job.key == key
    assert job.target == target and job.clean


def test_difficulty_to_target_monotonic():
    t1 = difficulty_to_target(1)
    t2 = difficulty_to_target(10_000)
    assert t1 > t2 > 0


# ---------- e2e against fake pool ----------
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_e2e_fake_pool_submit_accepted():
    """Майнер отправляет честный share (nonce=0, target=max) — фейковый пул
    пересчитывает hash независимо и принимает его."""
    import tests.fake_pool as fp

    port = _free_port()
    srv = threading.Thread(target=fp.main, args=(port,), daemon=True)
    srv.start()
    time.sleep(0.3)

    client = StratumClient("127.0.0.1", port, worker="test.rig", password="x")
    client.connect()
    client.subscribe()
    assert client.authorize() is True

    # дождаться job
    job = None
    deadline = time.time() + 5
    while time.time() < deadline and job is None:
        ev = client.poll_event(timeout=0.5)
        if ev and ev[0] == "job":
            job = ev[1]
    assert isinstance(job, Job) and len(job.key) == 32

    # форсируем target=max (diff=1), чтобы nonce=0 гарантированно был решением
    job.target = (1 << 256) - 1
    miner = PearlMiner(params=MiningParams(), backend="cpu")
    sols = miner.mine_batch(job.key, job.target, batch=1)
    assert len(sols) == 1
    nonce, hash_hex = sols[0]

    accepted = client.submit_share(job, nonce, hash_hex)
    assert accepted is True, "пул должен принять честный share"
    client.close()
