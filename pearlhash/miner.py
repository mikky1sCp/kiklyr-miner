"""Оркестратор майнера pearlhash.

Режимы:
  * solo (без пула) — локальный перебор nonce с самопроверкой;
  * pool (stratum)  — подключение к пулу, приём job'ов {key, target, job_id},
    локальный nonce-менеджмент (nonce инкрементируется на каждый грид, пулу
    не нужно выдавать новый job для каждой попытки), отправка share при
    H < pool_target. ZK-доказательство (Plonky2 ~60KB) строит пул/нода,
    не майнер.

Backend'и выбираются автоматически: cuda (INT8 tensor cores) -> cpu-reference
(analog DP4A эталона). Перед использованием нетensor-core пути выполняется
bit-exact сверка на случайных nonce.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .params import MiningParams, DEFAULT_PARAMS
from .core import NoisyGemmCore
from .cpu import CpuBackend

log = logging.getLogger("pearlhash")


def difficulty_to_target(difficulty: float) -> int:
    """diff -> 256-битный target: target = floor(2^256 / diff).

    diff=1 => любой хеш проходит; рост difficulty линейно уменьшает
    вероятность H < target.
    """
    if difficulty <= 0:
        raise ValueError("difficulty должна быть > 0")
    return (1 << 256) // int(difficulty)


@dataclass
class MinerStats:
    hashes: int = 0
    shares_found: int = 0
    shares_accepted: int = 0
    shares_stale: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def rate(self) -> float:
        dt = time.time() - self.started_at
        return self.hashes / dt if dt > 0 else 0.0


class PearlMiner:
    def __init__(self, params: MiningParams = DEFAULT_PARAMS,
                 backend: str = "auto",
                 pool_host: str | None = None, pool_port: int = 3370,
                 worker: str = "worker", password: str = "",
                 dry_run: bool = False):
        self.params = params
        self.stats = MinerStats()
        self.dry_run = dry_run
        self._stop = threading.Event()
        self.backend = self._make_backend(backend)
        self.pool = None
        if pool_host:
            from .stratum import StratumClient
            self.pool = StratumClient(pool_host, pool_port, worker, password)
        self.current_job = None          # stratum.Job
        self.nonce = 0                   # локальный счётчик на каждый грид
        log.info("backend: %s (%s)", self.backend.name,
                 getattr(self.backend, "device", "cpu"))

    # ---------- backend selection + bit-exact verify ----------
    def _make_backend(self, kind: str):
        if kind in ("auto", "cuda"):
            try:
                from .cuda import CudaBackend
                cb = CudaBackend(self.params,
                                 device="cuda" if kind == "auto" else None)
                if cb.available():
                    ok = self.verify_bit_exact(CpuBackend(self.params), cb, n=3)
                    if ok:
                        log.info("CUDA backend bit-exact verified vs CPU reference")
                        return cb
                    log.warning("CUDA backend NOT bit-exact; falling back to CPU")
            except Exception as e:  # noqa: BLE001
                if kind == "cuda":
                    raise
                log.debug("cuda unavailable: %s", e)
        return CpuBackend(self.params)

    @staticmethod
    def verify_bit_exact(ref, other, n: int = 3, key: bytes | None = None) -> bool:
        """Сверка путей на случайных nonce (как DP4A-эталон валидирует ядра)."""
        rng = np.random.default_rng(0xC0FFEE)
        key = key or rng.bytes(32)
        for _ in range(n):
            nonce = int(rng.integers(0, 1 << 32))
            h1 = ref.hash_grid(key, nonce) if hasattr(ref, "hash_grid") \
                else ref.hash_one(key, nonce)
            h2 = other.hash_grid(key, nonce) if hasattr(other, "hash_grid") \
                else other.hash_one(key, nonce)
            if h1 != h2:
                return False
        return True

    # ---------- stratum loop ----------
    def connect_pool(self):
        assert self.pool, "pool not configured"
        self.pool.connect()
        self.pool.subscribe()
        if not self.pool.authorize():
            raise RuntimeError("authorize failed")
        log.info("connected to %s:%s", self.pool.host, self.pool.port)

    def _handle_events(self):
        while not self._stop.is_set():
            ev = self.pool.poll_event(timeout=0.1)
            if ev is None:
                continue
            kind, payload = ev
            if kind == "job":
                from .stratum import Job
                job: Job = payload
                if job.clean or self.current_job is None or \
                        job.job_id != self.current_job.job_id:
                    self.nonce = 0  # сброс диапазона на новом блоке
                self.current_job = job
                log.info("new job id=%s key=%s... diff=%.1f",
                         job.job_id, job.key.hex()[:16], job.difficulty)
            elif kind == "difficulty":
                diff = float(payload)
                if self.current_job is not None:
                    self.current_job.difficulty = diff
                    self.current_job.target = difficulty_to_target(diff)
                log.info("difficulty set to %.1f", diff)
            elif kind == "disconnected":
                log.error("pool disconnected")
                self._stop.set()

    # ---------- mining loop ----------
    def mine_batch(self, key: bytes, target: int, batch: int = 8):
        """Перебор `batch` nonce от текущего self.nonce; возвращает решения."""
        solutions = []
        for _ in range(batch):
            n = self.nonce
            self.nonce += 1                      # nonce инкрементируется ЛОКАЛЬНО
            h = self.backend.hash_grid(key, n)
            self.stats.hashes += 1
            if NoisyGemmCore.meets_target(h, target):
                solutions.append((n, h.hex()))
                self.stats.shares_found += 1
        return solutions

    def run(self, duration: float | None = None, solo_key: bytes | None = None,
            solo_difficulty: float = 1.0, report_every: float = 5.0):
        """Основной цикл. duration=None — бесконечно до stop()."""
        t_end = time.time() + duration if duration else None
        last_report = time.time()

        if self.pool:
            self.connect_pool()
            threading.Thread(target=self._handle_events, daemon=True).start()

        while not self._stop.is_set():
            if self.pool and self.current_job:
                job = self.current_job
                sols = self.mine_batch(job.key, job.target)
                for nonce, hh in sols:
                    if self.dry_run:
                        log.info("[dry-run] share job=%s nonce=%d hash=%s",
                                 job.job_id, nonce, hh)
                        self.stats.shares_accepted += 1
                        continue
                    try:
                        acc = self.pool.submit_share(job, nonce, hh)
                        if acc:
                            self.stats.shares_accepted += 1
                            log.info("share ACCEPTED nonce=%d", nonce)
                        else:
                            self.stats.shares_stale += 1
                            log.warning("share REJECTED nonce=%d", nonce)
                    except Exception as e:  # noqa: BLE001
                        log.error("submit failed: %s", e)
            elif solo_key is not None:
                target = difficulty_to_target(solo_difficulty)
                self.mine_batch(solo_key, target)
            else:
                time.sleep(0.05)

            now = time.time()
            if now - last_report >= report_every:
                log.info("hashes=%d rate=%.2f grid/s shares=%d/%d",
                         self.stats.hashes, self.stats.rate,
                         self.stats.shares_accepted, self.stats.shares_found)
                last_report = now
            if t_end and now >= t_end:
                break

        if self.pool:
            self.pool.close()
        return self.stats

    def stop(self):
        self._stop.set()
