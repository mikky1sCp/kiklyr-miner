"""CLI майнера pearlhash.

Примеры:
  # solo-перебор 100 гридов с diff 1e6 (тестовый режим)
  python -m pearlhash.cli --solo-key deadbeef --duration 10 --difficulty 1e9

  # пул (stratum over TCP)
  python -m pearlhash.cli --pool prl.suprnova.cc --port 3370 \
      --worker WALLET.rig1 --password x

  # dry-run против локального фейкового пула (для отладки протокола)
  python tests/fake_pool.py &   # затем
  python -m pearlhash.cli --pool 127.0.0.1 --port 34567 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

from .params import MiningParams
from .miner import PearlMiner


def main(argv=None):
    ap = argparse.ArgumentParser("pearlhash", description="Pearl NoisyGEMM miner")
    ap.add_argument("--pool", help="stratum host (без scheme stratum+tcp://)")
    ap.add_argument("--port", type=int, default=3370)
    ap.add_argument("--worker", default="worker.1")
    ap.add_argument("--password", default="x")
    ap.add_argument("--solo-key", help="hex key для solo-режима")
    ap.add_argument("--difficulty", type=float, default=1e6,
                    help="solo difficulty (target = 2^256/diff)")
    ap.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--duration", type=float, default=None,
                    help="секунды работы (по умолчанию — бесконечно)")
    ap.add_argument("--batch", type=int, default=8, help="гридов за итерацию")
    ap.add_argument("--dry-run", action="store_true",
                    help="не отправлять share в пул")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    params = MiningParams()
    solo_key = bytes.fromhex(args.solo_key) if args.solo_key else None
    if not args.pool and solo_key is None:
        ap.error("нужно указать --pool POOL или --solo-key KEY")

    miner = PearlMiner(params=params, backend=args.backend,
                       pool_host=args.pool, pool_port=args.port,
                       worker=args.worker, password=args.password,
                       dry_run=args.dry_run)

    try:
        stats = miner.run(duration=args.duration, solo_key=solo_key,
                          solo_difficulty=args.difficulty)
    except KeyboardInterrupt:
        miner.stop()
        stats = miner.stats
    print(f"done: hashes={stats.hashes} rate={stats.rate:.2f} grid/s "
          f"shares found={stats.shares_found} accepted={stats.shares_accepted} "
          f"rejected={stats.shares_stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
