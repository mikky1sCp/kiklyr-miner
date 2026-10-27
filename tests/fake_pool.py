"""Локальный фейковый Stratum-пул для интеграционных тестов pearlhash.

Запуск: python tests/fake_pool.py [port]
Отправляет mining.set_difficulty + mining.notify (формат PRL-style:
[job_id, key_hex, target_hex, clean]), принимает mining.submit и валидирует
share независимым вызовом CPU-эталона.
"""

from __future__ import annotations

import json
import socket
import sys
import threading

sys.path.insert(0, ".")
from pearlhash.core import NoisyGemmCore            # noqa: E402
from pearlhash.cpu import CpuBackend                 # noqa: E402
from pearlhash.miner import difficulty_to_target     # noqa: E402

CORE = NoisyGemmCore()
BACKEND = CpuBackend()


JOBS: dict[str, tuple[bytes, int]] = {}


def handle(conn: socket.socket):
    buf = b""
    def send(obj):
        conn.sendall((json.dumps(obj) + "\n").encode())

    # handshake push
    send({"method": "mining.set_difficulty", "params": [10_000]})
    key = bytes(range(32))
    target = difficulty_to_target(10_000)
    job = {"method": "mining.notify",
           "params": ["0", key.hex(), f"{target:064x}", True]}
    send(job)
    JOBS["0"] = (key, target)

    while True:
        try:
            data = conn.recv(65536)
        except OSError:
            return
        if not data:
            return
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            msg = json.loads(line)
            rid = msg.get("id")
            method = msg.get("method")
            params = msg.get("params") or []
            if method == "mining.subscribe":
                send({"id": rid, "result": [["fake", "1.0"], "extranonce1", "en2"],
                      "error": None})
            elif method == "mining.authorize":
                send({"id": rid, "result": True, "error": None})
            elif method == "mining.submit":
                worker, job_id, nonce_hex, hash_hex = (list(params) + [None]*4)[:4]
                nonce = int(nonce_hex, 16)
                k, t = JOBS[str(job_id)]
                h, _ = CORE.hash_grid(k, nonce)
                ok = h.hex() == hash_hex and NoisyGemmCore.meets_target(h, t)
                print(f"[fake-pool] submit worker={worker} nonce={nonce} "
                      f"valid={ok}")
                send({"id": rid, "result": bool(ok), "error": None})


def main(port: int = 34567):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(4)
    print(f"fake pool listening on 127.0.0.1:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 34567)
