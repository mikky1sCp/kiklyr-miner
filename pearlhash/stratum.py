"""Stratum-over-TCP клиент для пулов Pearl (PRL).

Протокол — классический Stratum: JSON-объекты по строке (newline-delimited),
методы `mining.subscribe`, `mining.authorize`, уведомления `mining.set_difficulty`
и `mining.notify`. Формат job минимален:

  * `key`    — бинарные данные (hex), из которых майнер детерминированно
               генерирует A и B (+ локальный nonce -> бесконечный поток гридов);
  * `target` — 256-битный порог (hex, little-endian);
  * `job_id` — идентификатор задания для отправки share.

Майнер НЕ получает готовые матрицы — они выводятся из key+nonce локально,
поэтому один job превращается в непрерывный поток попыток без простоев.

Примеры эндпоинтов открытых пулов:
  stratum+tcp://prl.suprnova.cc:3370   (VarDiff, старт с diff 10k)
  stratum+tcp://prl.kryptex.network:7048
  stratum+tcp://pool.nacre.network:3333
"""

from __future__ import annotations

import json
import socket
import threading
import queue
import time
from dataclasses import dataclass, field


@dataclass
class Job:
    job_id: str
    key: bytes                 # seed для генерации A/B
    target: int = (1 << 256) - 1
    difficulty: float = 1.0
    clean: bool = False        # clean jobs -> сбросить nonce-диапазон


def parse_target_hex(t: str | int) -> int:
    """target приходит hex-строкой; принимаем и little-, и big-endian варианты,
    явный int — как есть."""
    if isinstance(t, int):
        return t
    b = bytes.fromhex(t.removeprefix("0x"))
    le = int.from_bytes(b, "little")
    be = int.from_bytes(b, "big")
    # heuristic: если старшие байты LE нулевые — вероятно big-endian запись
    return be if le > be else le


class StratumClient:
    """Минимальный Stratum-клиент с подписочной очередью событий."""

    def __init__(self, host: str, port: int, worker: str = "x",
                 password: str = "", timeout: float = 30.0):
        self.host, self.port = host, port
        self.worker, self.password = worker, password
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._rpc_id = 0
        self._lock = threading.Lock()
        self._rx: queue.Queue[dict] = queue.Queue()
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self.connected = False

    # ---------- transport ----------
    def connect(self):
        self._sock = socket.create_connection((self.host, self.port),
                                              timeout=self.timeout)
        self.connected = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def close(self):
        self._stop.set()
        self.connected = False
        try:
            if self._sock:
                self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send_raw(self, obj: dict):
        assert self._sock is not None, "not connected"
        data = json.dumps(obj) + "\n"
        with self._lock:
            self._sock.sendall(data.encode())

    def request(self, method: str, params: list | dict | None = None,
                wait: bool = True, timeout: float | None = None):
        with self._lock:
            self._rpc_id += 1
            rid = self._rpc_id
        msg = {"id": rid, "method": method, "params": params or []}
        self._send_raw(msg)
        if not wait:
            return None
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            try:
                resp = self._rx.get(timeout=0.5)
            except queue.Empty:
                continue
            if resp.get("id") == rid:
                if resp.get("error"):
                    raise RuntimeError(f"stratum error: {resp['error']}")
                return resp.get("result")
        raise TimeoutError(f"no response to {method} (id={rid})")

    def notify(self, method: str, params: list | dict):
        """Одностороннее сообщение без id (mining.submit и т.п.)."""
        self._send_raw({"method": method, "params": params})

    # ---------- reader / dispatch ----------
    def _read_loop(self):
        buf = b""
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except (socket.timeout, OSError):
                continue
            if not chunk:
                self.connected = False
                self._events.put(("disconnected", None))
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._dispatch(msg)

    def _dispatch(self, msg: dict):
        if "id" in msg and "result" in msg or "error" in msg and msg.get("id"):
            self._rx.put(msg)
            return
        method = msg.get("method")
        params = msg.get("params") or []
        if method == "mining.set_difficulty":
            diff = float(params[0])
            self._events.put(("difficulty", diff))
        elif method == "mining.notify":
            self._events.put(("job", self._parse_notify(params)))
        elif method:
            self._events.put((method, params))

    @staticmethod
    def _parse_notify(params: list) -> Job:
        """Разбор mining.notify в обоих распространённых форматах.

        Формат A (простой PRL-style): [job_id, key_hex, target_hex, clean]
        Формат B (Ethereum-style):    [job_id, prevhash, coinbase1, coinbase2,
                                       merkle_branch[], blob1, blob2,
                                       clean, diff...]
        """
        if len(params) >= 3 and isinstance(params[1], str) and \
                len(params[1]) in (32, 64, 128) and not isinstance(params[2], list):
            job_id = str(params[0])
            key = bytes.fromhex(str(params[1]))
            target = parse_target_hex(params[2])
            clean = bool(params[3]) if len(params) > 3 else False
            return Job(job_id=job_id, key=key, target=target, clean=clean)
        # Формат B: собираем key из prevhash+coinbase+blobs (детерминированно)
        job_id = str(params[0])
        parts = []
        for p in params[1:]:
            if isinstance(p, str):
                try:
                    parts.append(bytes.fromhex(p))
                except ValueError:
                    parts.append(p.encode())
            elif isinstance(p, list):
                continue
            elif isinstance(p, bool):
                clean = p
        key = b"".join(parts)[:64]
        clean = bool(params[7]) if len(params) > 7 else False
        return Job(job_id=job_id, key=key, clean=clean)

    # ---------- high-level ----------
    def subscribe(self, agent: str = "pearlhash/0.1"):
        return self.request("mining.subscribe", [agent, None])

    def authorize(self):
        return self.request("mining.authorize", [self.worker, self.password])

    def submit_share(self, job: Job, nonce: int, hash_hex: str):
        """Отправка share: [worker, job_id, nonce_hex, ...]. Возвращает bool."""
        resp = self.request("mining.submit", [
            self.worker,
            job.job_id,
            f"{nonce:016x}",
            hash_hex,
        ], timeout=15.0)
        return resp is True or (isinstance(resp, dict) and resp.get("status"))

    def poll_event(self, timeout: float = 0.2):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None
