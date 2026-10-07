"""Size-bounded on-disk LRU cache for large binary payloads.

Payloads (points / controls / thumbnails) are written under the backend
.cache dir keyed by a hash of the full parameter tuple (stride, calib, ...)
plus the mtimes of the source files, so regeneration invalidates entries.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, root: Path, max_bytes: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._sizes: OrderedDict[str, int] = OrderedDict()
        self._total = 0
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Adopt pre-existing files; prune oldest by mtime if over budget."""
        files = [(p, p.stat().st_mtime_ns, p.stat().st_size)
                 for p in self.root.glob("*.bin")]
        for p, _, size in sorted(files, key=lambda t: t[1]):
            self._sizes[p.name] = size
            self._total += size
        while self._total > self.max_bytes and self._sizes:
            name = next(iter(self._sizes))
            self._drop(name)

    def _drop(self, name: str) -> None:
        size = self._sizes.pop(name, 0)
        self._total -= size
        try:
            (self.root / name).unlink()
        except OSError:
            pass

    @staticmethod
    def _file_for(key) -> str:
        h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
        return h + ".bin"

    def get(self, key) -> bytes | None:
        name = self._file_for(key)
        with self._lock:
            if name not in self._sizes:
                return None
            self._sizes.move_to_end(name)
        try:
            return (self.root / name).read_bytes()
        except OSError:
            with self._lock:
                self._sizes.pop(name, None)
            return None

    def put(self, key, data: bytes) -> None:
        name = self._file_for(key)
        with self._lock:
            if name in self._sizes:
                self._total -= self._sizes[name]
            self._sizes[name] = len(data)
            self._sizes.move_to_end(name)
            self._total += len(data)
            while self._total > self.max_bytes and self._sizes:
                self._drop(next(iter(self._sizes)))
        try:
            (self.root / name).write_bytes(data)
        except OSError:
            log.warning("disk cache write failed for %s", name)
