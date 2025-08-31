import copy
import logging
import threading
from typing import Any, Hashable, TypeVar
from collections.abc import Callable

T = TypeVar("T")


class CacheStore:
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        validator: Callable[[Any], bool] | None = None,
        copier: Callable[[Any], Any] = copy.copy,
    ) -> None:
        self.name = name
        self.logger = logger
        self._validator = validator
        self._copier = copier
        self._cache: dict[Hashable, T] = {}
        self._lock: dict[Hashable, threading.Lock] = {}

    def get_or_fetch(self, key: Hashable, fetch_fn: Callable[[], T]) -> T:
        # fast path
        if key in self._cache:
            self.logger.debug("Cache hit for %s (key=%s)", self.name, key)
            return self._copier(self._cache[key])  # return a shallow copy

        self.logger.info("Cache miss for %s. Fetching (key=%s)...", self.name, key)
        lock = self._locks.setdefault(key, threading.Lock())

        with lock:
            # check again after acquiring lock
            if key in self._cache:
                self.logger.debug(
                    "Cache filled while waiting for lock for %s (key=%s)", self.name, key
                )
                return self._copier(self._cache[key])

            try:
                data = fetch_fn()
            except Exception:
                self.logger.exception("Error fetching %s (key=%s)", self.name, key)
                raise

            if not self._validator(data):
                msg = f"Fetched {self.name!r} is empty or invalid for key={key!r}"
                self.logger.error(msg)
                raise RuntimeError(msg)

            stored = self._copier(data)
            self._cache[key] = stored
            self.logger.debug("Cached %s (key=%s)", self.name, key)
            return self._copier(stored)
