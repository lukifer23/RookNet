from __future__ import annotations

"""Disk-backed replay buffer based on WebDataset shards.

The buffer streams self-play tuples (board_state: np.ndarray, policy: np.ndarray, value: float)
into sharded .tar files and exposes an iterable WebDataset pipeline for training.
"""

import os
from pathlib import Path
from typing import Iterable, Tuple, Any

import numpy as np

try:
    import webdataset as wds  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("webdataset must be installed (pip install webdataset)") from exc


class StreamingReplayBuffer:
    """Append-only sharded replay buffer.

    Parameters
    ----------
    root_dir: str | Path
        Directory where shards will be written (e.g. data/replay).
    samples_per_shard: int, default 10_000
        Rotate to a new tar shard after this many samples.
    compress: bool, default False
        Whether to gzip shards (adds .gz extension).
    """

    def __init__(self, root_dir: str | Path, *, samples_per_shard: int = 10_000, compress: bool = False) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.compress = compress

        base = self.root / "shard-%06d.tar"
        pattern = str(base) + (".gz" if compress else "")
        self._writer = wds.ShardWriter(pattern, maxcount=samples_per_shard, maxsize=10 * 2**30)  # 10 GiB cap
        self._written = 0

        # Lightweight RAM cache for quick sampling if the dataset is small / training start-up.
        self._ram_cache: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._ram_cache_limit = 100_000  # ≈ <2 GB fp16

        # Track logical *games* added – approximated for now via number of first-moves seen.
        self._games = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extend(self, samples: Iterable[Tuple[np.ndarray, np.ndarray, float]]):
        """Add *samples* (board_state, policy, value) to buffer."""
        for board, policy, value in samples:
            self.add(board, policy, value)

    def add(self, board: np.ndarray, policy: np.ndarray, value: float):
        """Add a single (state, policy, value) tuple to the buffer."""
        key = f"{self._written:09d}"
        self._writer.write({
            "__key__": key,
            "board.npy": board.astype(np.float16),
            "policy.npy": policy.astype(np.float16),
            "value.txt": str(value).encode(),
        })
        self._written += 1

        # Opportunistic store in RAM for small-buffer sampling.
        if len(self._ram_cache) < self._ram_cache_limit:
            self._ram_cache.append((board.astype(np.float32), policy.astype(np.float32), float(value)))

        # Heuristic: treat a terminal value (±1 or 0) as end-of-game marker. Count first appearance per game.
        if value in (-1.0, 0.0, 1.0):
            self._games += 1

    def close(self):
        """Flush and close the underlying shard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # Dataset helper
    # ------------------------------------------------------------------

    def webdataset(self, *, shuffle: int = 10_000, repeat: bool = True, shardshuffle: bool = False):
        """Return a WebDataset pipeline for PyTorch DataLoader."""
        pattern = str(self.root / "shard-*.tar*")  # glob both .tar and .tar.gz
        ds = wds.WebDataset(pattern, resampled=repeat, shardshuffle=shardshuffle).shuffle(shuffle)
        ds = ds.decode().to_tuple("board.npy", "policy.npy", "value.txt")

        # value.txt -> float tensor later; keep raw for now
        return ds

    # ------------------------------------------------------------------
    # Convenience wrappers expected by training loop
    # ------------------------------------------------------------------

    @property
    def total_games_added(self) -> int:  # noqa: D401
        """Return number of *games* added so far (approximate)."""
        return self._games

    # Alias expected by trainer
    def to_webdataset(self, *args, **kwargs):  # noqa: D401
        return self.webdataset(*args, **kwargs)

    def get_dataloader(self, *, batch_size: int = 256, num_workers: int = 2, shuffle: bool = True):
        """Return a PyTorch-compatible DataLoader backed by WebDataset."""
        import torch
        import webdataset as wds  # local import to avoid global dependency for users who don't train

        ds = self.webdataset(shuffle=batch_size * 10, repeat=True, shardshuffle=shuffle)

        def _collate(sample):
            board, policy, value = sample
            board = torch.from_numpy(board.astype(np.float32))
            policy = torch.from_numpy(policy.astype(np.float32))
            value = torch.tensor(float(value.decode()), dtype=torch.float32)
            return board, policy, value

        loader = wds.WebLoader(ds, batch_size=batch_size, num_workers=num_workers)
        loader = loader.map(_collate)
        return loader

    def sample(self, n: int):
        """Uniform random sample *n* triples from RAM cache.

        Returns torch tensors (states, policies, values) or (None, None, None) if
        cache is empty.
        """
        if not self._ram_cache:
            return None, None, None

        import torch
        import random

        n = min(n, len(self._ram_cache))
        batch = random.sample(self._ram_cache, n)
        states, policies, values = zip(*batch)

        states = torch.from_numpy(np.stack(states).astype(np.float32))
        policies = torch.from_numpy(np.stack(policies).astype(np.float32))
        values = torch.tensor(values, dtype=torch.float32)
        return states, policies, values

    def __len__(self):
        return self._written
