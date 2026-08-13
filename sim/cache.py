"""Block-granular KV pool with a prefix cache and pluggable eviction.

Model
-----
The pool holds `capacity` fixed-size blocks. A block is resident or absent; there is no
CPU offload tier. A resident block is either *held* (ref_count > 0, some running request
needs it) or *evictable* (ref_count == 0, it survives only as cache).

Reuse follows vLLM's prefix-cache semantics:
  * only a **full** block can be reused, and only once it has been **computed**;
  * lookup walks the block chain from the start and stops at the first miss, so a
    surviving block whose ancestor was evicted is not reusable;
  * blocks inside the shared system prompt carry a shared key, so every session hits
    them; every block past the system prompt is session-private.

Two eviction families
---------------------
"ttl"      Continuum-shaped. A block released at time t is protected until t + TTL.
           Eviction drains expired blocks (oldest release first) before touching
           protected ones (also oldest first). TTL = 0 for everything makes this
           exactly LRU, which is what vLLM does today.

"priority" Belady-shaped. A block released at time t carries a rank; eviction takes the
           lowest rank first. The rank is a `(tier, key)` pair so that a policy can say
           "this block will never be used again" (tier 0) separately from ordering the
           rest (tier 1), which is what lets the experiment separate the value of
           knowing a session has *ended* from the value of knowing *when* it returns.
           Fed the true next-use time this is Belady's rule -- but NOT a bound here:
           optimality assumes a fixed offline reference stream, and in this system
           eviction changes when work is recomputed. Fed a predicted next-use time it is
           the deployable version of the same rule, and EXP04 shows that version loses
           badly to LRU unless the prediction is confident.

The two families are separate mechanisms on purpose. A uniform TTL cannot express
"evict the session that comes back last", and furthest-in-future cannot express
"pin this block no matter what". Which mechanism a policy uses is part of what the
experiment measures, alongside the quality of the information fed into it.

Deviations from a real engine, all of which apply identically to every policy arm and
so cannot bias a policy comparison:
  * among expired blocks the order is by expiry time rather than by last use (identical
    whenever TTLs are uniform, which covers the `lru` and `const_ttl` arms exactly);
  * a block computed by a still-running request becomes reusable by another request as
    soon as its own prefill step completes, rather than at sub-step granularity.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass


BlockKey = tuple

TTL_FAMILY = "ttl"
PRIORITY_FAMILY = "priority"


@dataclass
class Block:
    key: BlockKey
    ref: int = 0
    full: bool = False
    ready: bool = False
    last_used: float = 0.0
    protected_until: float = 0.0
    stamp: int = 0  # invalidates stale heap entries


class BlockPool:
    def __init__(self, capacity: int, block_size: int, enable_prefix_caching: bool = True,
                 family: str = TTL_FAMILY):
        if family not in (TTL_FAMILY, PRIORITY_FAMILY):
            raise ValueError(f"unknown eviction family: {family}")
        self.capacity = capacity
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.family = family

        self.index: dict[BlockKey, Block] = {}

        # ttl family
        self._expired: OrderedDict[BlockKey, None] = OrderedDict()
        self._protected: OrderedDict[BlockKey, None] = OrderedDict()
        self._expiry_heap: list[tuple[float, int, BlockKey]] = []

        # priority family: min-heap on eviction rank (lowest rank leaves first)
        self._prio_heap: list[tuple[tuple[float, float], int, BlockKey]] = []
        self._evictable: dict[BlockKey, None] = {}

        self._stamp = 0
        self.n_evictions = 0
        self.n_protected_evictions = 0

    # ---------------------------------------------------------------- capacity

    @property
    def n_resident(self) -> int:
        return len(self.index)

    @property
    def n_free(self) -> int:
        return self.capacity - len(self.index)

    @property
    def n_evictable(self) -> int:
        if self.family == TTL_FAMILY:
            return len(self._expired) + len(self._protected)
        return len(self._evictable)

    def utilization(self) -> float:
        return len(self.index) / self.capacity if self.capacity else 0.0

    # ------------------------------------------------------------------ expiry

    def _reap(self, now: float) -> None:
        """Move blocks whose protection has lapsed into the expired class."""
        heap = self._expiry_heap
        while heap and heap[0][0] <= now:
            _, stamp, key = heapq.heappop(heap)
            block = self.index.get(key)
            if block is None or block.stamp != stamp or block.ref > 0:
                continue
            if key in self._protected:
                del self._protected[key]
                self._expired[key] = None

    # -------------------------------------------------------------- allocation

    def _evict(self, n: int, now: float) -> bool:
        """Make room for `n` more blocks. Returns False if impossible."""
        if n <= self.n_free:
            return True
        need = n - self.n_free
        if need > self.n_evictable:
            return False
        if self.family == TTL_FAMILY:
            self._reap(now)
            for source, is_protected in ((self._expired, False), (self._protected, True)):
                while need > 0 and source:
                    key, _ = source.popitem(last=False)
                    del self.index[key]
                    need -= 1
                    self.n_evictions += 1
                    if is_protected:
                        self.n_protected_evictions += 1
        else:
            while need > 0 and self._prio_heap:
                _, stamp, key = heapq.heappop(self._prio_heap)
                block = self.index.get(key)
                if block is None or block.stamp != stamp or block.ref > 0:
                    continue  # stale entry
                del self.index[key]
                del self._evictable[key]
                need -= 1
                self.n_evictions += 1
        return need <= 0

    def lookup_prefix(self, keys: list[BlockKey]) -> int:
        """Number of leading blocks that are resident, full and computed."""
        if not self.enable_prefix_caching:
            return 0
        hits = 0
        for key in keys:
            block = self.index.get(key)
            if block is None or not block.full or not block.ready:
                break
            hits += 1
        return hits

    def acquire(self, hit_keys: list[BlockKey], new_keys: list[BlockKey], now: float) -> bool:
        """Reference `hit_keys` and materialise `new_keys`. All-or-nothing.

        Feasibility is checked *before* anything is referenced, because a failed
        acquire must leave the pool byte-identical -- rolling a reference back would
        lose the block's eviction rank and silently corrupt the policy under test.
        """
        if self.family == TTL_FAMILY:
            self._reap(now)
        # Hits are currently evictable, so they cannot also count as room for the misses.
        hits_in_evictable = sum(1 for k in hit_keys
                                if (b := self.index.get(k)) is not None and b.ref == 0)
        if len(new_keys) > self.n_free + self.n_evictable - hits_in_evictable:
            return False

        for key in hit_keys:
            self._hold(key)
        if not self._evict(len(new_keys), now):  # guaranteed by the check above
            raise AssertionError("eviction failed after a positive feasibility check")

        for key in new_keys:
            if key in self.index:
                # A concurrent request of the same session already materialised it.
                self._hold(key)
                continue
            self.index[key] = Block(key=key, ref=1, full=False, ready=False, last_used=now)
        return True

    def _hold(self, key: BlockKey) -> None:
        block = self.index[key]
        if block.ref == 0:
            self._expired.pop(key, None)
            self._protected.pop(key, None)
            self._evictable.pop(key, None)
            self._stamp += 1
            block.stamp = self._stamp  # invalidate any heap entry pointing at it
        block.ref += 1

    def _release_one(self, key: BlockKey, now: float, evict_rank,
                     cacheable: bool = True) -> None:
        """Drop one reference.

        `evict_rank` is interpreted by the family:
          ttl      -> float, the TTL in seconds (protection window from `now`)
          priority -> (tier, key) tuple, lowest leaves first
        """
        block = self.index.get(key)
        if block is None:
            return
        block.ref -= 1
        if block.ref > 0:
            return
        block.ref = 0
        block.last_used = now
        if not cacheable or not block.full or not block.ready:
            # A partial or never-computed tail block has no reusable identity.
            del self.index[key]
            return
        self._stamp += 1
        block.stamp = self._stamp
        if self.family == TTL_FAMILY:
            ttl = max(0.0, evict_rank)
            block.protected_until = now + ttl
            if ttl <= 0.0:
                self._expired[key] = None
            else:
                self._protected[key] = None
                heapq.heappush(self._expiry_heap, (block.protected_until, block.stamp, key))
        else:
            self._evictable[key] = None
            heapq.heappush(self._prio_heap, (evict_rank, block.stamp, key))

    def release(self, keys: list[BlockKey], now: float, evict_rank,
                cacheable: bool = True) -> None:
        for key in keys:
            self._release_one(key, now, evict_rank, cacheable)

    def drop_rank(self):
        """Rank meaning 'evict this before anything else', for the active family."""
        return 0.0 if self.family == TTL_FAMILY else (-1.0, 0.0)

    def mark_full(self, key: BlockKey) -> None:
        block = self.index.get(key)
        if block is not None:
            block.full = True

    def mark_ready(self, keys: list[BlockKey]) -> None:
        for key in keys:
            block = self.index.get(key)
            if block is not None:
                block.ready = True


def block_keys(session_id: int, n_blocks: int, shared_blocks: int) -> list[BlockKey]:
    """Key chain for a session's context. Blocks inside the system prompt are shared."""
    keys: list[BlockKey] = []
    for i in range(n_blocks):
        if i < shared_blocks:
            keys.append(("sys", i))
        else:
            keys.append((session_id, i))
    return keys
