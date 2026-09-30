"""基线池 (bootstrap pool): the rated model archive behind the bootstrap arena.

The arena grows a pool of checkpoints by forking, training, qualifying and
trimming.  This module owns the pool state and its slot rules; it is pure
in-process data (no torch, no network) except for the explicit
:func:`save_pool` / :func:`load_pool` JSON seam.

Slot rules (the arena's niche archive):

* every member is bucketed by its rating into a ``band_width``-wide band,
  ``band_of(mu) = floor(mu / band_width)`` with the pinned RandomBot gauge at
  ``mu = 0`` (ADR-0012), so ``[0, 200)`` is band 0 and ``[-200, 0)`` is band -1;
* a band never holds more than ``band_cap`` members;
* a band never loses its last representative: an incumbent that is the *only
  incumbent* of its band is protected from eviction in that update (the
  "独苗" rule), and a band with no incumbent keeps a sole newcomer.  The floor
  is one member per occupied band.

When a band overflows, survivors are picked by *adjusted* mu: incumbents get a
``margin`` boost so a fresh snapshot only displaces one when it is clearly
better (the arena's noise floor); ties break towards the newer round and then
the lower id.  The merge is deterministic: the same inputs always yield the
same kept/evicted split regardless of input order.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .elo import DEFAULT_MU, DEFAULT_SIGMA, Fit, Prior

__all__ = [
    "DEFAULT_POOL_CONFIG",
    "POOL_SCHEMA",
    "Pool",
    "PoolConfig",
    "PoolMember",
    "PoolUpdate",
    "band_of",
    "load_pool",
    "save_pool",
    "update_pool",
]

#: Version of the ``pool.json`` document written by :func:`save_pool`.
POOL_SCHEMA = 1


def band_of(mu: float, band_width: float = 200.0) -> int:
    """The band index of ``mu``; floor division, so ``[-200, 0)`` is band -1."""
    return math.floor(mu / band_width)


@dataclass(frozen=True, slots=True)
class PoolMember:
    """One rated checkpoint in the pool.

    ``mu``/``sigma`` are the latest fitted rating; ``games`` is the cumulative
    number of rated 牌局 behind it.  ``prior_mu``/``prior_sigma`` are the prior
    the member entered its rating chain with (the parent's rating for forks),
    which is what a full-history replay needs to reproduce the chain.
    """

    id: str
    spec: str
    mu: float
    sigma: float
    games: int = 0
    parent: str | None = None
    seed: int | None = None
    round: int = 0
    step: int | None = None
    prior_mu: float = DEFAULT_MU
    prior_sigma: float = DEFAULT_SIGMA

    def __post_init__(self) -> None:
        if not self.id or "@" in self.id:
            raise ValueError(
                f"pool member id must be non-empty and free of '@': {self.id!r}"
            )
        if not self.spec.startswith("ckpt:"):
            raise ValueError(
                f"pool member {self.id!r} spec must be 'ckpt:<path>', "
                f"got {self.spec!r}"
            )
        for name in ("mu", "sigma", "prior_mu", "prior_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"pool member {self.id!r} {name} must be finite")
        if self.sigma <= 0.0 or self.prior_sigma <= 0.0:
            raise ValueError(f"pool member {self.id!r} sigma must be positive")
        if self.games < 0:
            raise ValueError(f"pool member {self.id!r} games must be non-negative")
        if self.round < 0:
            raise ValueError(f"pool member {self.id!r} round must be non-negative")

    @property
    def ckpt(self) -> str:
        """The checkpoint path behind :attr:`spec`."""
        return self.spec[len("ckpt:") :]

    def prior(self) -> Prior:
        """The prior this member entered its rating chain with."""
        return Prior(mu=self.prior_mu, sigma=self.prior_sigma)

    def to_document(self) -> dict:
        return {
            "id": self.id,
            "spec": self.spec,
            "mu": self.mu,
            "sigma": self.sigma,
            "games": self.games,
            "parent": self.parent,
            "seed": self.seed,
            "round": self.round,
            "step": self.step,
            "prior_mu": self.prior_mu,
            "prior_sigma": self.prior_sigma,
        }

    @classmethod
    def from_document(cls, document: Mapping) -> PoolMember:
        return cls(
            id=str(document["id"]),
            spec=str(document["spec"]),
            mu=float(document["mu"]),
            sigma=float(document["sigma"]),
            games=int(document.get("games", 0)),
            parent=document.get("parent"),
            seed=document.get("seed"),
            round=int(document.get("round", 0)),
            step=document.get("step"),
            prior_mu=float(document.get("prior_mu", DEFAULT_MU)),
            prior_sigma=float(document.get("prior_sigma", DEFAULT_SIGMA)),
        )


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """The band rules: width, capacity and the incumbent displacement margin."""

    band_width: float = 200.0
    band_cap: int = 5
    margin: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.band_width) or self.band_width <= 0.0:
            raise ValueError(f"band_width must be positive and finite: {self.band_width!r}")
        if self.band_cap < 1:
            raise ValueError(f"band_cap must be at least 1, got {self.band_cap!r}")
        if not math.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError(f"margin must be non-negative and finite: {self.margin!r}")


#: The band rules used when a caller does not inject its own.
DEFAULT_POOL_CONFIG = PoolConfig()


@dataclass(frozen=True, slots=True)
class PoolUpdate:
    """The outcome of one :func:`update_pool` merge."""

    kept: tuple[PoolMember, ...]
    evicted: tuple[PoolMember, ...]
    added: tuple[str, ...]
    bands: Mapping[int, tuple[str, ...]]


def _merge_by_id(
    members: Sequence[PoolMember], newcomers: Sequence[PoolMember]
) -> dict[str, PoolMember]:
    merged: dict[str, PoolMember] = {}
    for member in (*members, *newcomers):
        if member.id in merged:
            raise ValueError(f"duplicate pool member id {member.id!r}")
        merged[member.id] = member
    return merged


def _protected_ids(
    members: Sequence[PoolMember],
    newcomers: Sequence[PoolMember],
    config: PoolConfig,
) -> frozenset[str]:
    """Incumbent-band-floor protection ("独苗").

    An incumbent that is the only incumbent of its band is protected for this
    update; a band with no incumbent at all protects its sole newcomer so the
    band keeps a floor of one.
    """
    incumbent_counts = Counter(band_of(m.mu, config.band_width) for m in members)
    protected = {
        m.id
        for m in members
        if incumbent_counts[band_of(m.mu, config.band_width)] == 1
    }
    newcomer_counts = Counter(band_of(m.mu, config.band_width) for m in newcomers)
    for member in newcomers:
        band = band_of(member.mu, config.band_width)
        if incumbent_counts[band] == 0 and newcomer_counts[band] == 1:
            protected.add(member.id)
    return frozenset(protected)


def _keep_key(
    member: PoolMember, incumbents: frozenset[str], config: PoolConfig
) -> tuple[float, int, str]:
    """Ascending winner order: adjusted mu desc, newer round first, id asc."""
    boost = config.margin if member.id in incumbents else 0.0
    return (-(member.mu + boost), -member.round, member.id)


def update_pool(
    members: Sequence[PoolMember],
    newcomers: Sequence[PoolMember],
    *,
    config: PoolConfig = DEFAULT_POOL_CONFIG,
) -> PoolUpdate:
    """Merge ``newcomers`` into ``members``, enforcing the band slot rules.

    Pure and deterministic.  Raises ``ValueError`` for a duplicate id in
    ``members + newcomers`` or an invalid ``config``.
    """
    members = tuple(members)
    newcomers = tuple(newcomers)
    if not isinstance(config, PoolConfig):
        raise ValueError(f"config must be a PoolConfig, got {config!r}")
    merged = _merge_by_id(members, newcomers)
    incumbents = frozenset(member.id for member in members)
    protected = _protected_ids(members, newcomers, config)

    groups: dict[int, list[PoolMember]] = {}
    for member in merged.values():
        groups.setdefault(band_of(member.mu, config.band_width), []).append(member)

    kept: list[PoolMember] = []
    evicted: list[PoolMember] = []
    bands: dict[int, tuple[str, ...]] = {}
    for band in sorted(groups):
        group = groups[band]
        ranked = sorted(
            group,
            key=lambda member: (
                0 if member.id in protected else 1,
                _keep_key(member, incumbents, config),
            ),
        )
        survivors = ranked[: config.band_cap]
        kept.extend(survivors)
        evicted.extend(ranked[config.band_cap :])
        bands[band] = tuple(
            member.id for member in sorted(survivors, key=lambda m: (-m.mu, m.id))
        )

    kept_ids = {member.id for member in kept}
    added = tuple(sorted(m.id for m in newcomers if m.id in kept_ids))
    kept.sort(key=lambda member: member.id)
    evicted.sort(key=lambda member: member.id)
    return PoolUpdate(tuple(kept), tuple(evicted), added, bands)


class Pool:
    """Mutable pool state: the members, the band config and the round counter."""

    def __init__(
        self,
        members: Iterable[PoolMember] = (),
        *,
        config: PoolConfig = DEFAULT_POOL_CONFIG,
        round_: int = 0,
    ) -> None:
        self.config = config
        self.round = int(round_)
        self.members: dict[str, PoolMember] = {}
        for member in members:
            if member.id in self.members:
                raise ValueError(f"duplicate pool member id {member.id!r}")
            self.members[member.id] = member

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members.values())

    def sorted_members(self) -> list[PoolMember]:
        """Members strongest first (ties by id) — the display / fork order."""
        return sorted(self.members.values(), key=lambda member: (-member.mu, member.id))

    def bands(self) -> dict[int, tuple[PoolMember, ...]]:
        """Occupied bands, lowest first; members within a band strongest first."""
        groups: dict[int, list[PoolMember]] = {}
        for member in self.members.values():
            groups.setdefault(band_of(member.mu, self.config.band_width), []).append(member)
        return {
            band: tuple(sorted(group, key=lambda m: (-m.mu, m.id)))
            for band, group in sorted(groups.items())
        }

    def priors(self) -> dict[str, Prior]:
        """The current rating of every member as the next fit's prior."""
        return {
            member.id: Prior(mu=member.mu, sigma=member.sigma)
            for member in self.members.values()
        }

    def add(self, members: Iterable[PoolMember]) -> None:
        """Insert members without trimming; call :meth:`trim` afterwards."""
        for member in members:
            if member.id in self.members:
                raise ValueError(f"duplicate pool member id {member.id!r}")
            self.members[member.id] = member

    def trim(
        self, newcomer_ids: Collection[str] = ()
    ) -> PoolUpdate:
        """Re-apply the band rules to the current members (in place).

        ``newcomer_ids`` are the members that joined this update: they skip the
        incumbent ``margin`` boost, every other member gets it.
        """
        newcomer_set = set(newcomer_ids)
        newcomers = [
            self.members[member_id]
            for member_id in sorted(newcomer_set)
            if member_id in self.members
        ]
        incumbents = [
            member for member in self.members.values() if member.id not in newcomer_set
        ]
        update = update_pool(incumbents, newcomers, config=self.config)
        self.members = {member.id: member for member in update.kept}
        return update

    def apply_fit(self, fit: Fit) -> None:
        """Overwrite ratings from ``fit`` and add each member's new game count."""
        for member_id, member in list(self.members.items()):
            rating = fit.ratings.get(member_id)
            if rating is None:
                continue
            self.members[member_id] = replace(
                member,
                mu=rating.mu,
                sigma=rating.sigma,
                games=member.games + rating.n,
            )

    def to_document(self) -> dict:
        return {
            "schema": POOL_SCHEMA,
            "round": self.round,
            "config": {
                "band_width": self.config.band_width,
                "band_cap": self.config.band_cap,
                "margin": self.config.margin,
            },
            "members": [
                member.to_document()
                for member in sorted(self.members.values(), key=lambda m: m.id)
            ],
        }

    @classmethod
    def from_document(cls, document: Mapping) -> Pool:
        schema = document.get("schema")
        if schema != POOL_SCHEMA:
            raise ValueError(
                f"unsupported pool schema {schema!r}; expected {POOL_SCHEMA}"
            )
        config = PoolConfig(**dict(document.get("config", {})))
        members = [
            PoolMember.from_document(entry) for entry in document.get("members", [])
        ]
        return cls(members, config=config, round_=int(document.get("round", 0)))


def save_pool(pool: Pool, path: str | Path, *, updated_at: str | None = None) -> Path:
    """Write ``pool.json`` atomically (temp file + replace) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pool.to_document()
    document["updated_at"] = updated_at or datetime.now().isoformat(timespec="seconds")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)
    return path


def load_pool(path: str | Path) -> Pool:
    """Read a ``pool.json`` written by :func:`save_pool`."""
    return Pool.from_document(json.loads(Path(path).read_text(encoding="utf-8")))
