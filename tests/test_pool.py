"""基线池: band slot rules, sole protection, margin hysteresis, persistence."""
from __future__ import annotations

import json

import pytest

from seven523.elo import Fit, Rating
from seven523.pool import (
    Pool,
    PoolConfig,
    PoolMember,
    band_of,
    load_pool,
    save_pool,
    update_pool,
)


def member(
    id_: str,
    mu: float,
    *,
    sigma: float = 50.0,
    round_: int = 0,
    parent: str | None = None,
) -> PoolMember:
    return PoolMember(
        id=id_,
        spec=f"ckpt:runs/{id_}/agent.pt",
        mu=mu,
        sigma=sigma,
        round=round_,
        parent=parent,
    )


def test_band_of_boundaries():
    assert band_of(0.0) == 0
    assert band_of(199.999) == 0
    assert band_of(200.0) == 1
    assert band_of(-0.1) == -1
    assert band_of(-200.0) == -1
    assert band_of(400.0, 100.0) == 4


def test_update_under_cap_keeps_everyone():
    members = [member("a", 10.0), member("b", 20.0), member("c", 30.0)]
    update = update_pool(members, [], config=PoolConfig())
    assert [m.id for m in update.kept] == ["a", "b", "c"]
    assert update.evicted == ()
    assert update.added == ()
    assert update.bands == {0: ("c", "b", "a")}


def test_overflow_trims_to_cap_keeping_strongest():
    members = [member(f"m{i}", 10.0 * i) for i in range(6)]  # one band, over cap
    update = update_pool(members, [], config=PoolConfig(band_cap=5))
    assert {m.id for m in update.evicted} == {"m0"}
    assert len(update.kept) == 5
    assert update.bands == {0: ("m5", "m4", "m3", "m2", "m1")}


def test_newcomer_within_margin_does_not_displace():
    members = [member(f"m{i}", 100.0 + 10.0 * i) for i in range(5)]  # 100..140
    newcomer = member("new", 105.0, round_=1)  # weakest incumbent is 100+10=110
    update = update_pool(members, [newcomer], config=PoolConfig(margin=10.0))
    assert update.added == ()
    assert {m.id for m in update.evicted} == {"new"}
    assert {m.id for m in update.kept} == {"m0", "m1", "m2", "m3", "m4"}


def test_newcomer_beyond_margin_displaces_weakest():
    members = [member(f"m{i}", 100.0 + 10.0 * i) for i in range(5)]
    newcomer = member("new", 115.0, round_=1)
    update = update_pool(members, [newcomer], config=PoolConfig(margin=10.0))
    assert update.added == ("new",)
    assert {m.id for m in update.evicted} == {"m0"}
    assert "new" in {m.id for m in update.kept}


def test_sole_incumbent_is_protected_even_when_weakest():
    members = [member("old", 50.0)]  # band 0 has exactly one incumbent
    newcomers = [member(f"n{i}", 60.0 + 10.0 * i, round_=1) for i in range(5)]
    update = update_pool(members, newcomers, config=PoolConfig(band_cap=5))
    assert "old" in {m.id for m in update.kept}
    assert len(update.kept) == 5
    assert {m.id for m in update.evicted} == {"n0"}  # the weakest newcomer


def test_band_floor_keeps_a_sole_newcomer():
    update = update_pool([], [member("solo", 10.0)], config=PoolConfig())
    assert [m.id for m in update.kept] == ["solo"]
    assert update.added == ("solo",)


def test_update_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        update_pool([member("a", 0.0)], [member("a", 1.0)])


def test_update_is_order_independent():
    members = [member(f"m{i}", float(i)) for i in range(6)]
    newcomers = [member("new", 99.0, round_=1)]
    first = update_pool(members, newcomers, config=PoolConfig())
    second = update_pool(
        list(reversed(members)), list(reversed(newcomers)), config=PoolConfig()
    )
    assert [m.id for m in first.kept] == [m.id for m in second.kept]
    assert [m.id for m in first.evicted] == [m.id for m in second.evicted]


def test_update_pool_invariants_fuzz():
    import random

    rng = random.Random(0)
    for _ in range(200):
        config = PoolConfig(
            band_width=rng.choice([50.0, 200.0]),
            band_cap=rng.randint(1, 5),
            margin=rng.choice([0.0, 10.0]),
        )
        members = [member(f"m{i}", rng.uniform(-300.0, 900.0)) for i in range(rng.randint(0, 8))]
        newcomers = [
            member(f"n{i}", rng.uniform(-300.0, 900.0), round_=1)
            for i in range(rng.randint(0, 8))
        ]
        update = update_pool(members, newcomers, config=config)
        universe = {m.id for m in members} | {m.id for m in newcomers}
        kept = {m.id for m in update.kept}
        evicted = {m.id for m in update.evicted}
        assert kept | evicted == universe
        assert kept & evicted == set()

        kept_bands: dict[int, list[str]] = {}
        for m in update.kept:
            kept_bands.setdefault(band_of(m.mu, config.band_width), []).append(m.id)
        assert all(1 <= len(ids) <= config.band_cap for ids in kept_bands.values())
        assert {band: set(ids) for band, ids in kept_bands.items()} == {
            band: set(ids) for band, ids in update.bands.items()
        }

        incumbent_bands: dict[int, list[str]] = {}
        for m in members:
            incumbent_bands.setdefault(band_of(m.mu, config.band_width), []).append(m.id)
        for ids in incumbent_bands.values():
            if len(ids) == 1:
                assert ids[0] in kept  # 独苗 incumbent never evicted


def test_pool_add_and_trim():
    pool = Pool([member("a", 100.0)], config=PoolConfig(band_cap=1))
    pool.add([member("b", 150.0)])
    assert {m.id for m in pool} == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate"):
        pool.add([member("a", 200.0)])
    update = pool.trim()
    assert [m.id for m in update.kept] == ["b"]
    assert {m.id for m in update.evicted} == {"a"}


def test_pool_trim_keeps_the_incumbent_margin():
    pool = Pool(
        [member("old", 100.0), member("old2", 120.0)],
        config=PoolConfig(band_cap=2, margin=10.0),
    )
    pool.add([member("new", 105.0)])
    update = pool.trim(["new"])
    assert {m.id for m in update.kept} == {"old", "old2"}  # 105 < 100 + 10
    assert {m.id for m in update.evicted} == {"new"}
    pool.add([member("new2", 115.0)])
    update = pool.trim(["new2"])
    assert {m.id for m in update.kept} == {"old2", "new2"}  # 115 > 100 + 10


def test_member_validation():
    with pytest.raises(ValueError, match="'@'"):
        member("bad@id", 0.0)
    with pytest.raises(ValueError, match="ckpt"):
        PoolMember(id="a", spec="random", mu=0.0, sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        PoolMember(id="a", spec="ckpt:x", mu=0.0, sigma=0.0)


def test_pool_save_load_roundtrip(tmp_path):
    pool = Pool(
        [member("a", 10.0), member("b", 210.0)],
        config=PoolConfig(band_cap=3),
        round_=2,
    )
    path = save_pool(pool, tmp_path / "pool.json", updated_at="2026-01-01T00:00:00")
    loaded = load_pool(path)
    assert loaded.round == 2
    assert loaded.config == pool.config
    assert {m.id for m in loaded} == {"a", "b"}
    assert loaded.members["a"].mu == 10.0
    assert loaded.members["a"].prior_mu == pool.members["a"].prior_mu


def test_pool_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"schema": 99, "members": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_pool(path)


def test_apply_fit_overwrites_ratings_and_accumulates_games():
    pool = Pool([member("a", 10.0, sigma=50.0, parent="p")])
    pool.members["a"] = PoolMember(
        id="a", spec="ckpt:x", mu=10.0, sigma=50.0, games=100
    )
    fit = Fit(ratings={"a": Rating(mu=110.0, sigma=20.0, n=7)}, games=1)
    pool.apply_fit(fit)
    assert pool.members["a"].mu == 110.0
    assert pool.members["a"].sigma == 20.0
    assert pool.members["a"].games == 107


def test_pool_bands_and_priors():
    pool = Pool([member("a", 10.0), member("b", 150.0), member("c", 250.0, sigma=40.0)])
    assert list(pool.bands()) == [0, 1]
    assert [m.id for m in pool.bands()[0]] == ["b", "a"]
    priors = pool.priors()
    assert priors["c"].mu == 250.0
    assert priors["c"].sigma == 40.0
