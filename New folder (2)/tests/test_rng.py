"""Tests for RNG factory: determinism, independence, spawning."""

from bioventure.rng import RNGFactory


class TestDeterminism:
    def test_same_seed_same_stream(self):
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(42).spawn_one()
        assert r1.random() == r2.random()

    def test_different_seed_different_stream(self):
        r1 = RNGFactory(42).spawn_one()
        r2 = RNGFactory(99).spawn_one()
        assert r1.random() != r2.random()


class TestSpawning:
    def test_spawn_count(self, rng_factory):
        streams = rng_factory.spawn(5)
        assert len(streams) == 5

    def test_spawn_one(self, rng_factory):
        rng = rng_factory.spawn_one()
        val = rng.random()
        assert 0.0 <= val < 1.0

    def test_spawned_streams_independent(self, rng_factory):
        streams = rng_factory.spawn(3)
        values = [s.random() for s in streams]
        assert len(set(values)) == 3

    def test_successive_spawns_independent(self):
        factory = RNGFactory(42)
        first = factory.spawn_one()
        second = factory.spawn_one()
        assert first.random() != second.random()


class TestSeedProperty:
    def test_seed_preserved(self):
        factory = RNGFactory(42)
        assert factory.seed == 42
