"""Regression checks for the controlled noise-strength sweep."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments import FEATS, noise_strength_cases
from noise_sweep_data import generate_noise_sweep


LEVELS = (0.5, 1.0, 2.0, 4.0)
IDENTITY = ["circuit_id", "family", "depth", "observable", "obs_weight", "n_1q", "n_2q", "ideal"]
EXACT = [f"exact_noisy_l{scale}" for scale in (1, 2, 3)]
SAMPLED = [f"noisy_l{scale}" for scale in (1, 2, 3)]


def read_dataset(directory, level):
    path = directory / f"ds_noise{level}x.csv"
    return pd.read_csv(path, float_precision="round_trip")


def read_metadata(directory, level):
    path = directory / f"ds_noise{level}x_meta.json"
    return json.loads(path.read_text())


class NoiseStrengthSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.first = Path(scratch.name) / "first"
        cls.second = Path(scratch.name) / "second"
        for directory, shot_seed in ((cls.first, 101), (cls.second, 102)):
            generate_noise_sweep(directory, per_family=5, shots=257,
                                 circuit_seed=77, shot_seed=shot_seed)

    def test_circuit_pool_and_exact_values_are_independent_of_shot_seed(self):
        reference = read_dataset(self.first, 1.0)
        expected_metadata = read_metadata(self.first, 1.0)
        ideal = reference.ideal.to_numpy(dtype="<f8")
        self.assertEqual(expected_metadata["ideal_signature"], sha256(ideal.tobytes()).hexdigest())
        digest = sha256()
        circuit_rng = np.random.default_rng(expected_metadata["circuit_seed"])
        wires = expected_metadata["wires"]
        circuits = reference[["circuit_id", "family", "depth"]].drop_duplicates()
        for circuit_id, family, depth in circuits.itertuples(index=False, name=None):
            angles = circuit_rng.uniform(-np.pi, np.pi, size=(depth, wires, 2))
            digest.update(np.array([circuit_id, depth, wires], dtype="<u8").tobytes())
            digest.update(family.encode("ascii") + b"\0")
            digest.update(angles.astype("<f8").tobytes())
        self.assertEqual(expected_metadata["circuit_signature"], digest.hexdigest())

        child_streams = set()
        for level in LEVELS:
            with self.subTest(level=level):
                first, second = (read_dataset(directory, level) for directory in (self.first, self.second))
                meta = read_metadata(self.first, level)
                meta_other = read_metadata(self.second, level)

                self.assertEqual(meta["noise_mult"], level)
                self.assertEqual(meta["n_circuits"], 10)
                self.assertEqual(meta["n_rows"], 50)
                self.assertEqual(meta["circuit_seed"], meta_other["circuit_seed"])
                self.assertNotEqual(meta["shot_seed"], meta_other["shot_seed"])
                for field in ("circuit_signature", "ideal_signature"):
                    self.assertEqual(meta[field], expected_metadata[field])
                    self.assertEqual(meta[field], meta_other[field])
                stream_key = tuple(meta["shot_spawn_key"])
                child_streams.add(stream_key)

                pd.testing.assert_frame_equal(first[IDENTITY], reference[IDENTITY], check_exact=True)
                pd.testing.assert_frame_equal(first[IDENTITY + EXACT], second[IDENTITY + EXACT],
                                              check_exact=True)
                shots = meta["shots"]
                shot_rng = np.random.default_rng(np.random.SeedSequence(meta["shot_seed"], spawn_key=stream_key))
                probabilities = np.clip((1 + first[EXACT].to_numpy()) / 2, 0, 1)
                replay = [[2.0 * shot_rng.binomial(shots, p) / shots - 1.0 for p in row]
                          for row in probabilities]
                assert_array_equal(first[SAMPLED].to_numpy(), replay)
                self.assertFalse(first[SAMPLED].equals(second[SAMPLED]))
                if level != 1.0:
                    self.assertFalse(first[EXACT].equals(reference[EXACT]))
        self.assertEqual(len(child_streams), len(LEVELS))

    def test_four_conditions_use_the_same_grouped_split_and_reduced_budget(self):
        cases = list(noise_strength_cases(directory=self.first))
        self.assertEqual([case.config for case in cases], [f"{level}x" for level in LEVELS])
        reference = cases[0]
        for case, level in zip(cases, LEVELS, strict=True):
            with self.subTest(level=level):
                self.assertEqual(case.experiment, "E1_noise_strength")
                self.assertTrue(case.fast)
                self.assertEqual(tuple(case.feats), tuple(FEATS))
                self.assertEqual(case.arms, reference.arms)
                assert_array_equal(case.train, reference.train)
                assert_array_equal(case.test, reference.test)
                assert_array_equal(np.sort(np.r_[case.train, case.test]), np.arange(len(case.df)))
                train_groups = case.df.iloc[case.train].circuit_id.to_numpy()
                test_groups = case.df.iloc[case.test].circuit_id.to_numpy()
                self.assertEqual((len(np.unique(train_groups)), len(np.unique(test_groups))), (7, 3))
                self.assertEqual(len(np.intersect1d(train_groups, test_groups)), 0)
                pd.testing.assert_frame_equal(case.df[IDENTITY + SAMPLED],
                                              read_dataset(self.first, level)[IDENTITY + SAMPLED],
                                              check_exact=True)
                for train, valid in GroupKFold(n_splits=4).split(train_groups, groups=train_groups):
                    self.assertEqual(len(np.intersect1d(train_groups[train], train_groups[valid])), 0)

    def test_misaligned_targets_are_rejected(self):
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for path in self.first.iterdir():
                shutil.copy2(path, directory)
            corrupted = read_dataset(directory, 2.0)
            corrupted.loc[0, "ideal"] += 0.1
            corrupted.to_csv(directory / "ds_noise2.0x.csv", index=False)
            with self.assertRaises(ValueError):
                list(noise_strength_cases(directory=directory))

    def test_missing_base_condition_does_not_fall_back_to_primary_benchmark(self):
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            for path in self.first.iterdir():
                shutil.copy2(path, directory)
            (directory / "ds_noise1.0x.csv").rename(directory / "benchmark_data.csv")
            with self.assertRaises(FileNotFoundError):
                list(noise_strength_cases(directory=directory))


if __name__ == "__main__":
    unittest.main()
