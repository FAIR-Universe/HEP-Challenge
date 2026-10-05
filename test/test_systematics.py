#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion_program"
sys.path.insert(0, str(INGESTION_DIR))

import systematics as systematics_module


class TestNormalizationSystematics(unittest.TestCase):
    def test_normalization_scales_are_applied(self):
        data_set = {
            "data": pd.DataFrame({"event": [0, 1, 2, 3]}),
            "weights": np.ones(4, dtype=float),
            "labels": np.array([1, 0, 0, 0]),
            "detailed_labels": np.array(
                ["htautau", "ztautau", "ttbar", "diboson"]
            ),
        }

        with patch.object(
            systematics_module,
            "mom4_manipulate",
            side_effect=lambda data, **kwargs: data,
        ), patch.object(
            systematics_module,
            "DER_data",
            side_effect=lambda data: data,
        ):
            shifted = systematics_module.systematics(
                data_set,
                ttbar_scale=2.0,
                diboson_scale=3.0,
                bkg_scale=4.0,
                dopostprocess=False,
            )

        np.testing.assert_allclose(
            shifted["weights"],
            np.array([1.0, 4.0, 8.0, 12.0]),
        )

    def test_make_unweighted_set_uses_detailed_labels(self):
        data_set = {
            "data": pd.DataFrame({"event": [0, 1, 2, 3]}),
            "detailed_labels": np.array(
                ["htautau", "ztautau", "ttbar", "diboson"]
            ),
        }

        split = systematics_module.make_unweighted_set(data_set)

        self.assertEqual(split["htautau"]["event"].tolist(), [0])
        self.assertEqual(split["ztautau"]["event"].tolist(), [1])
        self.assertEqual(split["ttbar"]["event"].tolist(), [2])
        self.assertEqual(split["diboson"]["event"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()
