from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import build_vocabulary
from src.preprocessing import normalize_split


class PreprocessingTests(unittest.TestCase):
    def test_missing_text_does_not_become_nan_token(self):
        frame = pd.DataFrame(
            {"Sentence": [np.nan, "Ko sao"], "Emotion": ["Other", "Enjoyment"]}
        )
        normalized = normalize_split(frame, mode="light")
        self.assertEqual(normalized["text"].tolist(), ["không sao"])

    def test_vocabulary_is_deterministic(self):
        left = build_vocabulary(["b a", "a c"], min_frequency=1)
        right = build_vocabulary(["a c", "b a"], min_frequency=1)
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

