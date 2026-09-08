from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluation import metrics_from_prediction_file, save_predictions


class EvaluationTests(unittest.TestCase):
    def test_saved_metrics_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            save_predictions(
                path,
                texts=["a", "b", "c"],
                y_true=["Anger", "Other", "Fear"],
                y_pred=["Anger", "Fear", "Fear"],
            )
            metrics = metrics_from_prediction_file(path)
            self.assertEqual(metrics["samples"], 3)
            self.assertAlmostEqual(metrics["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

