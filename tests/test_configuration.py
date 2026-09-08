from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.configuration import ConfigurationError, load_config


class ConfigurationTests(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("task_id: uit_vsmec_7label\nunknown: true\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_native_xlstm_config_loads(self):
        config = load_config(Path(__file__).parents[1] / "configs" / "pilot_xlstm_2block.yaml")
        self.assertEqual(config["model"]["type"], "xlstm_native")
        self.assertEqual(config["training"]["selection_metric"], "weighted_f1")


if __name__ == "__main__":
    unittest.main()
