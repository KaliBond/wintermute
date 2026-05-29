from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.camcorp5_pipeline.config import PipelineConfig
from src.camcorp5_pipeline.pipeline import run_pipeline


class Camcorp5PipelineTest(unittest.TestCase):
    def test_pipeline_exports_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mean = pd.DataFrame(
                [
                    ["Example Corp", 2020, "Helm",     4.0, 6.0, 3.0, 5.0, 7.0, 12.0],
                    ["Example Corp", 2020, "Shield",   5.0, 7.0, 2.0, 4.0, 8.0, 14.0],
                    ["Example Corp", 2020, "Lore",     5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                    ["Example Corp", 2020, "Archive",  5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                    ["Example Corp", 2020, "Stewards", 5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                    ["Example Corp", 2020, "Craft",    5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                    ["Example Corp", 2020, "Hands",    5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                    ["Example Corp", 2020, "Flow",     5.0, 5.0, 3.0, 4.0, 7.5, 13.0],
                ],
                columns=[
                    "Entity",
                    "Year",
                    "Node",
                    "Coherence",
                    "Capacity",
                    "Stress",
                    "Abstraction",
                    "Node Value",
                    "Bond Strength",
                ],
            )
            envelope = pd.DataFrame(
                [
                    ["Example Corp", 2020, "Helm",     0.1, 0.2, 0.3, 0.4, 1.0, 6.5, 7.5],
                    ["Example Corp", 2020, "Shield",   0.2, 0.3, 0.4, 0.5, 0.0, 8.0, 8.0],
                    ["Example Corp", 2020, "Lore",     0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                    ["Example Corp", 2020, "Archive",  0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                    ["Example Corp", 2020, "Stewards", 0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                    ["Example Corp", 2020, "Craft",    0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                    ["Example Corp", 2020, "Hands",    0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                    ["Example Corp", 2020, "Flow",     0.2, 0.2, 0.3, 0.3, 1.0, 7.0, 8.0],
                ],
                columns=[
                    "Entity",
                    "Year",
                    "Node",
                    "C_sd",
                    "K_sd",
                    "S_sd",
                    "A_sd",
                    "V_range",
                    "V_min",
                    "V_max",
                ],
            )

            mean_path = tmp_path / "mean.csv"
            envelope_path = tmp_path / "envelope.csv"
            output_dir = tmp_path / "out"
            mean.to_csv(mean_path, index=False)
            envelope.to_csv(envelope_path, index=False)

            outputs = run_pipeline(
                PipelineConfig(
                    ensemble_mean_csv=mean_path,
                    envelope_csv=envelope_path,
                    output_dir=output_dir,
                    entity="Example Corp",
                )
            )

            self.assertIn("node_panel", outputs)
            self.assertIn("yearly_summary", outputs)
            self.assertIn("node_rankings", outputs)
            self.assertIn("corp_v02_edews", outputs)
            self.assertIn("corp_v02_crisis_score", outputs)
            self.assertIn("corp_v02_alarm_protocol", outputs)
            self.assertTrue(all(path.exists() for path in outputs.values()))

            summary = pd.read_csv(outputs["yearly_summary"])
            self.assertEqual(summary.loc[0, "node_count"], 8)
            self.assertEqual(summary.loc[0, "mean_node_value"], 7.5)


if __name__ == "__main__":
    unittest.main()

