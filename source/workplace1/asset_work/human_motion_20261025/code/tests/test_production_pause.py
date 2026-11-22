"""Pause new admissions without interrupting an admitted asset's export."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


class ProductionPauseTests(unittest.TestCase):
    def test_drain_ids_and_future_queue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)):
            self.assertFalse(common.production_paused("H0001"))
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            self.assertFalse(common.production_paused("H0001"))
            self.assertTrue(common.production_paused("H0002"))
            self.assertTrue(common.production_paused())

    def test_old_controller_child_waits_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "frontend.py"), "generate", "H0002"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            def release(_):
                common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": False})
            with patch.object(common.time, "sleep", side_effect=release) as sleep:
                common.wait_for_production_permission()
                self.assertEqual(sleep.call_count, 1)
            receipt = next((common.ROOT / "jobs").glob("pause_wait_*.json"))
            self.assertEqual(common.read(receipt)["state"], "resumed")

    def test_admitted_child_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "batch.py"), "package", "H0001"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            with patch.object(common.time, "sleep") as sleep:
                common.wait_for_production_permission()
                sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
