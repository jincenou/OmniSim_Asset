"""Check relocation boundaries and preservation of existing files."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "materialize", Path(__file__).resolve().parents[1] / "tools/materialize_workspace.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterializationTests(unittest.TestCase):
    def test_longest_prefix_and_no_cascading_rewrite(self):
        mappings = {"/mnt/ojc": "/tmp/new", "/mnt/ojc/miniconda3/envs": "/opt/envs",
                    "/tmp/new": "/do/not/rewrite"}
        value = "'/mnt/ojc/miniconda3/envs/grid4d' '/mnt/ojc/workplace1/human' '/mnt/ojc2/x'"
        self.assertEqual(MODULE.remap(value, mappings),
                         "'/opt/envs/grid4d' '/tmp/new/workplace1/human' '/mnt/ojc2/x'")

    def test_plan_does_not_write_and_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "entry.py").write_text("ROOT = '/mnt/ojc/workplace1'\n")
            output = root / "output"
            plan = MODULE.make_plan(source, output, {"/mnt/ojc": str(output)})
            self.assertFalse(output.exists())
            self.assertEqual(len(plan), 1)
            output.mkdir()
            (output / "entry.py").write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                MODULE.make_plan(source, output, {"/mnt/ojc": str(output)})
            self.assertEqual((output / "entry.py").read_text(), "do not overwrite")

    def test_symlink_to_asset_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "asset_library").mkdir(parents=True)
            (source / "asset_library" / "entry.py").write_text("x = 1\n")
            output = root / "output"
            output.mkdir()
            assets = root / "protected_assets"
            assets.mkdir()
            (output / "asset_library").symlink_to(assets)
            with self.assertRaises(ValueError):
                MODULE.make_plan(source, output, {"/mnt/ojc": str(output)})
            self.assertEqual(list(assets.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
