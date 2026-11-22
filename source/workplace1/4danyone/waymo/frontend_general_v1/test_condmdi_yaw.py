"""Verify replacement of model root yaw without changing local limb motion."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/CondMDI"))
from data_loaders.humanml.scripts.motion_process import recover_from_ric

rng = np.random.default_rng(42)
features = np.zeros((8, 263), dtype=np.float32)
features[:, 0] = np.linspace(-.8, .9, 8)
features[:, 1:4] = rng.normal(size=(8, 3))
features[:, 4:67] = rng.normal(size=(8, 63))
original = recover_from_ric(torch.tensor(features), 22, abs_3d=True).numpy()
changed = features.copy()
changed[:, 0] += .31
changed[:, 1:3] += 2
changed[:, 3] += .4
predicted = recover_from_ric(torch.tensor(changed), 22, abs_3d=True).numpy()
# Relative vertical coordinates must follow the translated root too.
changed[:, 4:67].reshape(8,21,3)[:,:,1] += .4
predicted = recover_from_ric(torch.tensor(changed), 22, abs_3d=True).numpy()
delta = 2 * (changed[:, 0] - features[:, 0])
corrected = np.einsum("tij,tkj->tki", Rotation.from_euler("y", delta[:, None]).as_matrix(),
                      predicted - predicted[:, :1])
error = float(np.abs(corrected - (original - original[:, :1])).max())
assert error < 2e-6, error
report = {"test": "nonzero_yaw_and_root_translation_removed_local_motion_preserved",
          "max_abs_error": error, "passed": True}
(ROOT / "text_motion/yaw_test.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
