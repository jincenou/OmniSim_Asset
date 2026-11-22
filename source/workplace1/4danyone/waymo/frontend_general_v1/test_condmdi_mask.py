import hashlib
import json
from pathlib import Path
import numpy as np
from condmdi_mask import observation_mask, normalized_condition

ROOT = Path(__file__).resolve().parent
support = np.ones((81, 22), dtype=bool)
root = np.ones(81, dtype=bool)
# Keep hips visible but hide knees, ankles and feet for one second at 20 fps.
support[30:50, [4, 5, 7, 8, 10, 11]] = False
mask = observation_mask(support, root)
assert mask.shape == (80, 263)
position = mask[:, 4:67].reshape(80, 21, 3)
rotation = mask[:, 67:193].reshape(80, 21, 6)
velocity = mask[:, 193:259].reshape(80, 22, 3)
assert not position[30:50, np.array([4, 5, 7, 8, 10, 11]) - 1].any()
assert not rotation[30:50, [0, 1]].any()  # hips depend on the missing knees
assert not velocity[29:50, [4, 5, 7, 8, 10, 11]].any()
assert position[30:50, 19:21].all()  # visible wrists remain enabled
assert not mask[:, 259:].any()
features = np.random.default_rng(42).normal(size=mask.shape).astype(np.float32)
stats = ROOT / "vendor/CondMDI/dataset/HumanML3D_abs"
mean = np.load(stats / "Mean_abs_3d.npy")
std = np.load(stats / "Std_abs_3d.npy")
condition = normalized_condition(features, mask, mean, std)
perturbed = features.copy()
perturbed[~mask] = np.nan
np.testing.assert_array_equal(condition, normalized_condition(perturbed, mask, mean, std))
assert np.all(condition[~mask] == 0)
root[20] = False
assert not observation_mask(support, root)[20].any()
report = dict(state="passed", layout="absolute-root HumanML3D 263",
              hidden_positions_excluded=True, adjacent_velocity_excluded=True,
              dependent_rotations_excluded=True, contact_not_observed=True,
              missing_values_do_not_change_condition=True,
              scope="Mask dependency and information isolation, not motion recovery quality",
              code_sha256=hashlib.sha256((ROOT / "condmdi_mask.py").read_bytes()).hexdigest())
(ROOT / "text_motion/mask_test.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
