"""Dependency-aware observation mask for the absolute-root HumanML3D layout.

An enabled feature is a trusted estimate, never an assertion of true 3D.
Foot contacts are unobserved unless separate evidence is supplied in a future protocol.
"""
import numpy as np

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def observation_mask(joint_support, root_support):
    support = np.asarray(joint_support, dtype=bool)
    root = np.asarray(root_support, dtype=bool)
    if support.ndim != 2 or support.shape[1] != 22 or root.shape != (len(support),):
        raise ValueError("Expected T x 22 joint support and T root support")
    if len(support) < 2:
        raise ValueError("Velocity representation requires adjacent times")
    mask = np.zeros((len(support) - 1, 263), dtype=bool)
    # Positions depend on the trusted root frame as well as the joint.
    supported = support & root[:, None]
    mask[:, :4] = root[:-1, None]
    mask[:, 4:67] = np.repeat(supported[:-1, 1:], 3, axis=1)
    # Conservative rotation support: joint, parent, and immediate children.
    children = PARENTS[None, :] == np.arange(22)[:, None]
    missing_child = (~support).astype(int) @ children.T.astype(int) > 0
    rotation = supported & support[:, np.maximum(PARENTS, 0)] & ~missing_child
    mask[:, 67:193] = np.repeat(rotation[:-1, 1:], 6, axis=1)
    # A velocity crossing the missing interval must not reveal its hidden endpoint.
    mask[:, 193:259] = np.repeat(supported[:-1] & supported[1:], 3, axis=1)
    assert not mask[:, 259:].any()
    return mask


def normalized_condition(features, mask, mean, std):
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    mean, std = np.asarray(mean), np.asarray(std)
    if features.shape != mask.shape or features.shape[1] != 263:
        raise ValueError("Expected matching T x 263 arrays")
    if mean.shape != (263,) or std.shape != (263,) or not np.all(std > 0):
        raise ValueError("Invalid official normalization statistics")
    if not np.isfinite(features[mask]).all():
        raise ValueError("Trusted features contain nonfinite values")
    condition = np.zeros_like(features)
    normalized = (np.where(mask, features, mean) - mean) / std
    condition[mask] = normalized[mask]
    assert np.isfinite(condition).all()
    return condition
