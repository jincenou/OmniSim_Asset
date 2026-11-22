"""Explicit cycle entry points for an existing Hasim installation."""
from hasim.shared.assets.library_v2 import load_asset
from hasim.shared.assets.human_cycle import CycleProfile, CyclicRuntimeAsset
from hasim.shared.insertion import InsertionScenario

def load_cyclic_asset(asset):
    profile = asset.meta.get('cycle_profile')
    if not profile or not profile.get('qualified'):
        raise ValueError(f'{asset.id}: no qualified cycle; consult needs_review.csv')
    return CyclicRuntimeAsset(load_asset(asset), CycleProfile.from_dict(profile))

def create_cyclic_scenario(trajectories, library, *, seed=0, bindings=None):
    """Track times must span the intended duration; walking speed must match the clip."""
    return InsertionScenario(trajectories, library, seed=seed, bindings=bindings,
                             playback={'default': 'cycle'})
