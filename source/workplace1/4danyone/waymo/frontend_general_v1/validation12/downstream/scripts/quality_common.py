"""Reuse original A recovery and losses with cohort-specific paths."""
import importlib.util
import sys

import grid_context

path = grid_context.GRID / "optimization_cross8_20261018/scripts/quality_common.py"
spec = importlib.util.spec_from_file_location("_frozen_grid_quality", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.NEW = grid_context.D
module.BASE = grid_context.D / "data"
module.manifest = grid_context.manifest
globals().update({name: value for name, value in vars(module).items() if not name.startswith("_")})
