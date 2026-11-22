"""The original 24-view export recipe, now using the tested common exporter."""
import os
from pathlib import Path
import runpy

R=Path(__file__).resolve().parents[1]
os.environ.setdefault('FDANYONE_ASSET_NAME','phone_full')
os.environ.setdefault('FDANYONE_ASSET_SOURCE',str(R.parent/'outputs/turbo'))
runpy.run_path(str(R/'scripts/export_sequence.py'),run_name='__main__')
