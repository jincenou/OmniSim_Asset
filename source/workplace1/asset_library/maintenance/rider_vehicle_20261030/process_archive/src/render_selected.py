import sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.export import render_evidence
r=W/'runs'/sys.argv[1]/'selected'
render_evidence(r/'canonical.ply',r/'render',size=384)
