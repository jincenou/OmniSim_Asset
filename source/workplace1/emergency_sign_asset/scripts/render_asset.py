import json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.export import canonicalize_raw,render_evidence
r=W/'runs'/sys.argv[1];row=json.loads((r/'generation/generation.json').read_text())['assets'][0];d=r/'curation';d.mkdir(exist_ok=True)
if not (d/'axis.json').exists():
 c=canonicalize_raw(row['raw_path'],d/'axis.ply',route=row['route'],category=row['category']);(d/'axis.json').write_text(json.dumps(c,indent=2))
render_evidence(d/'axis.ply',d/'axis_render',size=512)
