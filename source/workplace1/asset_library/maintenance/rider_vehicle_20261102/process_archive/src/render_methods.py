import sys,json,argparse
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.export import canonicalize_raw,render_evidence
from asset_extraction.io_utils import write_json
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--method',required=True);a=p.parse_args();work=a.run/a.method
state=json.loads((work/'generation.json').read_text());row=state['assets'][0]
if row['status']!='generated':raise ValueError('No generated asset')
dest=work/'curation';dest.mkdir(exist_ok=True)
meta=dest/'canonical.json'
if not meta.exists():
 c=canonicalize_raw(row['raw_path'],dest/'axis.ply',route=row['route'],category=row['category'],assembly_metadata=row.get('assembly'));write_json(meta,c)
else:c=json.loads(meta.read_text())
if (dest/'axis_render/render.json').exists() and (dest/'axis_render/turntable.mp4').stat().st_size>0:
 print('RENDER_CACHED',a.method,flush=True);sys.exit(0)
r=render_evidence(c['canonical_path'],dest/'axis_render',size=384);print('RENDER_DONE',a.method,flush=True)
