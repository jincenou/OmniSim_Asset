import sys,json
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.export import canonicalize_raw
from asset_extraction.io_utils import write_json
run=Path(sys.argv[1]);method=sys.argv[2];front=sys.argv[3];row=json.loads((run/method/'generation.json').read_text())['assets'][0]
out=run/'selected';out.mkdir(exist_ok=True)
meta=canonicalize_raw(row['raw_path'],out/'canonical.ply',route=row['route'],category=row['category'],assembly_metadata=row['assembly'],heading=dict(front=front,confidence=.95,source='Codex inspected 16-view sheet: front wheel, headlight and handlebars establish travel direction, independent of face'))
write_json(out/'canonical.json',meta);write_json(out/'selected.json',dict(method=method,row=row));print(out)
