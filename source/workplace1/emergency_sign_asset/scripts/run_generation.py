import sys,asyncio,json
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.generation import prepare_jobs,generate
from asset_extraction.io_utils import read_json
r=W/'runs'/sys.argv[1]
bundle=prepare_jobs(r/'flux_input.json',r/'routes.json',read_json(W/'generation_config.json'))
state=asyncio.run(generate(bundle,r/'generation',[int(sys.argv[2]) if len(sys.argv)>2 else 1]))
print(json.dumps(dict(status=state['status'],counts=state['counts'],assets=[dict(id=x['obj_id'],status=x['status'],reason=x.get('review_reason')) for x in state['assets']]),indent=2))
