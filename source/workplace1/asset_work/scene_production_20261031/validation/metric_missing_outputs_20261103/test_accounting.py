"""Exercise the real summarizer on present, deleted, and failed render fixtures."""
import ast,csv,json,tempfile,contextlib,io
from pathlib import Path
ROOT=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
tree=ast.parse((ROOT/'evaluate.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'summarize','fid_value'}]
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
with tempfile.TemporaryDirectory() as tmp:
 n=Path(tmp);out=n/'evaluation/assets';registry=n/'registry.json'
 rows=[]
 for oid,quality,present,status,nonempty in [('keep','keep',True,'completed',True),('deleted','reject',False,'failed',False),('cached','reject',False,'completed',True),('failed_keep','keep',True,'failed',True)]:
  ply=n/(oid+'.ply')
  if present:ply.write_text('fixture only')
  rows.append(dict(obj_id=oid,dataset='nuscenes',scene_id='test',category='car',route='H',quality_status=quality,ply=str(ply)))
  write_json(out/oid/'render.json',dict(status=status,all_16_nonempty=nonempty,error='FileNotFoundError: missing PLY' if oid=='deleted' else None))
 write_json(registry,dict(protocol={'queries':5,'exported_payloads':4},assets=rows))
 ns=dict(Path=Path,read_json=read_json,write_json=write_json,REGISTRY=registry,OUT=out,N=n,RESULT_NAME='test',csv=csv,json=json)
 exec(compile(ast.Module(body=functions,type_ignores=[]),str(ROOT/'evaluate.py'),'exec'),ns)
 with contextlib.redirect_stdout(io.StringIO()):ns['summarize']()
 result=read_json(n/'pilot_results/test.json');all_rows,keep_rows=result['summary'][:2]
 assert all_rows['registered_exports']==4 and all_rows['valid_queries']==2
 assert all_rows['excluded_invalid_render']==2 and all_rows['missing_ply_at_summary']==2
 assert all_rows['valid_cached_render_without_ply']==1
 assert keep_rows['registered_exports']==2 and keep_rows['valid_queries']==1
 assert keep_rows['missing_ply_at_summary']==0
 assert result['protocol']['queries']==5 and len(result['assets'])==4
 assert result['assets'][1]['render_error'].startswith('FileNotFoundError')
 assert not result['assets'][3]['valid_16_views']
 assert all_rows['metrics']['FID_2048']['value'] is None
 assert result['summary'][2]['registered_exports']==0
 csv_rows=list(csv.DictReader((n/'pilot_results/test_summary.csv').open()))
 assert csv_rows[0]['missing_ply_at_summary']=='2'
 publication=ast.parse((ROOT/'metrics.py').read_text())
 publish=next(x for x in publication.body if isinstance(x,ast.FunctionDef) and x.name=='publish_metrics')
 pub_ns=dict(ROOT=n,LIBRARY=n/'bank',SNAPSHOT='test',write_json=write_json)
 exec(compile(ast.Module(body=[publish],type_ignores=[]),str(ROOT/'metrics.py'),'exec'),pub_ns)
 pub_ns['publish_metrics'](result)
 page=(n/'bank/previews'/n.name/'test.html').read_text()
 assert '<th>Missing PLY</th>' in page and '<th>Valid cache without PLY</th>' in page
 print('PASS: missing outputs retained; failed render excluded; valid cache counted; original query denominator unchanged; empty cohorts null')
