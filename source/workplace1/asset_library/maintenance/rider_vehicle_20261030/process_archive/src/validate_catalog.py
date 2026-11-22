import sys,json,importlib.util,hashlib,gzip
from pathlib import Path
import numpy as np
L=Path('/mnt/ojc/workplace1/asset_library');W=Path(__file__).resolve().parents[1];src=Path('/mnt/ojc/workplace1/Hasim/src')
spec=importlib.util.spec_from_file_location('hasim',src/'__init__.py',submodule_search_locations=[str(src)]);m=importlib.util.module_from_spec(spec);sys.modules['hasim']=m;spec.loader.exec_module(m)
from hasim.preparation.trajectories.rules.assets import AssetLibrary,bind_rigid
records=[]
for manifest,root in [(L/'manifests/asset_library_rigid.json',L),(L/'rigid/asset_library_rigid.json',L/'rigid')]:
 lib=AssetLibrary(manifest,pipeline_root=root);new=[e for e in lib.entries if e.uid.startswith('rider_vehicle_20261030:')];assert len(new)==3
 for e in new:
  b=bind_rigid(e,np.random.default_rng(42));assert b.asset_path.is_file();target=e.raw['scale']['est_length_m'];assert .94*target<=b.size_lwh_m[0]<=1.06*target
  assert hashlib.sha256(b.asset_path.read_bytes()).hexdigest()==e.raw['canonical']['sha256']
  records.append(dict(manifest=str(manifest),uid=e.uid,asset_path=str(b.asset_path),bound_scale=b.scale,size_lwh_m=b.size_lwh_m,default_scale_jitter=.06))
 backup=L/'maintenance/rider_vehicle_20261030/before'/manifest.relative_to(L)
 before=json.loads(backup.read_text() if backup.exists() else gzip.open(str(backup)+'.gz','rt').read());after=json.loads(manifest.read_text());index={x['uid']:x for x in after['assets']};assert all(index[x['uid']]==x for x in before['assets']);assert len(after['assets'])==len(before['assets'])+3
out=dict(passed=True,actual_loader='Hasim AssetLibrary + bind_rigid',manifests_tested=2,new_unique_assets=3,existing_rows_unchanged=True,bindings=records)
(L/'maintenance/rider_vehicle_20261030/catalog_validation.json').write_text(json.dumps(out,indent=2));print('CATALOG_PASSED: 3 assets, both manifest roots, actual Hasim binding, prior rows unchanged')
