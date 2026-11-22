from pathlib import Path
import json,hashlib,sys,runpy,csv,os,collections
ROOT=Path('/mnt/ojc/workplace1/asset_library/human');REPORT=Path(__file__).parent;REPO=Path('/mnt/ojc/workplace1/Hasim_trajectory_assets')
summary=json.loads((REPORT/'inplace/summary.json').read_text());assert summary['complete'] and not summary['failures']
index=json.loads((ROOT/'library.json').read_text());assert len(index['assets'])==558 and len(index['excluded_assets'])==4
rows=summary['results'];assert len(rows)==562
h=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()
expected={};checks=[]
for row in rows:
 meta_path=ROOT/row['metadata'];meta=json.loads(meta_path.read_text());package=meta_path.parent
 assert {p.name for p in package.iterdir()}=={'asset.json','runtime.npz'},package
 assert all(p.is_file() and not p.is_symlink() for p in package.iterdir())
 assert h(meta_path)==row['metadata_sha256'];assert h(package/'runtime.npz')==row['runtime_sha256']
 assert row['roundtrip_exact'] and row['hasim']['array_equality'] and row['hasim']['sample_times']==5
 if row['qualified_cycle']:assert row['hasim']['cycle_checked']
 if row['preview']:assert (ROOT/row['preview']).is_file()
 assert bool(meta.get('skip_insertion'))==bool(row['excluded_reason'])
 expected[row['id']]=str(package/'runtime.npz');checks.append({'id':row['id'],'hashes_verified':True})
 assert not any((package/p).exists() for p in ['asset_4dgs.npz','manifest.json','runtime','refs','placement_playback.npz','library.json'])
 assert meta['id']==row['id']
print('562 hashes / file layouts / preserved flags PASS',flush=True)
api=runpy.run_path(str(REPO/'scripts/insertion/verify_asset_export.py'));lib=api['AssetLibrary'](ROOT.parent);human=api['AssetLibrary'](ROOT)
assert len(lib.candidates('human'))==558 and len(human.assets)==558
rigid_direct=api['AssetLibrary'](ROOT.parent/'releases/rigid')
assert {a.id for a in lib.candidates('rigid')}=={a.id for a in rigid_direct.assets}
assert set(human.excluded)=={r['id'] for r in rows if r['excluded_reason']}
for aid in ['waymo_grid4d_H0001','waymo_grid4d_H0007','waymo_grid4d_H1132','nuscenes_grid4d_HN0127','nuscenes_grid4d_HN0018']:
 runtime=api['load_asset'](lib.get(aid));runtime.sample(runtime.duration_s/2)
print(f'Mixed parent library {len(rigid_direct.assets)} rigid + 558 human PASS',flush=True)
# Repair only human payload paths in the existing inventory CSV; keep other historic rows untouched.
p=ROOT.parent/'previews/inventory_20261102/all_assets.csv'
if p.exists():
 with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;data=list(reader)
 changed=0
 for row in data:
  if row.get('family')=='human' and row['uid'] in expected:row['asset_path']=expected[row['uid']];changed+=1
 with p.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
 assert changed==562
 p=ROOT.parent/'previews/inventory_20261102/inventory.json'
 if p.exists():
  data=json.loads(p.read_text());data['human_runtime_migration']={'date':'2026-11-05','index':str(ROOT/'library.json'),'assets':562,'eligible':558,'excluded':4,'preview_index':str(ROOT/'previews/index.html'),'note':'Human payload paths in all_assets.csv updated; other inventory counts remain the historical snapshot.'};p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
result={'status':'passed','total':562,'eligible':558,'excluded':4,'rigid_candidates_match_authoritative_release':True,'rigid_count':len(rigid_direct.assets),'all_payload_hashes_checked':True,'all_assets_sample_equivalent':True,'cycle_profiles':sum(r['qualified_cycle'] for r in rows),'retained_preview_videos':sum(bool(r['preview']) for r in rows),'runtime_bytes':sum(r['runtime_bytes'] for r in rows),'no_asset_backups':not summary['source_assets_backed_up'],'checks':checks}
(REPORT/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)
