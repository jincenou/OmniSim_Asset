"""Remove three pilot admissions that violate the user's clarified seated two-wheel scope.
Payloads and all evidence are preserved; only their admission to the final pool changes.
"""
import sys,json,fcntl,time,collections
from pathlib import Path
sys.path.insert(0,'/mnt/ojc/DriveLab')
from asset_extraction.rigid_publication import read,atomic_json,manifest,sha256
ROOT=Path(__file__).resolve().parent;LIB=Path('/mnt/ojc/workplace1/asset_library');OUT=ROOT/'pilot_admission_recheck'
reasons={'oi_test_424992fbd868cfe3':'Source shows an airborne BMX stunt rather than seated ordinary riding','oi_test_97705e3fa1aa1144':'Source shows rider standing/straddling bike, not clearly supported by saddle','oi_validation_4eb478ec0835cfdd':'Source has training wheels; outside the selected ordinary two-wheel assembly scope'}
with (LIB/'.rigid.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 receipts=read(LIB/'manifests/rigid_publication_receipts.json');central=read(LIB/'manifests/asset_library_rigid.json');policy=read(LIB/'rigid_release_policy.json')
 saved=[]
 for source_id,reason in reasons.items():
  uid='web:openimages_riders_20261105:'+source_id
  entry=next((a for a in central['assets'] if (a.get('global_uid') or a['uid'])==uid),None)
  if uid not in receipts['assets']:continue
  prior=receipts['assets'].pop(uid);saved.append(dict(uid=uid,reason=reason,receipt=prior,central_before=entry,source='Codex inspection of input contact sheets; latest user seated-rider requirement'))
  if entry:
   entry.update(status='reject',publication_status='excluded_after_source_recheck',source_recheck_reason=reason)
 atomic_json(OUT/'excluded_receipts.json',dict(updated=time.time(),records=saved,payloads_deleted=False))
 atomic_json(LIB/'manifests/rigid_publication_receipts.json',receipts)
 atomic_json(LIB/'releases/rigid/drivelab_asset.json',manifest(policy['reference'],[a['entry'] for a in receipts['assets'].values()]))
 central['counts']=dict(collections.Counter(a['status'] for a in central['assets']));atomic_json(LIB/'manifests/asset_library_rigid.json',central)
 print('Excluded from final pool, payloads preserved:',len(saved))
