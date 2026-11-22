#!/usr/bin/env python3
"""Apply reviewed motion labels by payload identity, preserving all geometry and quality flags."""
import argparse,json,os
from pathlib import Path
from apply_cycle_update import sha,fingerprint,atomic_json
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
rows=json.loads((B/'classification_updates.json').read_text())['assets'];byfile={r['runtime_file_sha256']:r for r in rows};byarray={r['source_sha256']:r for r in rows};updates=[];report=[]
for path in a.root.rglob('asset.json'):
 if path.is_symlink():continue
 m=json.loads(path.read_text());payload=(path.parent/m.get('payload','runtime.npz')).resolve()
 if m.get('kind')!='human' or not payload.is_file():continue
 if m.get('motion_type')!='unclassified_motion' and not m.get('motion_classification'):continue
 r=byfile.get(sha(payload))
 # An unknown ID may be renamed, but only hash-matched payloads are ever changed.
 if r is None and (m.get('motion_type')=='unclassified_motion' or m.get('motion_classification')):r=byarray.get(fingerprint(payload))
 if r is None:continue
 if m.get('motion_type') not in [r['old_motion_type'],r['motion_type']]:
  report.append({'path':str(path),'status':'conflicting_existing_label_unchanged'});continue
 new=dict(m,motion_type=r['motion_type'],motion_classification=dict(r['classification'],original_motion_type=r['old_motion_type']))
 if r['classification']['cycle_review_blocked'] and m.get('cycle_profile',{}).get('qualified'):
  report.append({'path':str(path),'status':'existing_cycle_requires_manual_review_unchanged'});continue
 updates.append((path,new));report.append({'path':str(path),'id':m['id'],'motion_type':r['motion_type'],'status':'ready'})
if a.apply:
 for path,m in updates:atomic_json(path,m)
 atomic_json(a.root/'classification_install_report.json',{'updated':len(updates),'assets':report})
print(json.dumps({'apply':a.apply,'matched':len(updates),'conflicts':[r for r in report if r['status']!='ready']},ensure_ascii=False,indent=2))
