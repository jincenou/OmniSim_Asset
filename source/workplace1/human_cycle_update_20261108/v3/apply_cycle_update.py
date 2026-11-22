#!/usr/bin/env python3
"""Install source-bound cycle metadata; never modifies runtime.npz or asset identity."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprint(path):
 h=hashlib.sha256()
 with np.load(path,allow_pickle=False) as z:
  for key in sorted(z.files):
   v=np.asarray(z[key]);v=np.unwrap(v) if key=='root_yaw' else v;v=np.ascontiguousarray(v)
   h.update(json.dumps([key,v.dtype.str,v.shape],separators=(',',':')).encode());h.update(memoryview(v).cast('B'))
 return h.hexdigest()
def atomic_json(path,value):
 temp=path.with_name(path.name+'.cycle-update.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True,help='human package root; directories may have been renamed')
 p.add_argument('--updates',type=Path,default=BASE/'cycle_updates.json',help='JSON produced by collect_cycle_updates.py')
 p.add_argument('--apply',action='store_true',help='Without this flag only validate and print plan');p.add_argument('--hasim-root',type=Path,help='Optionally patch known quintic rounding bug in a Hasim checkout or portable runtime')
 args=p.parse_args();catalog=json.loads(args.updates.read_text());rows=catalog['assets'];by_file={r['runtime_file_sha256']:r for r in rows if r.get('runtime_file_sha256')};by_array={r['source_sha256']:r for r in rows if r.get('source_sha256')}
 output=[];updates=[];installed=[]
 for path in sorted(args.root.rglob('asset.json')):
  if path.is_symlink():continue
  m=json.loads(path.read_text());payload=(path.parent/m.get('payload','runtime.npz')).resolve()
  if m.get('kind')!='human' or not payload.is_file():continue
  if m.get('motion_classification',{}).get('cycle_review_blocked'):
   output.append({'path':str(path),'status':'classification_quality_blocked_unchanged'});continue
  if m.get('skip_insertion') or m.get('quality_state')=='severe_failure':
   output.append({'path':str(path),'status':'excluded_asset_unchanged'});continue
  digest=sha(payload);r=by_file.get(digest)
  if r is None:r=by_array.get(fingerprint(payload))
  if r is None:output.append({'path':str(path),'status':'unmatched_payload_unchanged'});continue
  prof=r.get('profile')
  if not prof or not prof.get('qualified'):
   output.append({'path':str(path),'source_id':r['id'],'status':'not_qualified_unchanged','reason':r.get('error') or r.get('failed_checks')});continue
  # Validate semantic calibration metadata too, never match only a renamed folder.
  mode=prof['mode'];allowed={'walking','jogging','running'} if mode=='locomotion' else {'standing','gesturing','standing_or_small_gesture'}
  if m.get('motion_type') not in allowed or not m.get('playback_capabilities',{}).get('heading_normalized') or not np.isclose(m.get('duration_s',-1),prof['source_duration_s'],atol=1e-8,rtol=0):
   output.append({'path':str(path),'status':'metadata_mismatch_unchanged'});continue
  prof=dict(prof,asset_id=m['id']);new=dict(m,cycle_profile=prof,playback='cycle')
  new['playback_capabilities']=dict(m.get('playback_capabilities',{}),qualified_cycle_available=True)
  # All unrelated metadata is preserved; the NPZ is never written.
  updates.append((path,new));installed.append(prof);output.append({'path':str(path),'source_id':r['id'],'target_id':m['id'],'status':'ready','runtime_sha256':digest})
 patch=None
 if args.hasim_root:
  matches=[x for x in [args.hasim_root/'src/shared/assets/human_cycle.py',args.hasim_root/'hasim/shared/assets/human_cycle.py',args.hasim_root/'runtime/hasim/shared/assets/human_cycle.py'] if x.is_file()]
  if len(matches)!=1:raise SystemExit('Could not uniquely locate human_cycle.py; no changes applied')
  target=matches[0];text=target.read_text();old='blend = float(u**3 * (10 + u * (-15 + 6 * u)))';new='blend = float(np.clip(u**3 * (10 + u * (-15 + 6 * u)), 0., 1.))'
  if old in text:
   text=text.replace(old,new);compile(text,str(target),'exec');patch=(target,text)
  elif new not in text:raise SystemExit('Unknown cycle runtime version; no changes applied. Review supplied patch.')
 report={'apply':args.apply,'root':str(args.root.resolve()),'matched_qualified':len(updates),'assets':output,'runtime_patch':str(patch[0]) if patch else None}
 if args.apply and not updates:raise SystemExit('No qualified matching assets; no files changed')
 if args.apply:
  if patch:
   target,text=patch;tmp=target.with_name(target.name+'.cycle-update.tmp');tmp.write_text(text);os.replace(tmp,target)
  for path,new in updates:atomic_json(path,new)
  atomic_json(args.root/'cycle_catalog.json',{'format':'hasim_human_cycles_v1','profiles':installed})
  atomic_json(args.root/'cycle_install_report.json',report)
 else:
  atomic_json(BASE/'last_dry_run.json',report)
 from collections import Counter
 print(json.dumps({'apply':args.apply,'matched_qualified':len(updates),'statuses':dict(Counter(r['status'] for r in output)),'runtime_patch':report['runtime_patch']},ensure_ascii=False,indent=2))
 if not updates:raise SystemExit(2)
if __name__=='__main__':main()
