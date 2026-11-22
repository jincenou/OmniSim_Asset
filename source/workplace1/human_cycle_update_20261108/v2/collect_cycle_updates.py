#!/usr/bin/env python3
"""Collect all per-asset batch reports into an installable metadata-only update."""
import argparse,csv,json
from collections import Counter
from pathlib import Path

def collect(root):
 rows=[json.loads(p.read_text()) for p in sorted((root/'reports').glob('*.json'))]
 if not rows:raise ValueError('No reports found')
 safe=[{k:v for k,v in r.items() if k!='traceback'} for r in rows]
 (root/'cycle_updates.json').write_text(json.dumps({'format':'hasim_cycle_incremental_v1','assets':safe},ensure_ascii=False,indent=2)+'\n')
 (root/'cycle_catalog.json').write_text(json.dumps({'format':'hasim_human_cycles_v1','profiles':[r['profile'] for r in rows if r.get('profile',{}).get('qualified')]},ensure_ascii=False,indent=2)+'\n')
 summary={'total_examined':len(rows),'statuses':dict(Counter(r['status'] for r in rows)), 'qualified_by_motion':dict(Counter(r['motion_type'] for r in rows if r.get('profile',{}).get('qualified'))),'note':'qualified_numeric is an automated geometry/runtime check, not a full visual or foot-lock certificate'}
 (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 with (root/'needs_review.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['id','folder','motion_type','status','reason'])
  for r in rows:
   if not r.get('profile',{}).get('qualified'):w.writerow([r['id'],r['folder'],r['motion_type'],r['status'],r.get('error') or ','.join(r.get('failed_checks',[]))])
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
