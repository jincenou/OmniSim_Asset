"""Freeze two long-window diagnostics before viewing any rendered result."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'review_batch8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not O.exists()
 reviewed=set()
 for p in (R/'samples').glob('*/*/*/review.json'):
  parts=p.relative_to(R/'samples').parts;reviewed.add(tuple(parts[:2]))
 rows=[]
 for p in (R/'samples').glob('*/*/*/READY.json'):
  d=p.parent;parts=d.relative_to(R/'samples').parts
  if tuple(parts[:2]) in reviewed or (d/'review.json').exists() or not (d/'skeleton_audit.json').exists():continue
  ready=json.loads(p.read_text())
  if ready['candidate']['output_slots']!=121:continue
  a=json.loads((d/'skeleton_audit.json').read_text());flag=bool(a['flags'])
  if (parts[0],flag) not in [('train',True),('val',False)]:continue
  rows.append(dict(package=str(d),split=parts[0],scene=parts[1],review_id=parts[2],flagged=flag,flags=a['flags'],rank=hashlib.sha256(('review8_121slots_20261021:'+str(d.relative_to(R))).encode()).hexdigest(),ready_sha256=sha(p)))
 selected=[min([v for v in rows if (v['split'],v['flagged'])==pair],key=lambda v:v['rank']) for pair in [('train',True),('val',False)]]
 O.mkdir();(O/'selection.json').write_text(json.dumps(dict(at=time.time(),rule='Two 121-slot windows from unseen reviewed scenes: train flagged, val unflagged; fixed salted path hash minimum; diagnostic only, not prevalence estimate',eligible=len(rows),selected=selected),indent=2)+'\n')
 print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
