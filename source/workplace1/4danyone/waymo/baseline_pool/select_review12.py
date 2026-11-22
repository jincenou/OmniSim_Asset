"""Eight fixed input windows, selected before output inspection."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'review_batch12'
def main():
    assert not O.exists()
    reviewed={tuple(p.relative_to(R/'samples').parts[:2]) for p in (R/'samples').glob('*/*/*/review.json')}
    rows=[]
    for p in (R/'samples').glob('*/*/*/READY.json'):
        d=p.parent;parts=d.relative_to(R/'samples').parts
        if tuple(parts[:2]) in reviewed or (d/'review.json').exists() or not (d/'skeleton_audit.json').exists():continue
        ready=json.loads(p.read_text())
        if ready['candidate']['output_slots']!=61:continue
        audit=json.loads((d/'skeleton_audit.json').read_text())
        rows.append(dict(package=str(d),split=parts[0],scene=parts[1],review_id=parts[2],flagged=bool(audit['flags']),flags=audit['flags'],rank=hashlib.sha256(('review12_61slots_20261022:'+str(d.relative_to(R))).encode()).hexdigest(),ready_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    picked=[];scenes=set()
    for split,flag in [('train',False),('train',True),('val',False),('val',True)]:
        eligible=sorted([r for r in rows if (r['split'],r['flagged'])==(split,flag)],key=lambda r:r['rank'])
        n=0
        for row in eligible:
            key=(row['split'],row['scene'])
            if key in scenes:continue
            picked.append(row);scenes.add(key);n+=1
            if n==2:break
        assert n==2
    O.mkdir();(O/'selection.json').write_text(json.dumps(dict(at=time.time(),rule='Two windows per train/val x numerical-risk/no-risk stratum; fixed salted SHA path rank; different previously unreviewed scenes; 61 slots; not a population quality estimate',eligible=len(rows),selected=picked),indent=2)+'\n')
    print([(r['split'],r['scene'],r['review_id'],r['flagged']) for r in picked])
if __name__=='__main__':main()
