import sys,json,collections,copy,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vehicle_priority import prioritize,POLICY

def sample(cats):
 rows=[dict(obj_id=str(i),category=c,original={'i':i}) for i,c in enumerate(cats)]
 return dict(rows=rows,routes=[dict(obj_id=r['obj_id'],route='H') for r in reversed(rows)],removed=['old duplicate'])
class Checks(unittest.TestCase):
 def test_preserve_prefix_routes_and_identity(self):
  s=sample(['car']*8+['bus']*4+['truck']*5+['van']*4);before=copy.deepcopy(s);out=prioritize(s,4)
  self.assertEqual(s,before);self.assertEqual(out['rows'][:4],s['rows'][:4])
  self.assertEqual({r['obj_id']:r for r in s['rows']},{r['obj_id']:r for r in out['rows']})
  self.assertEqual([r['obj_id'] for r in out['rows']],[r['obj_id'] for r in out['routes']])
  self.assertEqual([r['category'] for r in out['rows'][4:8]],['bus','truck','van','car'])
  self.assertIs(prioritize(out,8),out)
 def test_borrow_empty_classes(self):
  out=prioritize(sample(['car']*5+['truck']*7),0)
  self.assertEqual([r['category'] for r in out['rows'][:8]],['truck']*3+['car']+['truck']*3+['car'])
 def test_empty_and_single_classes(self):
  for cats in [[],['car']*7,['bus']*7,['cone','van','car']]:
   out=prioritize(sample(cats),0);self.assertEqual(collections.Counter(cats),collections.Counter(r['category'] for r in out['rows']))
 def test_bad_prefix(self):
  with self.assertRaises(ValueError):prioritize(sample(['car']),2)

if __name__=='__main__':unittest.main()
