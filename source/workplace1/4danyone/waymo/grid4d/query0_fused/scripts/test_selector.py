"""Tests only selector contracts: indices, probabilities, heldout isolation, empty cells."""
import unittest
import numpy as np
from selector import Selector,normalize

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(train_views=[1,2],observations=[dict(index=i,uid=10+i,view=1+i//2,frame=i%2,bin=i%2,areas=[128 if i%2==0 else 64]*6) for i in range(4)])
        self.table=dict(domain='training_only',views=[1,2],demand=[[.1,.8]+[None]*6 for _ in range(6)],full_difficulty={'1:0':.1,'1:1':.8,'2:0':.3,'2:1':.5})
        self.base=[10,11,12,13]*4
    def test_normalization(self):
        np.testing.assert_allclose(normalize([2,3]),[.4,.6])
        for values in [[0,0],[-1,2],[float('nan'),1],[]]:
            with self.assertRaises(ValueError):normalize(values)
    def test_shared_even_and_valid_indices(self):
        for arm in ['U','P','E','Q']:
            s=Selector(self.c,self.base,arm);s.update(self.table)
            for step in range(1,17):
                r=s.choose(step);self.assertIn(r['uid'],self.base);self.assertNotIn(r['view'],[0,8,16]);self.assertGreater(r['selection_path_probability'],0)
                if arm=='U' or step%2==0:self.assertEqual(r['uid'],self.base[step-1])
            self.assertEqual(int(s.exposure.sum()),16)
    def test_empty_part_cells_preserve_uniform(self):
        c=dict(self.c);c['observations']=[dict(r,areas=[0]*6) for r in c['observations']]
        for arm in ['P','Q']:
            s=Selector(c,self.base,arm);s.update(self.table);r=s.choose(1);self.assertEqual(r['uid'],10);self.assertEqual(r['selection_reason'],'empty_part_candidates_uniform')
    def test_forbidden_view(self):
        c=dict(self.c);c['observations']=[dict(r,view=0) for r in c['observations']]
        with self.assertRaises(AssertionError):Selector(c,self.base,'Q')
    def test_exposure_penalty_and_resume(self):
        s=Selector(self.c,self.base,'Q');s.update(self.table);ids,w=s.secondary((0,0));s.exposure[ids[0]]=99;_,changed=s.secondary((0,0));self.assertLess(changed[0],w[0])
        s.choose(1);state=s.state();nextitem=s.choose(3);t=Selector(self.c,self.base,'Q');t.update(self.table);t.restore(state);self.assertEqual(t.choose(3),nextitem)
    def test_unknown_demand_not_zero(self):
        t=dict(self.table,demand=[[None]*8 for _ in range(6)]);s=Selector(self.c,self.base,'Q');s.update(t);r=s.choose(1);self.assertEqual(r['selection_reason'],'empty_part_candidates_uniform')
if __name__=='__main__':unittest.main()
