"""Regression: unknown/stale/partial reviews must stop before generation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from gate import check_root, require_review, ReviewRequired, sha
from repair import merge_masks, composite
import numpy as np

class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root/'inputs').mkdir();(self.root/'preflight/conditioning').mkdir(parents=True)
        self.input=self.root/'inputs/main.mp4';self.input.write_bytes(b'reviewed RGB input')
        def file(name):
            p=self.root/name
            if not p.exists():p.write_bytes(name.encode())
            return dict(path=str(p),sha256=sha(p))
        self.review=dict(schema='accessory-input-review-v1',decision='approved_for_generation',reviewer='test fixture',
            accessories='preserved',frames=3,reviewed_frames=[0,1,2],source_input_compared=True,
            no_new_bystander_leakage=True,no_visible_accessory_whiteout=True,occlusion_drift_checked=True,
            files=dict(source_rgb=file('source.mp4'),mask=file('mask.npz'),generation_input=file('inputs/main.mp4')),
            evidence=[file('contact.jpg')])
        self.path=self.root/'preflight/accessory_review.json'
    def write(self):self.path.write_text(json.dumps(self.review))
    def test_valid(self):
        self.write();self.assertEqual(require_review(self.path,self.input)['accessories'],'preserved')
    def test_missing(self):
        with self.assertRaises(ReviewRequired):require_review(self.path,self.input)
    def test_unknown(self):
        self.review['accessories']='unknown';self.write()
        with self.assertRaises(ReviewRequired):require_review(self.path,self.input)
    def test_partial_or_drift(self):
        self.review['reviewed_frames']=[0,2];self.write()
        with self.assertRaises(ReviewRequired):require_review(self.path,self.input)
        self.review['reviewed_frames']=[0,1,2];self.review['occlusion_drift_checked']=False;self.write()
        with self.assertRaises(ReviewRequired):require_review(self.path,self.input)
    def test_stale_source_and_mask_and_input_and_evidence(self):
        self.write()
        paths=[self.review['files'][key]['path'] for key in ('source_rgb','mask','generation_input')]+[self.review['evidence'][0]['path']]
        for name in paths:
            p=Path(name);old=p.read_bytes();p.write_bytes(old+b'changed')
            with self.assertRaises(ReviewRequired):require_review(self.path,self.input)
            p.write_bytes(old)
    def test_other_input_and_symlink(self):
        self.write();other=self.root/'other.mp4';other.write_bytes(self.input.read_bytes())
        with self.assertRaises(ReviewRequired):require_review(self.path,other)
        link=self.root/'link.mp4';link.symlink_to(self.input);require_review(self.path,link)
    def test_cached_conditioning(self):
        cond=self.root/'preflight/conditioning';(cond/'appearance.mp4').write_bytes(b'actual cached appearance')
        (cond/'metadata.json').write_text(json.dumps(dict(source_video='appearance.mp4')))
        self.write()
        with self.assertRaises(ReviewRequired):check_root(self.root)
        self.review['files']['conditioning_source']=dict(path=str(cond/'appearance.mp4'),sha256=sha(cond/'appearance.mp4'))
        self.write();check_root(self.root)
        (cond/'appearance.mp4').write_bytes(b'stale whiteout')
        with self.assertRaises(ReviewRequired):check_root(self.root)
    def test_mask_union_and_no_recolor(self):
        body=np.array([[[255,0],[128,0]]],dtype=np.uint8)
        items=np.array([[[[0,255],[200,0]]]],dtype=np.uint8)
        merged=merge_masks(body,items)
        self.assertTrue(np.all(merged>=body));self.assertEqual(int(merged[0,0,1]),255)
        rgb=np.array([[[10,20,30],[180,60,120]],[[40,50,60],[70,80,90]]],np.uint8)
        out=composite(rgb,merged[0]);np.testing.assert_array_equal(out[0,1],rgb[0,1])
        np.testing.assert_array_equal(out[1,1],[255,255,255])
        with self.assertRaises(ValueError):merge_masks(body,items.astype(np.float32))

if __name__=='__main__':unittest.main()
