"""Regression checks without GPU inference or mutation of production results."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics
from identity import canonical_identity
import evaluate
from evaluate import extent_diagnostics


class MetricAccounting(unittest.TestCase):
    def test_full_summary_separates_failures_and_keep_cohort(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'assets';registry=root/'registry.json'
            rows=[]
            for oid,status,valid,width in [('a','keep',True,0),('b','reject',True,100),('c','failed',False,None)]:
                rows.append(dict(obj_id=oid,dataset='waymo',scene_id='scene',category='car',route='H',quality_status=status))
                folder=out/oid;folder.mkdir(parents=True)
                (folder/'render.json').write_text(json.dumps(dict(status='completed' if valid else 'failed',all_16_nonempty=valid,
                    dimensions={'Width_APE_percent':width} if valid else None)))
                if valid:
                    vector=np.array([1.,0.]) if oid=='a' else np.array([0.,1.])
                    np.savez(folder/'dino.npz',global_embedding=np.tile(vector,(17,1)),
                        names=np.array(['source']+[f'orbit_{i:02d}' for i in range(16)]))
            registry.write_text(json.dumps(dict(protocol={},assets=rows)))
            with patch.object(evaluate,'OUT',out),patch.object(evaluate,'N',root),patch.object(evaluate,'REGISTRY',registry),patch.object(evaluate,'RESULT_NAME','synthetic'):
                with contextlib.redirect_stdout(io.StringIO()):evaluate.summarize()
            result=json.loads((root/'pilot_results/synthetic.json').read_text())
            summaries={x['cohort']:x for x in result['summary'] if x['dataset']=='waymo'}
            self.assertEqual(len(result['assets']),3)
            self.assertEqual(summaries['all_valid_exports']['valid_queries'],2)
            self.assertEqual(summaries['all_valid_exports']['metrics']['Width_APE_percent'],{'n':2,'mean':50.})
            self.assertEqual(summaries['final_keep']['valid_queries'],1)
            self.assertEqual(summaries['final_keep']['metrics']['Width_APE_percent'],{'n':1,'mean':0.})
            self.assertIsNone(summaries['final_keep']['metrics']['FID_2048']['value'])
            self.assertIsNone(summaries['final_keep']['metrics']['FD_DINO']['value'])
            fd=summaries['all_valid_exports']['metrics']['FD_DINO']
            self.assertEqual(fd['n_identities'],2)
            self.assertEqual(fd['n_rendered'],32)
            self.assertEqual(fd['feature_dimension'],2)
            self.assertTrue(np.isfinite(fd['value']))

    def test_frechet_normalized_features(self):
        import numpy as np
        x=np.array([[1.,0.],[0.,1.]])
        y=-x
        self.assertAlmostEqual(evaluate.fid_value(x,x),0.,places=7)
        self.assertAlmostEqual(evaluate.fid_value(x,y),2.,places=7)
        self.assertAlmostEqual(evaluate.fid_value(x,y),evaluate.fid_value(y,x),places=7)
        self.assertIsNone(evaluate.fid_value(x[:1],y))

    def test_extent_diagnostics_are_scale_invariant_and_reject_invalid_inputs(self):
        for factor in [1.,10.]:
            result=extent_diagnostics([2*factor,1*factor,.5*factor],[4,2,1])
            self.assertEqual(result['Width_APE_percent'],0)
            self.assertEqual(result['Height_APE_percent'],0)
            self.assertEqual(result['Box_extent_IoU'],1)
        wide=extent_diagnostics([4,4,1],[4,2,1])
        self.assertEqual(wide['Width_APE_percent'],100)
        self.assertEqual(wide['Height_APE_percent'],0)
        self.assertEqual(wide['Box_extent_IoU'],.5)
        for extent in [[0,2,1],[4,-1,1],[float('nan'),2,1],[4,2]]:
            with self.assertRaises(ValueError):extent_diagnostics(extent,[4,2,1])

    def test_exclusions_do_not_mix_admission_cohorts(self):
        keys = ['ED_R', 'PSNR_dB', 'SSIM', 'LPIPS_alex', 'silhouette_IoU']
        assets = []
        for name, status, value in [('keep_good', 'keep', .2),
                                    ('reject_valid', 'reject', .8),
                                    ('keep_bad_pose', 'keep', .1)]:
            assets.append(dict(obj_id=name, dataset='waymo', quality_status=status,
                               valid_16_views=True, **dict.fromkeys(keys, value)))
        result = dict(protocol={}, assets=assets, summary=[
            dict(dataset='waymo', cohort=c, metrics={})
            for c in ['all_valid_exports', 'final_keep']])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evaluation').mkdir()
            (root / 'evaluation/test_matched_exclusions.json').write_text(
                json.dumps(dict(objects={'keep_bad_pose': 'unverified heading'})))
            with patch.object(metrics, 'ROOT', root), patch.object(metrics, 'SNAPSHOT', 'test'):
                metrics.validate_matched_summary(result)
            for key in keys:
                self.assertEqual(result['summary'][0]['metrics'][key]['n'], 2)
                self.assertAlmostEqual(result['summary'][0]['metrics'][key]['mean'], .5)
                self.assertEqual(result['summary'][1]['metrics'][key]['n'], 1)
                self.assertAlmostEqual(result['summary'][1]['metrics'][key]['mean'], .2)
                self.assertIsNone(result['assets'][2][key])
            preserved = json.loads((root / 'pilot_results/test_before_alignment_validation.json').read_text())
            self.assertEqual(preserved['assets'][2]['ED_R'], .1)

    def test_verified_native_aliases_agree_without_merging_other_tracks(self):
        scene = 'waymo_test_scene'
        direct = dict(native_id=f'waymo:{scene}:laser_association:object_a')
        normalized = dict(native_id=f'waymo:{scene}:lidar_track:object_a')
        other = dict(native_id=f'waymo:{scene}:lidar_track:object_b')
        before = copy.deepcopy(direct)
        key = lambda row: canonical_identity(row, 'waymo', scene)['native_id']
        self.assertEqual(key(direct), key(normalized))
        self.assertNotEqual(key(direct), key(other))
        self.assertEqual(direct, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
