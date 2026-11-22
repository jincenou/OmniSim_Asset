import unittest
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from copy import deepcopy
from asset_extraction.export import _verdict
from asset_extraction.quality import needs_shape_retry, retry_bundle, proportion_guard, SCHEMA
from jsonschema import Draft202012Validator
from asset_extraction.io_utils import write_json
from asset_extraction.quality import repair_shapes


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.audit=dict(keep=True,category_match=True,identity_consistent=False,quality_score=4,
                        severe_defects=[],proportion_status='acceptable',
                        scale=dict(scale_confidence='medium'))
        self.render=dict(all_16_nonempty=True)

    def test_minor_identity_differences_allowed_for_all_categories(self):
        self.assertEqual(_verdict(self.audit,self.render,None,False),'keep')

    def test_hard_failure_not_masked_by_high_score(self):
        for key,value in [('severe_defects',['large_floaters']),('proportion_status','severely_mismatched'),
                          ('category_match',False),('keep',False)]:
            with self.subTest(key=key):
                a=dict(self.audit,quality_score=9);a[key]=value
                self.assertEqual(_verdict(a,self.render,None,False),'reject')
        self.assertEqual(_verdict(self.audit,dict(all_16_nonempty=False),None,False),'reject')

    def test_uncertainty_not_a_shape_failure_and_heading_still_required(self):
        a=dict(self.audit,proportion_status='uncertain')
        self.assertEqual(_verdict(a,self.render,None,False),'keep')
        self.assertEqual(_verdict(a,self.render,None,True),'review')

    def test_retry_only_for_proportions_not_wrong_input(self):
        a=dict(self.audit,proportion_status='severely_mismatched')
        entry=lambda q:dict(entry=dict(quality=dict(audit=q)))
        self.assertTrue(needs_shape_retry(entry(a)))
        self.assertFalse(needs_shape_retry(entry(dict(a,category_match=False))))
        self.assertFalse(needs_shape_retry(entry(dict(a,severe_defects=['wrong_category']))))
        self.assertFalse(needs_shape_retry(entry(self.audit)))

    def test_retry_removes_real_inputs_and_preserves_identity_and_original(self):
        original=dict(signature='parent',config=dict(seed=42))
        row=dict(item_index=7,obj_id='bus1',category='bus',real_images=[dict(rgba_path='/real.png')],
                 real_view_selection=dict(count=1),prompts=['front','side'],stages={'flux':{}},status='generated',
                 raw_path='/old.ply',raw_sha256='old')
        before=deepcopy(row)
        for regenerate in [False,True]:
            bundle=retry_bundle(original,row,regenerate,'city bus')
            job=bundle['assets'][0]
            self.assertEqual(job['obj_id'],'bus1')
            self.assertEqual(job['real_images'],[])
            self.assertEqual(job['input_mode'],'flux_only')
            self.assertNotIn('raw_path',job)
            self.assertEqual(job['prompts']==row['prompts'],not regenerate)
            self.assertEqual(bundle['config']['seed'],43)
        self.assertEqual(row,before)

    def test_strict_json_schema_has_no_optional_new_fields(self):
        Draft202012Validator.check_schema(SCHEMA)
        self.assertEqual(set(SCHEMA['properties']),set(SCHEMA['required']))
        self.assertFalse(SCHEMA['additionalProperties'])

    def test_short_bus_cannot_be_relabelled_to_pass_but_minibus_is_allowed(self):
        row=dict(category='bus',prompts=['long, rectangular public transit bus'])
        shape=dict(size_xyz=[5,2.5,3])
        audit,guard=proportion_guard(row,shape,self.audit)
        self.assertTrue(guard['triggered'])
        self.assertEqual(audit['proportion_status'],'severely_mismatched')
        row['prompts']=['minibus']
        self.assertFalse(proportion_guard(row,shape,self.audit)[1]['triggered'])

    def test_retry_cannot_change_original_subtype_constraint(self):
        row=dict(item_index=0,obj_id='bus',category='bus',real_images=[],real_view_selection={},
                 prompts=['full-size public transit bus'])
        job=retry_bundle(dict(signature='parent',config=dict(seed=1)),row,True,'minibus')['assets'][0]
        self.assertIn('minibus',job['prompts'][0])
        audit,guard=proportion_guard(job,dict(size_xyz=[5,2.5,3]),self.audit)
        self.assertTrue(guard['triggered'])
        self.assertEqual(guard['minimum'],3.0)

        row=dict(category='truck',prompts=['semi-truck cab with partial trailer'])
        self.assertTrue(proportion_guard(row,dict(size_xyz=[6.75,2.5,3]),self.audit)[1]['triggered'])
        row['prompts']=['small box truck']
        self.assertFalse(proportion_guard(row,dict(size_xyz=[5,2.5,3]),self.audit)[1]['triggered'])

    def test_retry_failure_does_not_restore_malformed_original_or_double_count(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);image=root/'flux.png';image.write_bytes(b'test_image_bytes')
            row=dict(obj_id='bus1',item_index=0,category='bus',real_images=[{'rgba_path':'original_real.png'}],
                real_view_selection={},prompts=['public transit bus'],status='generated',
                stages={'flux':dict(status='succeeded',images=[dict(path=str(image),sha256='fixture')])})
            generation=dict(signature='parent',config=dict(seed=42),assets=[row])
            path=write_json(root/'generation.json',generation)
            audit=dict(self.audit,proportion_status='severely_mismatched')
            state=dict(assets={'bus1':dict(status='reject',entry=dict(quality=dict(audit=audit),
                label=dict(sub_type='bus')))})
            client=AsyncMock()
            client.call_json.return_value=dict(proportion_status='acceptable',reason='complete generated view')
            async def generated(bundle,work,gpus):
                self.assertEqual(bundle['assets'][0]['real_images'],[])
                self.assertEqual(bundle['assets'][0]['obj_id'],'bus1')
                return dict(assets=[dict(bundle['assets'][0],status='failed')])
            config=type('Config',(),{'vlm':type('VLM',(),{'concurrency':2})()})()
            with patch('asset_extraction.quality.AsyncOpenAIClient',return_value=client), \
                 patch('asset_extraction.quality.generate',side_effect=generated) as generate, \
                 patch('asset_extraction.export.curate',new=AsyncMock(return_value=dict(
                     assets={'bus1':dict(status='failed',error='test renderer failure')},counts=dict(queries=1,failed=1)))):
                final,selected=asyncio.run(repair_shapes(path,state,config,root/'retry',renderer_python='python',gpus=[0]))
                self.assertEqual(final['counts'],dict(queries=1,failed=1))
                self.assertEqual(len(selected['assets']),1)
                self.assertEqual(selected['assets'][0]['obj_id'],'bus1')
                self.assertEqual(final['assets']['bus1']['shape_retry_history']['attempt'],1)
                asyncio.run(repair_shapes(path,state,config,root/'retry',renderer_python='python',gpus=[0]))
                self.assertEqual(generate.call_count,1)


if __name__=='__main__':unittest.main()
