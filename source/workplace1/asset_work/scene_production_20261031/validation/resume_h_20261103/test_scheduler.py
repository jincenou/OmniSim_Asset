import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run
from asset_extraction.io_utils import write_json,read_json


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,stop_after=None,vehicles_only=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scenes=['a','b','c'];log=[];calls=[]
            write_json(root/'dataset_control.json',dict(nuscenes='running',waymo='paused',**(dict(other_queue_enabled=False,generation_categories=['car','truck','van','bus'],required_route='H') if vehicles_only else {})))
            write_json(root/'inputs/nuscenes/complete.json',dict(scenes=scenes))
            write_json(root/'nuscenes_status.json',dict(dataset='nuscenes',kept=24,kept_other=12,queries=24,
                chunks=[],scenes={'a':dict(offset=12,vehicle_queries=12,other_offset=12,status='processing')}))
            for scene,count,other in [('a',12,312),('b',96,12),('c',48,12)]:
                work=root/'runs'/f'nuscenes_{scene}'
                vehicles=[dict(obj_id=f'{scene}_v{i}',scene_id=scene,category='car',route='H') for i in range(count)]
                others=[dict(obj_id=f'{scene}_o{i}',scene_id=scene,category='barrier') for i in range(other)]
                write_json(work/'handoff/flux_input.json',vehicles+others)
                write_json(work/'handoff/asset_routes.json',dict(assets=vehicles+others))
                write_json(work/'vehicle_selection.json',dict(rows=vehicles,duplicates=[]))
                write_json(work/'production_selection.json',dict(rows=vehicles,routes=vehicles,removed=[],protocol='fixture'))
                write_json(root/'identity'/f'nuscenes_{scene}.json',{})
            reg=dict(nuscenes=[dict(scene_id=s) for s in scenes],existing_complete_pilot_scenes={'nuscenes':[]})
            async def frontend(dataset,item):return root/'runs'/f'{dataset}_{item["scene_id"]}',object()
            async def process(chunk,rows,*args):
                log.append((rows[0]['scene_id'],'V' if rows[0]['category']=='car' else 'O'))
                calls.extend(r['obj_id'] for r in rows)
                if stop_after and len(log)==stop_after:run.DRAIN=True
                return dict(queries=len(rows),kept=len(rows))
            with patch.object(run,'ROOT',root),patch.object(run,'DRAIN',False), \
                 patch.object(run,'FRONTEND_TASKS',{}),patch.object(run,'registry',return_value=reg), \
                 patch.object(run,'_frontend',side_effect=frontend),patch.object(run,'load_config',return_value=object()), \
                 patch.object(run,'prefetch_next'),patch.object(run,'process_chunk',side_effect=process):
                await run.dataset_run('nuscenes')
                if stop_after:
                    self.assertEqual(read_json(root/'nuscenes_status.json')['status'],'drained')
                    run.DRAIN=False
                    await run.dataset_run('nuscenes')
            self.assertEqual(len(calls),len(set(calls)))
            self.assertFalse(any(x.startswith('a_v') for x in calls))
            self.assertFalse(any(f'a_o{i}' in calls for i in range(12)))
            if vehicles_only:
                self.assertEqual(log,[('b','V')]*4+[('c','V')]*2)
                state=read_json(root/'nuscenes_status.json')
                self.assertEqual(state['kept_other'],12)
                self.assertTrue(state['other_queue'])
                self.assertEqual(state['queries'],24+96+48)
                return
            self.assertEqual(log[:8],[('b','V'),('b','V'),('b','V'),('a','O'),('b','V'),('c','V'),('c','V'),('b','O')])
            state=read_json(root/'nuscenes_status.json')
            self.assertEqual(state['other_queue'],[])
            self.assertEqual(state['kept_other'],336)
            self.assertTrue(all(p['status']=='completed' for p in state['scenes'].values()))
            self.assertEqual(state['queries'],24+96+48+300+12+12)

    async def test_h_only_preserves_other_queue_and_resume(self):
        await self.exercise(stop_after=2,vehicles_only=True)

    async def test_cross_scene_priority_interleaving_and_unlimited_other(self):
        await self.exercise()

    async def test_boundary_restart_preserves_offsets_credit_and_query_identity(self):
        await self.exercise(stop_after=3)


if __name__=='__main__':unittest.main()
