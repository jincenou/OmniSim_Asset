"""Pause new admissions without interrupting an admitted asset's export."""
import sys
import asyncio
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import batch


class ProductionPauseTests(unittest.TestCase):
    def test_device_handoff_requires_whole_asset_and_all_workers_drained(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'ROOT', Path(directory)):
            root=Path(directory)
            plan=dict(drain_ids=['HN0001'],previous_gpus=[0,1],gpus=[2,3,4,5,6,7])
            policy=dict(paused=True,drain_ids=plan['drain_ids'],next_gpus=plan['gpus'])
            common.save(root/'PRODUCTION_PAUSE.json',policy)
            common.save(root/'jobs/HN0001.json',dict(state='running',stage='reconstruction_A'))
            common.save(root/'jobs/worker_gpu0.json',dict(state='running'))
            common.save(root/'jobs/worker_gpu1.json',dict(state='paused_pending_asset_metadata_validation'))
            self.assertFalse(batch.handoff_ready(plan))
            common.save(root/'jobs/HN0001.json',dict(state='replayed_visual_review_pending'))
            self.assertFalse(batch.handoff_ready(plan))
            common.save(root/'jobs/worker_gpu0.json',dict(state='paused_pending_asset_metadata_validation'))
            self.assertTrue(batch.handoff_ready(plan))
            policy['paused']=False
            common.save(root/'PRODUCTION_PAUSE.json',policy)
            with self.assertRaisesRegex(RuntimeError,'policy changed'):
                batch.handoff_ready(plan)

    def test_handoff_retries_only_explicitly_prepared_unchanged_failures(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)), \
             patch.object(batch,'verified_package',side_effect=lambda i:i=='HN0001'):
            root=Path(directory);ids=['HN0001','HN0002','HN0003','HN0004','HN0005']
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=ids))
            for identity in ['HN0002','HN0003','HN0004']:
                common.save(root/'jobs'/f'{identity}.json',dict(state='failed_or_needs_review',error='CUDA OOM'))
            plan=dict(cases=ids,prepared_failures={i:common.sha(root/'jobs'/f'{i}.json')
                                                for i in ['HN0002','HN0003']})
            common.save(root/'jobs/HN0003.json',dict(state='failed_or_needs_review',error='New failure'))
            ready,withheld=batch.handoff_cases(plan)
            self.assertEqual(ready,['HN0002','HN0005'])
            self.assertEqual([x['identity'] for x in withheld],['HN0003','HN0004'])
            common.save(root/'jobs/HN0005.json',dict(state='running'))
            with self.assertRaisesRegex(RuntimeError,'unexpected job state'):
                batch.handoff_cases(plan)

    def test_drain_ids_and_future_queue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)):
            self.assertFalse(common.production_paused("H0001"))
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            self.assertFalse(common.production_paused("H0001"))
            self.assertTrue(common.production_paused("H0002"))
            self.assertTrue(common.production_paused())

    def test_old_controller_child_waits_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "frontend.py"), "generate", "H0002"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            def release(_):
                common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": False})
            with patch.object(common.time, "sleep", side_effect=release) as sleep:
                common.wait_for_production_permission()
                self.assertEqual(sleep.call_count, 1)
            receipt = next((common.ROOT / "jobs").glob("pause_wait_*.json"))
            self.assertEqual(common.read(receipt)["state"], "resumed")

    def test_admitted_child_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "ROOT", Path(directory)), \
             patch.object(sys, "argv", [str(common.CODE_ROOT / "batch.py"), "package", "H0001"]), \
             patch.dict(common.os.environ, {"HUMAN_GPU_UUID": "GPU-test"}):
            common.save(common.ROOT / "PRODUCTION_PAUSE.json", {"paused": True, "drain_ids": ["H0001"]})
            with patch.object(common.time, "sleep") as sleep:
                common.wait_for_production_permission()
                sleep.assert_not_called()


class SupplementalQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_gpu_generation_is_left_with_its_owner(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)), \
             patch.object(batch,'ASSET_ROOT',Path(directory)/'assets'):
            root=Path(directory); ids=['H0001','H0002','HN0003']
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=ids,target_count=3))
            common.save(root/'BATCH_APPROVAL.json',dict(approved=True,demo_ids=ids[:2]))
            common.save(root/'selection.json',dict(demo_ids=ids[:2]))
            for identity in ids[:2]:
                common.save(root/'assets'/identity/'visual_review.json',dict(allow_batch_baseline=True))
            common.save(root/'jobs/HN0003.json',dict(state='running',stage='frontend_generate',gpu=4))
            completed=set(ids[:2]); reused=[]; real_sleep=asyncio.sleep
            with (root/'jobs/HN0003.lock').open('a+') as owner:
                fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                async def release_owner(_):
                    await real_sleep(.01)
                    completed.add('HN0003')
                    fcntl.flock(owner,fcntl.LOCK_UN)
                async def execute(identity,gpu):
                    self.assertIn(identity,completed)
                    reused.append(identity)
                with patch.object(batch,'case',return_value=dict(input_review=dict(allow_generation=True))), \
                     patch.object(batch,'verified_package',side_effect=lambda i:i in completed), \
                     patch.object(batch,'active_child',return_value=[12345]), \
                     patch.object(batch,'memory_free',return_value=48000), \
                     patch.object(batch,'required_memory',return_value=24000), \
                     patch.object(batch,'production_paused',return_value=False), \
                     patch.object(batch,'execute',side_effect=execute), \
                     patch.object(batch.asyncio,'sleep',side_effect=release_owner):
                    await asyncio.wait_for(batch.main(['run','--cases','HN0003','--gpus','0',
                                                       '--queue-name','foreign']),timeout=3)
            self.assertEqual(reused,['HN0003'])

    async def test_two_named_queues_do_not_optimize_the_same_asset_twice(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)), \
             patch.object(batch,'ASSET_ROOT',Path(directory)/'assets'):
            root=Path(directory);ids=['H0001','H0002','HN0003']
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=ids,target_count=3,input_selection_complete=True))
            common.save(root/'BATCH_APPROVAL.json',dict(approved=True,demo_ids=ids[:2]))
            common.save(root/'selection.json',dict(demo_ids=ids[:2]))
            for identity in ids[:2]:
                common.save(root/'assets'/identity/'visual_review.json',dict(allow_batch_baseline=True))
            completed=set(ids[:2]);optimizations=[];real_sleep=asyncio.sleep
            async def short_sleep(_):await real_sleep(.01)
            async def execute(identity,gpu):
                if identity in completed:return
                optimizations.append((identity,gpu))
                await real_sleep(.06)
                completed.add(identity)
            spec=dict(input_review=dict(allow_generation=True))
            with patch.object(batch,'case',return_value=spec), \
                 patch.object(batch,'verified_package',side_effect=lambda i:i in completed), \
                 patch.object(batch,'memory_free',return_value=48000), \
                 patch.object(batch,'required_memory',return_value=24000), \
                 patch.object(batch,'production_paused',return_value=False), \
                 patch.object(batch,'execute',side_effect=execute), \
                 patch.object(batch.asyncio,'sleep',side_effect=short_sleep):
                await asyncio.wait_for(asyncio.gather(
                    batch.main(['run','--cases','HN0003','--gpus','0','--queue-name','extra0']),
                    batch.main(['run','--cases','HN0003','--gpus','1','--queue-name','extra1'])),timeout=3)
            self.assertEqual(len(optimizations),1)
            self.assertEqual(completed,set(ids))
            self.assertTrue((root/'queue_exit_extra0.json').exists())
            self.assertTrue((root/'queue_exit_extra1.json').exists())

    async def test_named_queue_cannot_implicitly_claim_the_whole_pool(self):
        with self.assertRaisesRegex(AssertionError,'explicit admitted'):
            await batch.main(['run','--all','--gpus','0','--queue-name','extra'])


class DeviceHandoffProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_controllers_drain_before_front_four_queue(self):
        from audit import watched_process
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)):
            root=Path(directory)
            controllers=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
                'run.py','batch','run'],cwd=batch.CODE_ROOT,
                env=dict(os.environ,HUMAN_RUN_ROOT=str(root))) for _ in range(2)]
            real_sleep=asyncio.sleep
            async def short_sleep(_):await real_sleep(.01)
            launched=[]
            async def launch(*args,**kwargs):
                self.assertTrue(all(p.poll() is not None for p in controllers))
                self.assertFalse(common.read(root/'PRODUCTION_PAUSE.json')['paused'])
                at=args.index('--gpus')
                self.assertEqual(args[at+1:at+5],('0','1','2','3'))
                launched.append(args)
                class Child:
                    pid=999999
                    async def wait(self):return 0
                return Child()
            plan=dict(state='prepared',approved=True,
                controllers=[dict(pid=p.pid,start_ticks=watched_process(p.pid)) for p in controllers],
                gpus=[0,1,2,3],previous_gpus=[0,4],drain_ids=['HN0001'],
                cases=['HN0002'],prepared_failures={},min_free_mib=30000)
            common.save(root/'PRODUCTION_PAUSE.json',dict(paused=True,drain_ids=['HN0001'],next_gpus=plan['gpus']))
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0001','HN0002']))
            common.save(root/'jobs/HN0001.json',dict(state='running'))
            for gpu in [0,4]:common.save(root/'jobs'/f'worker_gpu{gpu}.json',dict(state='running'))
            path=root/'handoff.json';common.save(path,plan)
            try:
                with patch.object(batch,'verified_package',return_value=False), \
                     patch.object(batch.asyncio,'sleep',side_effect=short_sleep), \
                     patch.object(batch.asyncio,'create_subprocess_exec',side_effect=launch):
                    task=asyncio.create_task(batch.handoff_queue(path))
                    await real_sleep(.06)
                    self.assertTrue(all(p.poll() is None for p in controllers))
                    self.assertFalse(launched)
                    common.save(root/'jobs/HN0001.json',dict(state='replayed_visual_review_pending'))
                    for gpu in [0,4]:common.save(root/'jobs'/f'worker_gpu{gpu}.json',dict(state='paused_pending_asset_metadata_validation'))
                    await asyncio.wait_for(task,timeout=3)
                self.assertEqual(len(launched),1)
                self.assertEqual(common.read(path)['state'],'completed_pending_quality_review')
            finally:
                for p in controllers:
                    if p.poll() is None:p.terminate()
                    p.wait(timeout=3)

    async def test_live_child_prevents_stopping_any_controller(self):
        from audit import watched_process
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)):
            root=Path(directory)
            idle=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                'run.py','batch','run'],cwd=batch.CODE_ROOT,env=dict(os.environ,HUMAN_RUN_ROOT=str(root)))
            parent=subprocess.Popen([sys.executable,'-c',
                'import subprocess,sys,time;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);print(p.pid,flush=True);time.sleep(30)',
                'run.py','batch','run'],cwd=batch.CODE_ROOT,env=dict(os.environ,HUMAN_RUN_ROOT=str(root)),stdout=subprocess.PIPE,text=True)
            child=int(parent.stdout.readline())
            plan=dict(state='prepared',approved=True,controllers=[dict(pid=p.pid,start_ticks=watched_process(p.pid)) for p in [idle,parent]],
                gpus=[0,1,2,3],previous_gpus=[0,4],drain_ids=['HN0001'],cases=['HN0002'],prepared_failures={},min_free_mib=30000)
            common.save(root/'PRODUCTION_PAUSE.json',dict(paused=True,drain_ids=['HN0001'],next_gpus=plan['gpus']))
            common.save(root/'jobs/HN0001.json',dict(state='replayed_visual_review_pending'))
            for gpu in [0,4]:common.save(root/'jobs'/f'worker_gpu{gpu}.json',dict(state='paused_pending_asset_metadata_validation'))
            path=root/'handoff.json';common.save(path,plan)
            try:
                with self.assertRaisesRegex(AssertionError,'still has children'):
                    await batch.handoff_queue(path)
                self.assertIsNone(idle.poll())
                self.assertIsNone(parent.poll())
                self.assertTrue(common.read(root/'PRODUCTION_PAUSE.json')['paused'])
            finally:
                os.kill(child,15)
                for p in [idle,parent]:
                    if p.poll() is None:p.terminate()
                    p.wait(timeout=3)
                parent.stdout.close()

    async def test_real_idle_controller_exit_precedes_restricted_queue_launch(self):
        from audit import watched_process
        with tempfile.TemporaryDirectory() as directory, patch.object(batch,'ROOT',Path(directory)):
            root=Path(directory)
            controller=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
                'run.py','batch','run'],cwd=batch.CODE_ROOT,
                env=dict(os.environ,HUMAN_RUN_ROOT=str(root)))
            real_sleep=asyncio.sleep
            launched=[]
            async def short_sleep(_):
                await real_sleep(.01)
            async def launch(*args,**kwargs):
                self.assertIsNotNone(controller.poll(), 'New queue started before old controller exited')
                self.assertFalse(common.read(root/'PRODUCTION_PAUSE.json')['paused'])
                launched.append(args)
                class Child:
                    pid=999999
                    async def wait(self):return 0
                return Child()
            plan=dict(state='prepared',approved=True,controller_pid=controller.pid,
                controller_start_ticks=watched_process(controller.pid),gpus=[2,3,4,5,6,7],
                previous_gpus=[0,1],drain_ids=['HN0001'],cases=['HN0002'],
                prepared_failures={},min_free_mib=30000)
            common.save(root/'PRODUCTION_PAUSE.json',dict(paused=True,drain_ids=['HN0001'],next_gpus=plan['gpus']))
            common.save(root/'PRODUCTION_INPUTS.json',dict(ids=['HN0001','HN0002']))
            common.save(root/'jobs/HN0001.json',dict(state='running'))
            common.save(root/'jobs/worker_gpu0.json',dict(state='running'))
            common.save(root/'jobs/worker_gpu1.json',dict(state='paused_pending_asset_metadata_validation'))
            path=root/'handoff.json';common.save(path,plan)
            try:
                with patch.object(batch,'verified_package',return_value=False), \
                     patch.object(batch.asyncio,'sleep',side_effect=short_sleep), \
                     patch.object(batch.asyncio,'create_subprocess_exec',side_effect=launch):
                    task=asyncio.create_task(batch.handoff_queue(path))
                    await real_sleep(.08)
                    self.assertIsNone(controller.poll(), 'Running round was interrupted')
                    self.assertFalse(launched)
                    common.save(root/'jobs/HN0001.json',dict(state='replayed_visual_review_pending'))
                    common.save(root/'jobs/worker_gpu0.json',dict(state='paused_pending_asset_metadata_validation'))
                    await asyncio.wait_for(task,timeout=3)
                args=launched[0];at=args.index('--gpus')
                self.assertEqual(args[at+1:at+7],('2','3','4','5','6','7'))
                self.assertEqual(common.read(path)['state'],'completed_pending_quality_review')
            finally:
                if controller.poll() is None:controller.terminate()
                controller.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
