"""Bind the same frozen backend to a visually checked accessory-input ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
PERSON = 'W02'
TRIAL = ROOT / f'diagnostics/accessory_{PERSON}_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    review = json.loads((TRIAL / 'GENERATED_VISUAL_REVIEW.json').read_text())
    expected_review = {'W02':'black_bag_restored_in_reviewed_generated_frames',
                       'W04':'pink_accessory_restored_in_reviewed_generated_frames'}
    assert review['state'] == expected_review[PERSON]
    validation = json.loads((TRIAL / 'frontend' / PERSON / 'research/generated_validation.json').read_text())
    assert validation['state'] == 'all24_videos_decoded_same_retained_cameras'
    assert len(validation['videos']) == 24
    for row in validation['videos']:
        assert sha(TRIAL / 'frontend' / PERSON / 'outputs/turbo/videos' / f'{row["view"]:02d}.mp4') == row['sha256']
    assert not (TRIAL / 'backend').exists()
    for relative in ['scripts', 'backend/scripts', 'backend/configs', 'backend/research',
                     'backend/four_dg/data', 'backend/runs', 'backend/logs', 'backend/tmp',
                     'logs', 'tmp', f'inputs/{PERSON}/inputs']:
        (TRIAL / relative).mkdir(parents=True, exist_ok=True)
    copied = []
    sources = list((ROOT / 'backend/scripts').glob('*.py'))
    sources += [ROOT / 'scripts' / name for name in [
        'prepare_person.py', 'source_io.py', 'runtime_env.py', 'execute_stage.py',
        'run_leased.py', 'run_people_queue.py', 'package_asset.py', 'publish_previews.py']]
    sources += [ROOT / name for name in ['SELECTED_TRACKS.json', 'SELECTED_TRACKS.sha256',
                                         'BASELINE_LOCK.json', 'backend/PLAN.json', 'backend/plan_hash.txt',
                                         'backend/research/source.patch']]
    for source in sources:
        relative = source.relative_to(ROOT)
        target = TRIAL / relative
        assert not target.exists()
        shutil.copyfile(source, target)
        assert sha(target) == sha(source)
        copied.append(dict(source=str(source), target=str(target), sha256=sha(target)))
    # New-process replay must also reject symlink-resolved original source data.
    replay = TRIAL / 'backend/scripts/replay_asset.py'
    text = replay.read_text()
    needle = '    def guard(event,args):'
    assert text.count(needle) == 1
    extra = [str(ROOT / name) for name in ['inputs', 'raw', 'sources', 'frontend', 'rgba', 'backend/four_dg', 'backend/runs']]
    text = text.replace(needle, '    denied.extend(Path(p) for p in ' + repr(extra) + ')\n' + needle)
    replay.write_text(text)
    # This is one ablation, not another eight-person gallery.
    preview = TRIAL / 'scripts/publish_previews.py'
    text = preview.read_text()
    needle = "    else:selected(a.id);{'generated':generated,'comparison':comparison}[a.stage](a.id);gallery()"
    assert needle in text
    preview.write_text(text.replace(needle, "    else:selected(a.id);{'generated':generated,'comparison':comparison}[a.stage](a.id)"))
    for name, source in [
        ('source_clip.mp4', ROOT / 'inputs' / PERSON / 'source_clip.mp4'),
        ('source_crop.mp4', TRIAL / 'corrected_input.mp4'),
        ('pose_preview.mp4', ROOT / 'inputs' / PERSON / 'pose_preview.mp4'),
        ('inputs/frame_mapping.json', ROOT / 'inputs' / PERSON / 'inputs/frame_mapping.json')]:
        assert source.is_file()
        target = TRIAL / 'inputs' / PERSON / name
        if name == 'inputs/frame_mapping.json':
            mapping = json.loads(source.read_text())
            mapping['original_baseline_mapping_sha256'] = sha(source)
            mapping['original_baseline_video_sha256'] = mapping.get('video_sha256')
            mapping['video_sha256'] = sha(TRIAL / 'corrected_input.mp4')
            mapping['appearance_change'] = 'Accessory alpha union only; original source/frame/timestamp/crop mapping unchanged'
            target.write_text(json.dumps(mapping, indent=2)+'\n')
        else:
            target.symlink_to(source)
    runner = TRIAL / 'scripts/run_backend.py'
    runner.write_text('''"""Continue only the validated W02 repair through unchanged backend stages."""
import asyncio,fcntl,os,time,json,hashlib
from pathlib import Path
import run_people_queue as q

async def main():
    lock=(q.ROOT/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (q.ROOT/'queue_status.json').exists()
    q.STATE.update(state='running',phase='accessory_backend',pid=os.getpid(),started_at=time.time(),people={'W02':{'state':'pending'}})
    q.save()
    try:
        stages=[('masks',True,14000),('hull',False,0),('manifest',False,0),
                ('geometry',False,0),('configure',False,0),('native',True,24000),
                ('alpha',True,24000),('A',True,24000),('package',False,0),
                ('replay',True,10000),('comparison',False,0)]
        for name,gpu,memory in stages:
            await q.stage('W02',name,gpu,memory)
            if name=='masks':
                review_path=q.ROOT/'MASK_ACCESSORY_REVIEW.json'
                q.STATE['people']['W02'].update(state='downstream_accessory_review_required')
                q.save()
                while not review_path.exists():
                    await asyncio.sleep(30)
                checked=json.loads(review_path.read_text())
                assert checked['allow_backend_training'] is True
                report=q.ROOT/'backend/research/W02-mask-ready-complete.json'
                assert hashlib.sha256(report.read_bytes()).hexdigest()==checked['mask_report_sha256']
        q.STATE['people']['W02'].update(state='asset_replayed_quality_unreviewed')
    except Exception as error:
        q.STATE['people']['W02'].update(state='failed',error=repr(error),failed_at=time.time())
        raise
    finally:
        q.STATE.update(state='phase_finished',finished_at=time.time())
        q.save()

if __name__=='__main__':asyncio.run(main())
'''.replace('W02', PERSON))
    import ast
    for file in [*sources, replay, preview, runner]:
        if file.suffix == '.py':
            ast.parse(file.read_text())
    binding = dict(created_at=time.time(), parent=str(ROOT), root=str(TRIAL), person=PERSON,
        scope='Single source-accessory-mask ablation, same locked full A; not a ninth identity or unmodified G0 result',
        copies=copied, runtime_changes=['Extra denied source roots in replay', 'No duplicate eight-person gallery'],
        actual_scripts={str(f.relative_to(TRIAL)):sha(f) for folder in ['scripts','backend/scripts'] for f in (TRIAL/folder).glob('*.py')},
        training_updates=52999, generated_validation_sha256=sha(TRIAL/'frontend'/PERSON/'research/generated_validation.json'),
        source_review_sha256=sha(TRIAL/'GENERATED_VISUAL_REVIEW.json'), baseline_modified=False)
    (TRIAL / 'BACKEND_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    print('Separate corrected-input backend prepared; no models or datasets duplicated.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--person',choices=['W02','W04'],default='W02')
    PERSON=parser.parse_args().person;TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1'
    main()
