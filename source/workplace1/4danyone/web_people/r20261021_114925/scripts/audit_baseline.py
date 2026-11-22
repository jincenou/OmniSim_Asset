"""Freeze historical evidence and verify existing model/asset bytes without modification."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/mnt/ojc/workplace1/4danyone')
D=PROJECT/'waymo/frontend_general_v1/validation12/downstream'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024**2):h.update(chunk)
    return h.hexdigest()


async def main():
    evidence=ROOT/'research/baseline_evidence'
    evidence.mkdir(exist_ok=False)
    paths=[D/'PLAN.json',D/'PREP_READY_PROTOCOL.json',
        D/'runs/P31_G0_A/frontend_provenance.json',D/'runs/P31_G0_A/complete.json',
        D/'runs/P31_G0_A/snapshots/53000.json',D/'runs/P31_G0_A/validation/final_replay.json',
        D/'runs/P31_G0_A/config.json',D/'runs/P31_G0_A/asset_camera_data.json',
        PROJECT/'waymo/scene20/r20261019/scenes/092/P31/generation.json',
        PROJECT/'outputs/turbo/metadata.json',
        PROJECT/'waymo/frontend_general_v1/OPTIMIZATION_HANDOFF_20261020.md']
    records=[]
    for i,path in enumerate(paths):
        dest=evidence/f'{i:02d}_{path.name}'
        shutil.copyfile(path,dest)
        records.append(dict(source=str(path),copy=str(dest.relative_to(ROOT)),sha256=digest(dest)))
    repos={}
    for name,path in [('4danyone',PROJECT/'repo'),('gvhmr',PROJECT/'repo/third_party/GVHMR'),
                      ('grid4d',PROJECT/'waymo/grid4d/source')]:
        def git(*args):return subprocess.check_output(['git','-C',str(path),*args],text=True)
        patch=evidence/f'{name}.patch';patch.write_text(git('diff','HEAD','--'))
        repos[name]=dict(path=str(path),commit=git('rev-parse','HEAD').strip(),
            working_tree=git('status','--porcelain'),patch=str(patch.relative_to(ROOT)),patch_sha256=digest(patch))
    weights=json.loads((PROJECT/'research/download-manifest.json').read_text())
    selected=[r for r in weights if '/models/4danyone/' in r['path']]
    sem=asyncio.Semaphore(2)
    async def verify(row):
        async with sem:
            path=Path(row['path']);actual=await asyncio.to_thread(digest,path)
            assert actual==row['sha256'] and path.stat().st_size==row['size'],str(path)
            return dict(row,verified_at=time.time())
    verified=await asyncio.gather(*(verify(r) for r in selected))
    asset=D/'runs/P31_G0_A'
    snap=json.loads((asset/'snapshots/53000.json').read_text())
    asset_hashes={}
    for name,expected in snap['files'].items():
        actual=await asyncio.to_thread(digest,asset/name)
        assert actual==expected,name
        asset_hashes[name]=actual
    environments={}
    async def env(name):
        python=f'/mnt/ojc/miniconda3/envs/{name}/bin/python'
        proc=await asyncio.create_subprocess_exec(python,'-m','pip','freeze',stdout=asyncio.subprocess.PIPE)
        output,_=await proc.communicate();assert proc.returncode==0
        path=evidence/f'{name}_requirements.txt';path.write_bytes(output)
        environments[name]=dict(python=python,requirements=str(path.relative_to(ROOT)),sha256=digest(path))
    await asyncio.gather(env('4danyone'),env('grid4d'))
    protocol=dict(state='method_locked_input_tracks_pending',frozen_at=time.time(),
        scope='Pre-frontend-repair G0 + established full Grid4D A; not original paper backend defaults',
        camera_authorization=json.loads((ROOT/'research/CAMERA_AUTHORIZATION.json').read_text()),
        frontend=dict(seed=42,turbo=True,steps=4,rcp=True,tcr=True,views=24,pitch_degrees=15,
            yaw_span=360,attention_backend='sdpa',frames=121,fps=25,width=704,height=1280,
            temporal_sampling='nearest real source frame at 25 Hz; retain exact PTS mapping; no synthesized intermediate images',
            crop='source-bbox-following constant dimensions, 1.3 margin and 11:20 ratio from old m02_prepare',
            foreground='old SAM2 bbox plus three fixed bbox-relative points, straight alpha over white',
            motion='official ordinary-video GVHMR static_cam=True approximation, estimated K',
            no_new_repairs=True),
        backend=json.loads((D/'PLAN.json').read_text())['training'],
        source_repositories=repos,environments=environments,weights=verified,
        historical_evidence=records,historical_asset_verified=asset_hashes,
        historical_replay='P31_G0_A: independent_dynamic_replay_passed; source_reads=0',
        adaptation='Replace Waymo file/calibration loader with authorized ordinary-video input. Preserve model recipes.',
        selected_tracks=None,new_assets_completed=0,allowed_physical_gpus=[4,5,6,7])
    path=ROOT/'BASELINE_LOCK.json';assert not path.exists()
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    print('Historical weights and complete exported asset verified; baseline method locked',flush=True)


if __name__=='__main__':asyncio.run(main())
