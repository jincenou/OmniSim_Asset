"""Move the complete A terminal asset into its delivery location without duplication."""
import json,os,shlex,sys
from pathlib import Path
from prepare_person import ROOT,selected,sha,save

def main():
    ident=sys.argv[1];choice=selected(ident)
    source=ROOT/'backend/runs'/f'{ident}_A';target=ROOT/'assets'/ident
    assert source.is_dir() and not target.exists()
    completed=json.loads((source/'complete.json').read_text())
    assert completed['total_actual_updates']==52999 and completed['extra_updates']==5000
    snapshot=json.loads((source/'snapshots/53000.json').read_text())
    for name,expected in snapshot['files'].items():assert sha(source/name)==expected
    cameras=json.loads((source/'asset_camera_data.json').read_text())
    assert cameras['heldout_views']==[0,8,16] and len(cameras['train_views'])==21
    for name in ['config.json','asset_camera_data.json','snapshots/53000_reference.npz']:
        assert (source/name).is_file()
    target.parent.mkdir(exist_ok=True);source.rename(target)
    save(target/'playback.json',dict(fps=25,frames=121,seconds=4.8,source_timestamps='../../inputs/'+ident+'/inputs/frame_mapping.json',
        scope='Source motion replay only; no arbitrary new action or metric world registration',
        generated_camera_space='normalized root-local human coordinates; generated cameras differ from source camera'))
    save(target/'package_manifest.json',dict(state='exported_independent_replay_pending',person=choice,
        model='Grid4D historical full A',actual_updates=52999,
        source_stage_directory=str(source),asset_directory=str(target),
        files={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))
    command=[str(ROOT/'scripts/execute_stage.py'),'replay',ident]
    replay='#!/bin/sh\nset -eu\n: "${CUDA_VISIBLE_DEVICES:?Set the authorized GPU UUID}"\n: "${FD_EXPECTED_GPU_UUID:?Set the matching GPU UUID}"\n: "${1:?Pass a new output directory as the first argument}"\nexec /mnt/ojc/miniconda3/envs/4danyone/bin/python '+shlex.join(command)+' --output "$1"\n'
    (target/'replay.sh').write_text(replay);os.chmod(target/'replay.sh',0o755)
    print(ident,'complete A exported; independent replay pending',flush=True)

if __name__=='__main__':main()
