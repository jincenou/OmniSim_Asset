"""Summarize verified artifacts without promoting loader checks to reconstruction."""
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def main():
    inp=read('research/input-validation.json')
    assert inp['state']=='passed'
    smoke=read('outputs/smoke/preview/final_validation.json')
    turbo=read('outputs/turbo/preview/final_validation.json')
    assert smoke['state']==turbo['state']=='validated'
    assert smoke['visual_review']=='accepted_for_24_view_test_with_limitations'
    assert turbo['visual_review']=='accepted_for_bridge_probe_with_limitations'
    assert turbo['raw_decoded_pts_checked']==24*121
    metadata=read('outputs/turbo/metadata.json')
    parallel=metadata['generation']['parallelism']
    assert parallel['backend']=='nccl' and len(parallel['workers'])==4
    assert set(parallel['used_devices'])=={'cuda:0','cuda:1','cuda:2','cuda:3'}
    bridge=read('bridge/waymo_validation.json')
    assert bridge['status']=='passed_actual_backend_loader' and bridge['rgba_samples_checked']==72
    assert read('bridge/validation.json')['status']=='passed_actual_backend_loader'
    for label in ('official_sample','waymo_sample'):
        assert len(list((ROOT/'bridge'/label/'images').glob('*/*.png')))==72
    for name in ('generate-smoke','generate-turbo','bridge-waymo'):
        assert (ROOT/'logs'/f'{name}.exit').read_text().strip()=='0'
    report=ROOT/'WAYMO_4DANYONE_RESEARCH.md'
    links=re.findall(r'\]\(([^)]+)\)',report.read_text())
    missing=[p for p in links if not p.startswith(('http:','https:','#')) and not (ROOT/p).exists()]
    assert not missing,missing
    assert not git(ROOT.parent/'repo','status','--porcelain')
    assert not git(ROOT/'bridge/backend','status','--porcelain')
    scripts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').glob('*')) if p.is_file()}
    result={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'status':'minimum_waymo_generation_and_bridge_validation_complete',
            'input':inp,'smoke':smoke,'turbo':turbo,'waymo_bridge':bridge,
            'generation_physical_gpu_ids':[4,5,6,7],
            'preflight_device_incident':read('experiments/device-incident.json'),
            'gpu_mapping_evidence':'logs/generate-turbo.log records physical GPU UUIDs; outputs/turbo/metadata.json records four logical CUDA workers',
            'official_repository_clean':True,'backend_repository_clean':True,
            'commits':{'4DAnyone':git(ROOT.parent/'repo','rev-parse','HEAD'),
                       'FreeTimeGsVanilla':git(ROOT/'bridge/backend','rev-parse','HEAD')},
            'experiment_script_sha256':scripts,'report_local_links_checked':len(links),
            'free_disk_bytes':shutil.disk_usage(ROOT).free,
            'not_done':['FreeTimeGS CUDA optimization or reconstructed rendering','Full 121-frame bridge export','Waymo world/scale/root placement alignment','Walking-cycle validation','DriveLab trajectory reuse and pose retargeting'],
            'training_started':False}
    (ROOT/'research/final-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'report_links':len(links),'waymo_points':bridge['initialization_points'],'training_started':False}),flush=True)


if __name__=='__main__':main()
