"""Current file-backed delivery audit; historical failures remain unchanged."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
from datetime import datetime,timezone

R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    gallery=read(R/'gallery_manifest.json');rows=[];identities=set()
    for item in gallery['items']:
        if item['technical']!='已通过':continue
        identity=(str(item['scene']),item['track'])
        assert identity not in identities;identities.add(identity)
        if item['id']=='walk063_rootlocal24':
            root=R/'reconstruction/outputs/walk063_rootlocal24';proof=read(root/'demo/replay_validation.json')
            assert max(x['joint_translation_render_max_abs'] for x in proof['translation_checks'])<1e-5
            geometry=read(R/'experiments/walk063_rootlocal24_v1/research/root_local.json')
            assert all(x['all_joints_all_frames_inside'] for x in geometry['coverage'])
            rows.append({'label':item['id'],'identity':identity,'technical':True,'frames':121,'scope':'Dedicated root replay validation, separate from generic demo format','production_approved':False})
            continue
        demo=R/'demos'/item['id'];proof=read(demo/'validation.json');job=proof['job']
        asset=read(Path(job['checkpoint']).parents[1]/'asset.json');frames=asset['frames']
        assert frames in (61,121)
        for key,field in [('checkpoint','checkpoint_sha256'),('input','input_sha256'),('mapping','mapping_sha256'),('native_mapping','native_mapping_sha256')]:
            assert sha(Path(job[key]))==proof[field],(item['id'],key)
        for name,video in proof['videos'].items():
            assert sha(demo/name)==video['sha256']
            assert video['frames']==(72 if name=='orbit_fixed_time.mp4' else frames)
        metric_paths=[(R/'gallery'/unquote(p)).resolve() for p in item['links'].values() if p.endswith('/quality_metrics/metrics.json')]
        assert len(metric_paths)==1,(item['id'],metric_paths)
        m=read(metric_paths[0]);assert m['ply_roundtrip']['render_max_abs']<=1e-5
        assert max(m['ply_roundtrip']['all_parameter_max_abs'].values())<=1e-6
        assert len([x for x in m['samples'] if x['split']=='validation_generated'])==117
        assert len(list(demo.glob('details_view*.jpg')))==9
        rows.append({'label':item['id'],'identity':identity,'frames':frames,'technical':True,
            'checkpoint_input_time_and_video_hashes':'match original full-decode records','detail_previews':9,'heldout_generated_observations':117,
            'ply_render_max_error':m['ply_roundtrip']['render_max_abs'],'metrics':str(metric_paths[0]),'production_approved':False})
    assert len(rows)==gallery['technical_assets']
    controls=[]
    for row in read(R/'quality_results.json')['rows']:
        s=read(Path(row['metrics']).parents[1]/'sampling_step4999.json');counts=s['counts_by_camera_and_frame']
        assert s['optimized_steps']==5000 and sum(map(sum,counts))==5000
        assert all(sum(counts[i])==0 for i in (0,8,16))
        assert len(s['train_camera_ids'])==row['training_views']
        controls.append({'label':row['label'],'train_views':row['training_views'],'steps':5000,'holdout_updates':0})
    queues={}
    for name in ['mobility_queue','short_queue','temporal_queue','single_time_queue','time_budget_queue','weak_velocity_queue','weak_velocity_duration_queue','m02_short24_queue','m02_short_assets_queue']:
        d=read(R/(name+'.json'));queues[name]={'state':d['state'],'jobs':{k:{'state':v.get('state'),'stage':v.get('stage')} for k,v in d.get('jobs',{}).items()}}
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'state':'current_delivery_snapshot_verified_quality_limits_retained',
        'technical_assets':len(rows),'distinct_tracks':len(identities),'production_approved':0,'records':rows,'strict_view_controls':controls,'current_queues':queues,
        'completed_scope':['12 baseline tracks and retained failures','Matched generated-vs-GS demos, backgrounds and details','Common-six initialization with6/12/21view controls','Budget/resolution/capacity and independent controls','Root-local walking and original/new root-trajectory replay','61-frame24-view short-clip dynamic export','Generated-observation metrics with honest non-GT scope'],
        'remaining_quality_and_integration':['Queue records are historical stage receipts; check actual host processes before treating a queue as live','Visual quality not approved: clothing drift, blur, ghosting and foot errors remain','Moving-camera source action fidelity/world motion not verified','Waymo metric scale/ground/world placement and arbitrary SMPL-X retargeting not implemented'],
        'sampling_scope':'117 =3 heldout generated views x13 times x3 backgrounds; does not imply real novel-view GT',
        'pause_record_historical':read(R/'pause_request.json'),'current_pause_status':read(R/'status.json')['pause_status'],
        'alternative_backend':'14 completed 4DGaussians variants on6 existing tracks; audited separately in phase2_completion_audit.json',
        'not_claimed':['100 completed assets','paper-author unreleased backend reproduced','technical completion implies visual fidelity']}
    target=R/'delivery_requirement_audit.json'
    archive=R/'report_snapshots/delivery_requirement_audit_before_current.json'
    if target.exists() and not archive.exists():archive.write_text(target.read_text())
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'technical_assets':len(rows),'verified_controls':len(controls),'state':result['state']}))

if __name__=='__main__':main()
