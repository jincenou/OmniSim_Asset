"""Aggregate complete cases only; retain the fixed seven-person cohort."""
import csv,hashlib,json,statistics
from pathlib import Path
from run_queue import ROOT,save,stamp
BASE=ROOT.parent/'random8/r20261015_113332'
MAIN=['N01','N02','N03','N04','N06','N07','N08']
METHODS=['hust_grid61','cem_stage1','cem_stage2']
KEYS=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']

def main():
    results={};resources={}
    for label in MAIN+['N11']:
        p=ROOT/'results'/label/'results.json'
        if not p.exists():continue
        d=json.loads(p.read_text())
        assert all(len(d['samples'][m])==39 for m in METHODS)
        assert all(v<1e-4 for v in d['baseline_metric_max_absolute_error'].values())
        for name,info in d['videos'].items():
            video=p.parent/name
            assert hashlib.sha256(video.read_bytes()).hexdigest()==info['sha256']
        assert len(d['videos'])==3
        results[label]=d
        stages=[json.loads((ROOT/'runs'/label/f'stage{s}/complete.json').read_text()) for s in [1,2]]
        events=json.loads((ROOT/'runs'/label/'stage2/correction_events.json').read_text())
        hust=json.loads((BASE/'four_dg/runs'/f'{label}_grid61/complete.json').read_text())
        resources[label]={'hust_seconds':hust['elapsed_seconds'],'hust_updates':hust['actual_optimizer_updates'],
            'cem_stage1_seconds':stages[0]['elapsed_seconds'],'cem_stage2_seconds':stages[1]['elapsed_seconds'],
            'cem_total_updates':sum(s['updates_this_stage'] for s in stages),
            'cem_peak_allocated_gib':max(s['peak_allocated_gib'] for s in stages),
            'cem_final_points':stages[1]['points'],'cem_motion_groups':stages[1]['motion_groups'],
            'cec_calls':len(events),'cec_nonempty_calls':sum(not e['no_candidates'] for e in events),
            'cec_points_added_at_calls':sum(e['points_after']-e['points_before'] for e in events)}
    assert set(MAIN)<=results.keys(),'Do not publish an incomplete main-cohort mean'
    cohorts={'main7':MAIN,'walking2':['N04','N08']}
    if 'N11' in results:cohorts['separate_stress_N11']=['N11']
    average={scope:{m:{key:statistics.mean(results[l]['mean'][m][key] for l in labels) for key in KEYS} for m in METHODS} for scope,labels in cohorts.items()}
    temporal={scope:{m:{key:statistics.mean(results[l]['temporal_mean'][m][key] for l in labels) for key in results[labels[0]]['temporal_mean'][m]} for m in METHODS} for scope,labels in cohorts.items()}
    rows=[]
    for label,d in results.items():
        rows.extend({'scope':'person','label':label,'method':m,**d['mean'][m]} for m in METHODS)
    for scope,means in average.items():
        rows.extend({'scope':scope,'label':','.join(cohorts[scope]),'method':m,**means[m]} for m in METHODS)
    path=ROOT/'results/metrics.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','label','method',*KEYS]);writer.writeheader();writer.writerows(rows)
    save(ROOT/'results/summary.json',dict(updated_at=stamp(),completed_labels=list(results),
        main_cohort=MAIN,separate_stress_case='N11',average=average,temporal_average=temporal,
        per_person={l:d['mean'] for l,d in results.items()},resources=resources,
        comparison='Same generated images, cameras, initialization, heldout protocol; unequal recipe/compute, not a single-variable ablation.',
        reference='Generated heldout views; not real Waymo novel-view truth.',
        interpretation='CEM complete is lower than HUST on foreground PSNR and worse on crop LPIPS for all seven main cases. Keep HUST as the current baseline.',
        production_approved=False))
    print(json.dumps({'completed':list(results),'mean':average['main7'],'resources':resources},indent=2))

if __name__=='__main__':main()
