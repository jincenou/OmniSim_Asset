"""Check common denominators and aggregate subjects without treating frames as identities."""
import collections
import csv
import json
from pathlib import Path
import numpy as np
from roi_candidates import ROOT,PARTS

MAIN=['N01','N02','N03','N04','N06','N07','N08']
METHODS=['ftgs_base5k','ftgs_opt15k','4dgs_grid25','4dgs_grid61']
KEYS=['coverage','soft_missing','recall_alpha05','local_window_iou','mse','psnr','boundary_distance_px',
      'outside_reference_alpha','temporal_mse','temporal_valid_fraction']


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    temp=path.with_suffix('.writing')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


def mean(rows,key):
    values=[float(r[key]) for r in rows if r[key] not in ['',None]]
    return float(np.mean(values)) if values else None


def main():
    baseline=json.loads((ROOT/'audit/part_queue.json').read_text());assert baseline['state']=='complete'
    new=json.loads((ROOT/'audit/grid_part_queue.json').read_text());assert new['state']=='complete'
    allrows=[];summaries=[];checks=[]
    for label in MAIN+['N11']:
        methods=METHODS+(['grid4d_33k'] if label in ['N04','N11'] else [])
        denominators=None
        for method in methods:
            root=ROOT/'audit/metrics'/label;rows=read(root/f'{method}.csv')
            index={(r['view'],r['frame'],r['part']):tuple(r[k] for k in ['reference_core_pixels','reference_foreground_pixels','unknown_foreground_pixels','boundary_reference_pixels','outside_window_pixels','temporal_valid_pixels']) for r in rows}
            if denominators is None:denominators=index
            else:assert index==denominators,(label,method,'reference denominators differ')
            allrows.extend(rows)
            info=json.loads((root/f'{method}.json').read_text())
            for part in PARTS:
                subset=[r for r in rows if r['part']==part];known=[r for r in subset if r['status']=='measured']
                summary=dict(scope='per_subject',label=label,cohort='reserve' if label=='N11' else 'main',method=method,part=part,
                    identities_expected=1,identities_evaluable=int(bool(known)),observations_total=len(subset),observations_evaluable=len(known),
                    reference_core_pixels=sum(int(r['reference_core_pixels']) for r in subset),
                    temporal_valid_pixels=sum(int(r['temporal_valid_pixels']) for r in subset),
                    coverage_p05=float(np.quantile([float(r['coverage']) for r in known],.05)) if known else None,
                    longest_missing_seconds=max(v['longest_missing_seconds'] for v in info['longest_missing'] if v['part']==part),
                    **{k:mean(known,k) for k in KEYS})
                summaries.append(summary)
            checks.append(dict(label=label,method=method,observations=len(rows),reference_denominators_identical=True))
    per_subject=summaries.copy()
    for method in METHODS:
        for part in PARTS:
            cases=[r for r in per_subject if r['label'] in MAIN and r['method']==method and r['part']==part]
            known=[r for r in cases if r['identities_evaluable']]
            row=dict(scope='main7_equal_subject_mean',label=','.join(MAIN),cohort='main',method=method,part=part,
                identities_expected=7,identities_evaluable=len(known),observations_total=sum(r['observations_total'] for r in cases),
                observations_evaluable=sum(r['observations_evaluable'] for r in cases),reference_core_pixels=sum(r['reference_core_pixels'] for r in cases),
                temporal_valid_pixels=sum(r['temporal_valid_pixels'] for r in cases),coverage_p05=mean(known,'coverage_p05'),
                longest_missing_seconds=mean(known,'longest_missing_seconds'),**{k:mean(known,k) for k in KEYS})
            summaries.append(row)
    paired=[]
    for row in summaries:
        base=next(r for r in summaries if r['scope']==row['scope'] and r['label']==row['label'] and r['part']==row['part'] and r['method']=='4dgs_grid61')
        item=dict(row)
        item.update({k+'_delta_vs_hust61':row[k]-base[k] if row[k] is not None and base[k] is not None else None for k in KEYS})
        paired.append(item)
    write(ROOT/'audit/part_metrics.csv',allrows)
    write(ROOT/'audit/paired_summary.csv',paired)
    (ROOT/'audit/part_metric_audit.json').write_text(json.dumps(dict(state='passed',all_rows=len(allrows),subject_method_pairs=len(checks),checks=checks,
        denominator_rule='Same reference core,exterior band,boundary and flow pixels for every method per identity/view/time/part',
        aggregation='First valid observations within each person; then equal-person mean for7main identities. N11 separate. Unknown cases explicitly counted; Grid4D has no fabricated7-person mean.',
        limitations='Anatomical cores do not measure full semantic parts,head hair,accessories or occluded areas. Low-confidence reference regions remain unknown.'),indent=2)+'\n')
    for label in ['N04','N11']:
        for part in ['left_arm','right_arm','left_leg','right_leg']:
            print(label,part,[(r['method'],r['coverage'],r['psnr'],r['temporal_mse']) for r in summaries if r['label']==label and r['part']==part and r['method'] in ['4dgs_grid61','grid4d_33k']],flush=True)


if __name__=='__main__':main()
