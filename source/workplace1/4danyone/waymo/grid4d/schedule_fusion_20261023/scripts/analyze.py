"""Fixed numerical gates, development-only continuation decision and final tables."""
import argparse,csv,json,math,time
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
FIELDS=['foreground_psnr','ssim','crop_lpips','outside_alpha_mean']


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def gates(label,name,value,base):
    checks=[]
    def add(metric,a,b,rule,limit):
        delta=None if a is None or b is None else b-a if rule=='drop' else a-b if rule=='increase' else a/b if b else (1. if a==0 else float('inf'))
        checks.append({'metric':metric,'candidate':a,'reference':b,'rule':rule,'limit':limit,'value':delta if delta is None or math.isfinite(delta) else None,
            'state':'unknown' if delta is None else 'pass' if delta<=limit else 'fail'})
    a,b=value['mean'],base['mean']
    add('foreground_psnr',a['foreground_psnr'],b['foreground_psnr'],'drop',.2)
    add('ssim',a['ssim'],b['ssim'],'drop',.001)
    add('crop_lpips',a['crop_lpips'],b['crop_lpips'],'increase',.002)
    add('outside_alpha',a['outside_alpha_mean'],b['outside_alpha_mean'],'ratio',1.05)
    for part in base['parts']:
        add('coverage_'+part,value['parts'][part]['coverage'],base['parts'][part]['coverage'],'drop',.01)
        add('temporal_rmse_'+part,value['parts'][part]['compensated_rmse'],base['parts'][part]['compensated_rmse'],'ratio',1.05)
    failed=[x['metric'] for x in checks if x['state']=='fail'];unknown=[x['metric'] for x in checks if x['state']=='unknown']
    return {'label':label,'method':name,'checks':checks,'failed':failed,'unknown':unknown,
        'numerical_state':'fail' if failed else 'unknown' if unknown else 'pass','visual_state':'pending; numerical pass alone is not quality acceptance'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args()
    plan=read(STUDY/'PLAN.json');labels=plan['subjects'];metrics={};all_gates=[];rows=[]
    for label in labels:
        data=read(STUDY/'results'/label/'metrics.json')['methods']
        extra=STUDY/'results'/f'{label}_extension/metrics.json'
        if args.final and extra.exists():data=read(extra)['methods']
        metrics[label]=data
        for name,value in data.items():
            rows.append({'label':label,'method':name,**value['mean']})
            if name!='A':all_gates.append(gates(label,name,value,data['A']))
    methods=list(metrics[labels[0]])
    means=[{'method':name,'identities':len(labels),**{key:sum(metrics[label][name]['mean'][key] for label in labels)/len(labels) for key in metrics[labels[0]][name]['mean']}} for name in methods]
    if not args.final:
        passing=[name for name in ['F1','F2'] if all(g['numerical_state']=='pass' for g in all_gates if g['method']==name)]
        ranking=sorted([{'method':name,'failed_measured_gates':sum(len(g['failed']) for g in all_gates if g['method']==name),
             'mean_LPIPS':next(m['crop_lpips'] for m in means if m['method']==name)} for name in ['F1','F2']],key=lambda x:(x['failed_measured_gates'],x['mean_LPIPS']))
        decision={'state':'numeric_review_complete','at':time.time(),'passing_all_three_numerically':passing,
            'selected_for_extension':None if passing else ranking[0]['method'],'ranking':ranking,
            'selection_scope':'Pre-registered H-development selection, not blind test; one shared recipe across all3, no threshold tuning',
            'extension_updates':0 if passing else 5000}
        save(STUDY/'DECISION_33K.json',decision)
    save(STUDY/('GATES_FINAL.json' if args.final else 'GATES_33K.json'),{'checks':all_gates,'unknown_is_not_pass':True})
    save(STUDY/('MEANS_FINAL.json' if args.final else 'MEANS_33K.json'),{'rows':means,'reference':'generated H development references; identity macro means'})
    with (STUDY/('metrics_final.csv' if args.final else 'metrics_33k.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'means':means,'gates':[(g['label'],g['method'],g['numerical_state'],g['failed']) for g in all_gates]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
