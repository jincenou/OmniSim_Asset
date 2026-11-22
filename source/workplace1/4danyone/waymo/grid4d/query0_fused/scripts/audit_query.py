"""Check saved experiment contracts and exact exports without GPU optimization."""
import importlib.util,json,random
from pathlib import Path
import numpy as np,torch
from analyze_query import NEW,ROOT,PLAN,read,descriptor
from runtime import save
from selector import Selector
spec=importlib.util.spec_from_file_location('old_asset_audit',ROOT/'optimization_quality_v3_20261018/scripts/audit_assets.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.sha

def main():
    torch.set_num_threads(2)
    assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    implementation=read(NEW/'IMPLEMENTATION.json')
    for p,h in implementation['files'].items():assert sha(NEW/p)==h,p
    for k in ['cross8','quality_v3']:assert sha(PLAN['prior_audit'][k])==PLAN['prior_audit'][k+'_sha256']
    original=read(NEW/'ANCHOR_AUDIT.json')
    for label,h in original['manifest_hashes'].items():
        assert sha(ROOT.parent/'random8/r20261015_113332/four_dg/data'/label/'fdanyone.json')==h
    lrs=read(NEW/'data/learning_rates.json');exports=[];protocol=[];assets=[];replays=[]
    for label in PLAN['query_labels']:
        anchor=PLAN['anchors'][label]['fused'];parent=Path(anchor['run']);assert sha(parent/'recovery.pt')==anchor['recovery_sha256']
        base=torch.load(parent/'recovery.pt',map_location='cpu',weights_only=False);cand=read(NEW/'data'/label/'candidates.json');uids=[r['uid'] for r in cand['observations']]
        table=read(NEW/'demand_tables'/f'{label}_fused_initial.json');assert len(table['rows'])==168 and set(table['views']).isdisjoint({0,8,16})
        assert [(r['view'],r['frame']) for r in table['rows']]==[(r['view'],r['frame']) for r in cand['scoring']]
        prng=random.Random();prng.setstate(base['rng']['python']);stack=list(base['stack']);sequence=[];states={}
        for step in range(1,2001):
            if not stack:stack=list(uids)
            sequence.append(stack.pop(prng.randrange(len(stack))))
            if step in [500,1000,2000]:states[step]=(list(stack),prng.getstate())
        paired_rng=None
        for arm in ['U','P','E','Q']:
            name=f'{label}_fused_{arm}';run=NEW/'runs'/name;done=read(run/'complete.json');trace=[json.loads(l) for l in (NEW/'sampling_traces'/f'{name}.jsonl').read_text().splitlines()]
            assert len(trace)==2000 and done['extra_updates']==2000 and done['global_actual_updates']==34999
            assert read(run/'base_sequence.json')==sequence and read(run/'learning_rates.json')==lrs
            assert sha(NEW/'sampling_traces'/f'{name}.jsonl')==done['sampling_sha256']
            regularizer=[r['perturbation_rng_before_sha256'] for r in trace]
            if paired_rng is None:paired_rng=regularizer
            else:assert paired_rng==regularizer
            selector=Selector(cand,sequence,arm);selector.update(table);snap_states={};empty=0
            for step,r in enumerate(trace,1):
                chosen=selector.choose(step)
                assert all(r[k]==v for k,v in chosen.items()),(name,step,'selector replay')
                assert r['learning_rates']==lrs[step-1] and r['step']==step and r['global_actual_updates']==32999+step
                assert r['points']==anchor['points'] and r['view'] not in [0,8,16]
                assert r['demand_version']==(0 if step<=1000 else 1000)
                if step%2==0:assert r['uid']==sequence[step-1]
                if r['selected_part'] is not None:assert cand['observations'][r['candidate_index']]['areas'][r['selected_part']]>=64
                empty+=r['selection_reason'].startswith('empty_')
                if step==1000:
                    refreshed=read(NEW/'demand_tables'/f'{name}_1000.json');assert [(r['view'],r['frame']) for r in refreshed['rows']]==[(r['view'],r['frame']) for r in table['rows']]
                    assert refreshed['render_calls']==336 and refreshed['optimizer_updates']==0;selector.update(refreshed)
                if step in [500,1000,2000]:snap_states[step]=selector.state()
            snapshots=[]
            for step in [500,1000,2000]:
                actual=32999+step;path=run/'recoveries'/f'{actual}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
                assert state['global_actual_updates']==actual and state['stage_update']==step
                assert state['active_sh_degree']==base['active_sh_degree'];assert state['gaussians']['_xyz'].shape==base['gaussians']['_xyz'].shape
                assert state['plan_sha256']==sha(NEW/'PLAN.json');assert state['query_selector']==snap_states[step]
                assert state['stack']==states[step][0] and state['base_sampler_state']==states[step][1]
                assert sha(state['demand_table'])==state['demand_table_sha256']
                for key in ['gaussian_optimizer','deform_optimizer']:
                    a,b=base[key],state[key];assert len(a['param_groups'])==len(b['param_groups'])
                    for g,h in zip(a['param_groups'],b['param_groups']):
                        assert g['name']==h['name'] and len(g['params'])==len(h['params'])
                        for i,j in zip(g['params'],h['params']):assert float(b['state'][j]['step'])-float(a['state'][i]['step'])==step
                exports.append(old.verify_recovery(run,actual,state));meta=read(run/'snapshots'/f'{actual}.json')
                for f,h in meta['files'].items():assert sha(run/f)==h,(name,f)
                val=read(run/'validation'/f'{actual}.json');assert val['state']=='passed' and val['source_data_reads']==0 and val['independent_replay_max_abs_error']<1e-5 and val['background_composition_max_abs_error']<1e-5
                assert any(x['frame']==12.25 for x in val['time_checks']);replays.append(dict(run=name,actual=actual,**val))
                # Restore the private sampler and demand, then reproduce the next native choice.
                if step<2000:
                    resumed=Selector(cand,sequence,arm);resumed.update(read(state['demand_table']));resumed.restore(state['query_selector']);next_item=resumed.choose(step+1)
                    assert all(trace[step][k]==v for k,v in next_item.items())
                    rng=random.Random();rng.setstate(state['base_sampler_state']);st=list(state['stack'])
                    if not st:st=list(uids)
                    assert st.pop(rng.randrange(len(st)))==sequence[step]
                snapshots.append(dict(actual_updates=actual,metadata=str(run/'snapshots'/f'{actual}.json'),recovery=str(path),files=meta['files'],independent_replay=str(run/'validation'/f'{actual}.json')))
                del state
            assets.append(dict(label=label,arm=arm,run=str(run),terminal_actual_updates=34999,parent=str(parent),config=str(run/'config.json'),cameras=str(run/'asset_camera_data.json'),base_sequence=str(run/'base_sequence.json'),sampling_trace=str(NEW/'sampling_traces'/f'{name}.jsonl'),snapshots=snapshots))
            protocol.append(dict(run=name,actual_updates=2000,full_parent_sha256=anchor['recovery_sha256'],selector_replay_exact=True,shared_even_samples=1000,paired_regularizer_rng_steps=2000,LR_steps_exact=2000,points_fixed=anchor['points'],all_optimizer_groups_increment_exact=True,empty_candidate_uniform_count=empty,selector_and_base_sampler_resume_next_choice_passed=[500,1000]))
            print('audited',name,flush=True)
        del base
    alpha=[]
    for label in PLAN['labels']:
        d=descriptor(label,'alpha48');run=d['run'];meta=read(run/'snapshots'/f'{d["iteration"]}.json');assert sha(run/'snapshots'/f'{d["iteration"]}.json')==PLAN['anchors'][label]['alpha']['snapshot_sha256']
        validation=NEW/'results'/f'{label}_alpha_48000'/'replay.json' if label in ['N04','N11'] else run/'validation'/f'{d["iteration"]}.json'
        val=read(validation);assert val['state']=='passed' and val['source_data_reads']==0
        alpha.append(dict(label=label,run=str(run),snapshot=str(run/'snapshots'/f'{d["iteration"]}.json'),independent_replay=str(validation),parts=str(d['parts']),raw=str(d['raw']),actual_updates=47999))
    assert len(exports)==len(replays)==36 and len(assets)==12
    save(NEW/'research/recovery_audit.json',dict(state='passed',exports=exports,independent_replays=replays,alpha_reused_validation=alpha))
    save(NEW/'research/protocol_audit.json',dict(state='passed',plans_sources_and_old_audits_unchanged=True,branches=protocol,candidate_inputs='Reuse preparation actual-read hashes and prior cross8 full audit; no redundant full image scan',limits='Selector/base RNG next-choice resumed exactly; no claim of bitwise whole-CUDA optimization replay. Generic old trainer lacks query selector restoration and must not silently be used for this experiment.'))
    save(NEW/'ASSETS.json',dict(state='validated',new_assets=assets,alpha_stage_assets=alpha,environment='/mnt/ojc/miniconda3/envs/grid4d',runtime=str(ROOT/'source'),contract='PLY + hash/deform + config + camera/time metadata + matching Grid4D runtime; rendering needs no training observations. Full recoveries include optimizer/RNG/stage/selector/base sampler/demand. Not arbitrary new actions or metric world placement.'))
if __name__=='__main__':main()
