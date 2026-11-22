"""Read-only formal-state audit, added after freeze; no training/selection changes."""
import argparse,asyncio,hashlib,json,math,os
from pathlib import Path
import torch

N=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def opt_steps(state):return sorted(set(float(s['step']) for s in state['state'].values() if 'step' in s))
def save(p,value):
    temporary=p.with_suffix('.writing');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        async def wait_for_queue():
            while not (N/'pipeline_queue.complete.json').exists():
                status=read(N/'STATUS.json')
                assert status['failed']==0,'Training/evaluation failure requires repair'
                os.kill(status['controller_pid'],0)
                await asyncio.sleep(30)
            assert read(N/'pipeline_queue.complete.json')['all_passed']
        asyncio.run(wait_for_queue())
        args.all=True
    torch.set_num_threads(2)
    plan=read(N/'PLAN.json')
    assert sha(N/'PLAN.json')==(N/'PLAN.sha256').read_text().split()[0]
    for rel,h in plan['implementation_sha256'].items():assert sha(N/rel)==h,rel
    for path,h in plan['dependency_sha256'].items():assert sha(path)==h,path
    for label in plan['labels']:
        source=plan['source_checkpoints'][label];assert sha(source['path'])==source['sha256']
        item=plan['inputs'][label]
        assert sha(item['manifest'])==item['manifest_sha256']
        assert sha(item['initialization'])==item['initialization_sha256']
        for name,h in item['local_file_hashes'].items():assert sha(N/'data'/label/name)==h
    results=[];pending=[]
    for label in plan['labels']:
        for arm in plan['arms']:
            key=f'{label}_{arm}';run=N/'runs'/key;asset=N/'assets'/key
            if not (asset/'replay_validation.json').exists():pending.append(key);continue
            done=read(run/'training.complete.json');resume=read(run/'resume.json')
            assert done['step']==32999 and done['counts']['additional_updates']==30000
            assert resume['step']==2999 and resume['sha256']==plan['source_checkpoints'][label]['sha256']
            assert done['counts']['target_renders']==60000 and done['counts']['alpha_renders']==35998
            assert done['counts']['visibility_renders']==(120000 if arm=='P4' else 0)
            trace=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
            assert trace[0]['step']==3000 and trace[-1]['step']==32999
            required=['context.0.weight','head.weight']+(['image_projection.weight'] if arm!='P1' else [])+(['query.weight','key.weight','offset.weight'] if arm in ('P3','P4') else [])
            gradient={k:max(r['query_grad_abs_sum'].get(k,0) for r in trace) for k in required}
            assert all(math.isfinite(v) and v>0 for v in gradient.values())
            ck=run/'recovery_32999.pt';snap=read(run/'snapshot_32999.json');assert sha(ck)==snap['recovery_sha256']
            state=torch.load(ck,map_location='cpu');middle=torch.load(run/'recovery_18000.pt',map_location='cpu')
            assert state['step']==32999 and middle['step']==18000
            assert opt_steps(state['query_optimizer'])==[30000.]
            assert opt_steps(middle['query_optimizer'])==[15001.]
            assert state['gaussians']['_xyz'].shape==middle['gaussians']['_xyz'].shape
            for name in ('gaussian_optimizer','deform_optimizer'):
                assert state[name]['state'].keys()==middle[name]['state'].keys()
                for i,s in state[name]['state'].items():
                    assert float(s['step'])-float(middle[name]['state'][i]['step'])==14999,(key,name,i)
            del middle
            portable=torch.load(asset/'query.pt',map_location='cpu');table=portable['Z'];length=float(portable['state']['length'])
            assert table.dtype==torch.float32 and table.shape==(plan['inputs'][label]['frame_count'],len(state['gaussians']['_xyz']),6)
            assert torch.isfinite(table).all() and table[:,:,:3].abs().max()<=.03*length+1e-6 and table[:,:,3:].abs().max()<=.100001
            assert state['query'].keys()==portable['state'].keys()
            assert all(torch.equal(state['query'][k],portable['state'][k]) for k in state['query'])
            geometry_saturated=float((table[:,:,:3].abs()>.95*.03*length).any(-1).float().mean())
            color_saturated=float((table[:,:,3:].abs()>.095).any(-1).float().mean())
            replay=read(asset/'replay_validation.json');assert replay['passed'] and replay['original_Random8_read_guard_active']
            assert len(replay['rows'])==36
            assert {r['background'] for r in replay['rows']}=={0.,.5,1.}
            assert {r['frame'] for r in replay['rows']}=={0.,12.25,30.5,float(table.shape[0]-1)}
            error=max(r['max_abs'] for r in replay['rows']);assert error<=1e-5
            results.append(dict(key=key,actual_logical_updates=32999,new_updates=30000,training_head_and_reader_gradients=gradient,optimizer_steps_verified=True,no_late_topology_change=True,points=table.shape[1],query_parameters=read(run/'parameter_counts.json')['query'],replay_max_abs=error,geometry_any_component_saturated_fraction=geometry_saturated,RGB_any_component_saturated_fraction=color_saturated,saturation_is_diagnostic_not_quality_gate=True,visibility_prior_mean_last=trace[-1]['diagnostics']['soft_visibility_mean']))
            del state,portable,table
    if args.all:assert not pending,pending
    save(N/'research/formal_state_audit.json',dict(state='all_formal_states_verified' if not pending else 'partial_formal_states_verified',frozen_code_unchanged=True,source_inputs_unchanged=True,results=results,pending=pending,visual_quality_verified=False,default_changed=False))
    print('formal audit:',len(results),'verified;',len(pending),'pending',flush=True)


if __name__=='__main__':main()
