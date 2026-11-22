"""Bounded paired initialization diagnostic, retained in tmux on GPUs4/6."""
import asyncio
import argparse
import fcntl
import os
from pathlib import Path
import shutil
from after_batch import ROUND,W,read,write,now
from resource_lease import lease,wait_memory

PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
FTGS='/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python'

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sigma8',action='store_true');args=parser.parse_args()
    path=ROUND/('weak_velocity_duration_queue.json' if args.sigma8 else 'weak_velocity_queue.json')
    suffix='_sigma8' if args.sigma8 else ''
    with path.with_suffix('.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():raise FileExistsError(path)
        state={'state':'preparing','pid':os.getpid(),'started_at':now(),'jobs':{}}
        def save():state['updated_at']=now();write(path,state)
        save()
        def environment(gpu):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',TORCH_EXTENSIONS_DIR=str(W/'reconstruction/deps/torch_extensions'),
                TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH'];env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            return env
        async def stage(name,python,script,args,gpu,handles):
            if shutil.disk_usage(ROUND).free<35*1024**3:raise RuntimeError('Disk reserve below35GiB')
            log=ROUND/'logs'/f'weak_velocity_{name}.log'
            command=[python,str(ROUND/'scripts'/script),*map(str,args)]
            with log.open('xb') as stream:
                child=await asyncio.create_subprocess_exec(*command,cwd=ROUND/'backend',env=environment(gpu),stdout=stream,stderr=asyncio.subprocess.STDOUT,pass_fds=(*handles,own.fileno()))
                state['jobs'][name]={'state':'running','gpu':gpu,'pid':child.pid,'command':command,'log':str(log),'started_at':now()};save()
                code=await child.wait()
            state['jobs'][name].update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=now());save()
            log.with_suffix('.exit').write_text(str(code)+'\n')
            if code:raise RuntimeError(f'{name} exited{code}: {log}')
        try:
            if args.sigma8:
                if read(ROUND/'diagnostics/weak_velocity/paired_audit.json')['state']!='only_initial_velocities_changed':raise ValueError('Original pair not audited')
                if not (ROUND/'diagnostics/weak_velocity/duration_init.json').is_file():raise FileNotFoundError('Duration initialization missing')
            else:
                async with lease([4]) as handles:
                    await wait_memory([4],6000)
                    await stage('prepare',PY,'weak_velocity_init.py',[],4,handles)
            state['state']='paired_training_and_evaluation';save()
            async def case(job):
                variant,gpu=job
                async with lease([gpu]) as handles:
                    await wait_memory([gpu],6000)
                    await stage(variant+suffix,FTGS,'weak_velocity_train.py',['--variant',variant]+(['--sigma8'] if args.sigma8 else []),gpu,handles)
                    out=ROUND/'diagnostics/weak_velocity'/(variant+suffix)
                    await stage(variant+suffix+'_metrics',FTGS,'evaluate_asset61.py',['--checkpoint',out/'ckpts/ckpt_14999.pt',
                        '--data',ROUND/'reconstruction/data/r07_short61_24','--output',out/'quality_metrics'],gpu,handles)
            results=await asyncio.gather(*map(case,[('zero',6),('body',7)] if args.sigma8 else [('zero',4),('body',6)]),return_exceptions=True)
            errors=[str(x) for x in results if isinstance(x,BaseException)]
            if errors:raise RuntimeError('; '.join(errors))
            root=ROUND/'diagnostics/weak_velocity'
            a=read(root/('zero'+suffix)/'initial_parameters.json')['parameter_sha256'];b=read(root/('body'+suffix)/'initial_parameters.json')['parameter_sha256']
            changed=[k for k,v in a.items() if b[k]!=v]
            if changed!=['velocities']:raise ValueError(f'Unexpected initial parameter changes:{changed}')
            a=read(root/('zero'+suffix)/'experiment.json')['config'];b=read(root/('body'+suffix)/'experiment.json')['config']
            changes={k:[v,b[k]] for k,v in a.items() if b[k]!=v}
            if set(changes)!={'result_dir','init_npz_path'}:raise ValueError(changes)
            duration_changes={}
            if args.sigma8:
                for variant in ('zero','body'):
                    old=read(root/variant/'initial_parameters.json')['parameter_sha256'];new=read(root/(variant+suffix)/'initial_parameters.json')['parameter_sha256']
                    diff=[k for k,v in old.items() if new[k]!=v]
                    if diff!=['durations']:raise ValueError(f'{variant}:unexpected sigma8 changes:{diff}')
                    duration_changes[variant]=diff
            write(root/('paired_audit'+suffix+'.json'),{'state':'only_initial_velocities_changed','initial_parameters_changed':changed,'config_differences':changes,'sigma_only_parameter_changes':duration_changes,
                'scope':'Same seed and sampling policy; optimizer relocation can diverge after different velocities. All results are generated-observation fitting.'})
            state['state']='pair_trained_evaluated_review_pending';save()
        except Exception as error:
            state.update(state='failed',error=f'{type(error).__name__}: {error}');save();raise

if __name__=='__main__':asyncio.run(main())
