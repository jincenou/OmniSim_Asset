from runtime import *
async def main():
    path=NEW/'STATUS.json';old=json.loads(path.read_text());assert old['state']=='preparing_candidates_and_alpha_evaluation'
    assert json.loads((NEW/'research/DEMAND_REPAIR_STATUS.json').read_text())['state']=='complete'
    assert (NEW/'IMPLEMENTATION.json').exists();state=dict(state='running_initial_twelve',pid=os.getpid(),started_at=time.time(),jobs={});queue=asyncio.Queue()
    jobs=[(label,arm) for arm in ['U','P','E','Q'] for label in ['N04','N06','N11']]
    for label,arm in jobs:key=f'{label}_fused_{arm}';state['jobs'][key]=dict(state='queued');queue.put_nowait((label,arm))
    for _ in UUIDS:queue.put_nowait(None)
    save(path,state)
    async def worker(gpu):
        while True:
            item=await queue.get()
            if item is None:queue.task_done();return
            label,arm=item;key=f'{label}_fused_{arm}';job=state['jobs'][key];job.update(state='waiting_gpu',gpu=gpu,uuid=UUIDS[gpu]);save(path,state);lock=await acquire(gpu)
            try:
                job['state']='training';save(path,state)
                await command(state,path,job,gpu,key+'_train','train_query.py',['--label',label,'--arm',arm])
                run=NEW/'runs'/key;job['state']='evaluating';save(path,state)
                for it in [33499,33999,34999]:
                    name=key+'_'+str(it)
                    await command(state,path,job,gpu,name+'_validate','validate_quality.py',['--run',str(run),'--iteration',str(it)])
                    await command(state,path,job,gpu,name+'_metrics','evaluate_quality.py',['--run',str(run),'--iteration',str(it)]+(['--dense'] if it==34999 else []))
                await command(state,path,job,gpu,key+'_parts','evaluate_parts.py',['--label',label,'--arm',key+'_34999'])
                job.update(state='complete',ended_at=time.time());save(path,state)
            except Exception as e:job.update(state='failed',error=repr(e));save(path,state)
            finally:lock.close();queue.task_done()
    await asyncio.gather(*(worker(g) for g in [6,7,0,1,2,3,4,5]))
    state['state']='initial_twelve_complete_gates_pending' if all(j['state']=='complete' for j in state['jobs'].values()) else 'terminal_with_failures';state['ended_at']=time.time();save(path,state)
if __name__=='__main__':
    with (NEW/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
