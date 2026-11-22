from runtime import *
async def main():
    repair='--repair' in sys.argv;path=NEW/'research'/('DEMAND_REPAIR_STATUS.json' if repair else 'DEMAND_STATUS.json');assert not path.exists();state=dict(state='running',pid=os.getpid(),jobs={});save(path,state)
    async def one(label,gpu):
        job=state['jobs'][label]=dict(state='waiting_gpu',gpu=gpu,uuid=UUIDS[gpu]);save(path,state);lock=await acquire(gpu)
        try:
            job['state']='running';save(path,state);await command(state,path,job,gpu,'initial_demand_'+label+('_projectionrepair' if repair else ''),'demand.py',['--label',label]);job['state']='complete'
        except Exception as e:job.update(state='failed',error=repr(e))
        finally:lock.close();save(path,state)
    await asyncio.gather(*(one(l,g) for l,g in zip(['N04','N06','N11'],[0,1,4])))
    state['state']='complete' if all(j['state']=='complete' for j in state['jobs'].values()) else 'terminal_with_failures';save(path,state)
if __name__=='__main__':asyncio.run(main())
