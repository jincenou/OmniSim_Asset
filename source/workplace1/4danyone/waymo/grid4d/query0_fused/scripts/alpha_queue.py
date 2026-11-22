from runtime import *
async def main():
    repair='--repair' in sys.argv;path=NEW/('EVALUATION_REPAIR_STATUS.json' if repair else 'EVALUATION_STATUS.json');assert not path.exists()
    plan=json.loads((NEW/'PLAN.json').read_text());state=dict(state='running_alpha_missing_evaluation',pid=os.getpid(),jobs={})
    async def one(label,gpu):
        job=state['jobs'][label]=dict(state='waiting_gpu',gpu=gpu,uuid=UUIDS[gpu]);save(path,state);lock=await acquire(gpu)
        try:
            job['state']='running';save(path,state);key=label+'_alpha_48000';run=plan['anchors'][label]['alpha']['run']
            suffix='_envrepair' if repair else ''
            await command(state,path,job,gpu,key+'_dense'+suffix,'evaluate_quality.py',['--run',run,'--iteration','48000','--dense'])
            await command(state,path,job,gpu,key+'_parts'+suffix,'evaluate_parts.py',['--label',label,'--arm',key])
            job['state']='complete';save(path,state)
        except Exception as e:job.update(state='failed',error=repr(e));save(path,state)
        finally:lock.close()
    await asyncio.gather(*(one(label,gpu) for label,gpu in zip(['N02','N03','N06','N08'],[6,7,2,3])))
    state['state']='alpha_missing_evaluation_complete' if all(j['state']=='complete' for j in state['jobs'].values()) else 'terminal_with_failures';save(path,state)
if __name__=='__main__':asyncio.run(main())
