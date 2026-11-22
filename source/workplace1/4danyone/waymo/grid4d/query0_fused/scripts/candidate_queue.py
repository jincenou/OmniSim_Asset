from runtime import *
async def main():
    path=NEW/'research/CANDIDATE_STATUS.json';assert not path.exists();state=dict(state='running',pid=os.getpid(),jobs={});save(path,state)
    async def one(label):
        job=state['jobs'][label]=dict(state='running');save(path,state)
        try:
            await command(state,path,job,None,'candidates_'+label,'prepare_candidates.py',['--label',label]);job['state']='complete'
        except Exception as e:job.update(state='failed',error=repr(e))
        save(path,state)
    await asyncio.gather(*(one(l) for l in ['N04','N06','N11']))
    state['state']='complete' if all(j['state']=='complete' for j in state['jobs'].values()) else 'terminal_with_failures';save(path,state)
    if state['state']=='complete':save(NEW/'candidate_manifest.json',dict(state='complete',identities={l:str(NEW/'data'/l/'candidates.json') for l in state['jobs']},training_views_only=True,heldout_not_read=True))
if __name__=='__main__':asyncio.run(main())
