import asyncio,os,time
from runtime import NEW,save,acquire,command
from analyze_query import descriptor
async def main():
    status=dict(state='running',pid=os.getpid(),started_at=time.time(),jobs={});path=NEW/'research/ALPHA_REPLAY_STATUS.json';save(path,status)
    async def one(label,gpu):
        job=status['jobs'][label]=dict(gpu=gpu);lock=await acquire(gpu)
        try:
            d=descriptor(label,'alpha48');await command(status,path,job,gpu,label+'_alpha_replay','validate_quality.py',['--run',str(d['run']),'--iteration',str(d['iteration']),'--report',str(NEW/'results'/f'{label}_alpha_48000'/'replay.json')]);job['state']='complete'
        finally:lock.close();save(path,status)
    await asyncio.gather(one('N04',6),one('N11',7));status.update(state='complete',ended_at=time.time());save(path,status)
if __name__=='__main__':asyncio.run(main())
