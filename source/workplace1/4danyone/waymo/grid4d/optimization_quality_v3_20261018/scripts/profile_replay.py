"""Timed asset-only full-sequence replay with no video duplication or source reads."""
import argparse,os
from quality_common import *
@torch.no_grad()
def main():
    begin=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--iteration',type=int,required=True);args=ap.parse_args();device=verify();run=Path(args.run)
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):assert not Path(os.fsdecode(values[0])).absolute().is_relative_to(BASE)
    sys.addaudithook(guard);load_start=time.monotonic();model,deform,pipe,config=load_asset(run,args.iteration);data=json.loads((run/'asset_camera_data.json').read_text());torch.cuda.synchronize();load_seconds=time.monotonic()-load_start;durations=[];pixels=0;checksum=0.
    for frame in range(data['frame_count']):
        start=time.monotonic();rgb,alpha=raw_render(model,deform,pipe,camera(data,0,frame));image=(rgb+.5*(1-alpha)).clamp(0,1);assert bool(torch.isfinite(image).all());torch.cuda.synchronize();durations.append(time.monotonic()-start);checksum+=float(image.mean());pixels+=data['width']*data['height']
    save(run/'validation'/f'{args.iteration}_timing.json',dict(state='passed',source_data_reads=0,device=device,frames=data['frame_count'],view=0,render_calls=2*data['frame_count'],pixels=pixels,load_wall_seconds=load_seconds,replay_wall_seconds=sum(durations),process_entry_seconds=time.monotonic()-begin,frame_p50_seconds=float(np.median(durations)),frame_p95_seconds=float(np.quantile(durations,.95)),checksum=checksum,scope='New timed verification replay, original video export timing unmeasured; no video encoding or writes, shared GPU wall clock, not pure GPU time'))
if __name__=='__main__':main()
