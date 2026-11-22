"""Create a NEW immutable clip/run manifest with independently selected observations and output clock."""
import argparse,copy,json,math
from pathlib import Path
from data import ROOT,output_path,save_json,sha,observation_seconds,model_time

def configure(source,output,name,views,frames,init_frames,accumulation,start=None,end=None,render_fps=25):
 source=Path(source);output=output_path(output)
 if output.exists():raise FileExistsError(output)
 m=copy.deepcopy(json.loads(source.read_text()));available=m['available_frame_ids'];all_times={f:observation_seconds(m,f) for f in available}
 start=min(all_times.values()) if start is None else start;end=max(all_times.values()) if end is None else end
 if not(min(all_times.values())<=start<end<=max(all_times.values())):raise ValueError('Window outside source real time range')
 if len(set(views))!=len(views) or not views or not set(views)<=set(m['available_camera_ids']):raise ValueError('Invalid training cameras')
 if set(views)&set(m['validation_camera_ids']):raise ValueError('Validation camera leakage')
 if not frames or frames!=sorted(set(frames)) or not set(frames)<=set(available):raise ValueError('Invalid training frame IDs')
 if any(not(start<=all_times[f]<=end) for f in frames):raise ValueError('Training observations outside real window')
 if not init_frames or not set(init_frames)<=set(frames):raise ValueError('Initialization must use training times only')
 if accumulation<1:raise ValueError('Accumulation must be positive')
 if render_fps<=0:raise ValueError('Output FPS must be positive')
 intervals=(end-start)*render_fps
 if abs(intervals-round(intervals))>1e-7:raise ValueError('Window must contain an integer number of output intervals; no padding/repetition')
 shortened=(start!=min(all_times.values()) or end!=max(all_times.values()))
 if shortened and output.parent==source.parent:raise ValueError('Changing real time window requires a new clip directory')
 m['parent_manifest']={'path':str(source.resolve()),'sha256':sha(source)};m['clip_window_seconds']=[start,end];m['new_clip_due_to_window_change']=shortened
 m['available_frame_ids']=[f for f in available if start<=all_times[f]<=end];m['evaluation_frame_ids']=m['available_frame_ids'][:]
 m['source_material_total_frame_count']=len(available);m['source_material_original_observations']=m['source_real_frame_count']
 window_records=[r for r in m['frame_mapping'] if r['slot'] in m['available_frame_ids']]
 m['source_real_frame_count']=len({r[k] for r in window_records for k in ['left_source_frame','right_source_frame']})
 m['direct_source_slots']=sum(not r['interpolated'] for r in window_records);m['source_interpolated_slots']=len(window_records)-m['direct_source_slots']
 m['sample_span_seconds']=end-start;m['real_seconds_to_model_time']={'scale':1/(end-start),'offset':-start/(end-start)};m['model_time_to_real_seconds']={'scale':end-start,'offset':start}
 m['render_fps']=render_fps;m['render_timestamps_sec']=[start+i/render_fps for i in range(round(intervals)+1)]
 m['protocols']={name:{'train_camera_ids':views,'train_frame_ids':frames,'init_frame_ids':init_frames,'views_per_step':1,'times_per_step':1,'microbatch':1,'gradient_accumulation':accumulation}}
 m['scope']='New configuration only; not trained. Real source clock retained. Sampling subset never relabels observations as consecutive25fps.'
 save_json(output,m);return m

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output-manifest',type=Path,required=True);p.add_argument('--protocol-name',required=True);p.add_argument('--train-views',type=int,nargs='+',required=True);p.add_argument('--train-frames',type=int,nargs='+',required=True);p.add_argument('--init-frames',type=int,nargs='+',required=True);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--window-start-sec',type=float);p.add_argument('--window-end-sec',type=float);p.add_argument('--render-fps',type=float,default=25);a=p.parse_args();configure(a.source_manifest,a.output_manifest,a.protocol_name,a.train_views,a.train_frames,a.init_frames,a.accumulation,a.window_start_sec,a.window_end_sec,a.render_fps);print(a.output_manifest)
if __name__=='__main__':main()
