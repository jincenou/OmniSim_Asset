"""Assemble independent object demos; baseline is an explicitly mismatched-budget reference."""
import argparse,json,shutil,html,os,time
from pathlib import Path
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageOps
from data import ROOT,save_json,sha,output_path
from video_tools import Video,panel,uint8
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',choices=['R07','M02'],required=True);a=p.parse_args();oid={'R07':'OBJ_0018','M02':'OBJ_0019'}[a.object];clip=ROOT/'objects'/oid/'clips/C001';m=json.loads((clip/'manifest.json').read_text());out=clip/'07_demo';out.mkdir(exist_ok=True)
runs={v:clip/'05_reconstruction'/f'{v}_S1_12k_v1' for v in ['E0','E1']}
for d in runs.values():assert json.loads((d/'evaluation/complete.json').read_text())['state']=='dynamic_rendering_and_evaluation_complete'
# Paired controls must match; confidence is the only objective difference.
complete={v:json.loads((d/'complete.json').read_text()) for v,d in runs.items()};assert complete['E0']['initial_parameter_hashes']==complete['E1']['initial_parameter_hashes'];assert complete['E0']['sample_sequence_sha256']==complete['E1']['sample_sequence_sha256'];assert complete['E0']['observation_counts']==complete['E1']['observation_counts']
old=ROOT.parent/'waymo/round2';b0_root=old/('diagnostics/weak_velocity/zero_sigma8' if a.object=='R07' else 'diagnostics/m02_duration/sigma8');comparison=old/('diagnostics/weak_velocity/before_after' if a.object=='R07' else 'diagnostics/m02_duration/comparison');source_video=b0_root/'demo/four_panel.mp4';source_hashes={str(source_video):sha(source_video)};videos=[]

def decode(path):
 reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24',input_params=['-threads','1']);meta=next(reader);assert meta['fps']==25
 w,h=meta['size']
 for b in reader:yield np.frombuffer(b,np.uint8).reshape(h,w,3)

def target(view,frame):
 path=Path(m['data_root'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png';rgba=np.asarray(Image.open(path),np.float32)/255;return uint8(rgba[...,:3]*rgba[...,3:]+1-rgba[...,3:]),rgba[...,3]>.5

def prediction(v,view,frame):return np.array(Image.open(runs[v]/'evaluation/renders'/f'cam{view:02d}'/f'{frame:06d}.png'))

# Reuse only old rendered source/input display panes, preserving source-hold convention.
writer=Video(out/'01_pipeline.mp4',1408,680)
for f,old_frame in enumerate(decode(source_video)):
 assert f<61 and old_frame.shape==(664,1408,3)
 ref,_=target(0,f);new=prediction('E1',0,f)
 columns=[panel(old_frame[24:,0:352],'Waymo native crop / left source held'),panel(old_frame[24:,352:704],'Actual input / RIFE + mask'),panel(ref,'4DAnyone generated / v00'),panel(new,f'E1 native4D / t={f/25:.2f}s\n12 training views x31 times')]
 writer.add(np.concatenate(columns,1))
assert writer.frames==61;videos.append(writer.close())
# Same generated cameras/timestamps; B0 supplied from previously validated encoded reference.
for view in [0,8,16]:
 path=comparison/f'view{view:02d}.mp4';source_hashes[str(path)]=sha(path);name='02_backend_comparison.mp4' if view==8 else f'02_backend_comparison_view{view:02d}.mp4';writer=Video(out/name,1408,680)
 detail=Video(out/'05_details.mp4',1408,680) if view==8 else None
 for f,baseline_frame in enumerate(decode(path)):
  assert baseline_frame.shape==(664,1408,3) and f<61
  ref,mask=target(view,f);b0=baseline_frame[24:,1056:1408];arrays=[ref,b0,prediction('E0',view,f),prediction('E1',view,f)];labels=['Generated reference','B0 sigma8 /15k:21x61\nREFERENCE, unequal observations','E0:12x31 /48k rendered samples','E1:12x31 /48k rendered samples']
  writer.add(np.concatenate([panel(im,label+f' / t={f/25:.2f}s') for im,label in zip(arrays,labels)],1))
  if f in [0,10,20,30,40,50,60]:Image.fromarray(np.concatenate(arrays,1)).save(out/f'comparison_v{view:02d}_f{f:03d}.png')
  if detail:
   ys,xs=np.where(mask);x0=max(0,int(xs.min())-10);x1=min(352,int(xs.max())+11);y0,y1=int(ys.min()),int(ys.max())+1;h=y1-y0;canvas=Image.new('RGB',(1408,680),(28,28,28));draw=ImageDraw.Draw(canvas)
   boxes=[(x0,int(y0+.15*h),x1,int(y0+.75*h)),(x0,int(y0+.65*h),x1,min(640,y1+10))]
   for col,im in enumerate(arrays):
    for row,box in enumerate(boxes):
     crop=ImageOps.contain(Image.fromarray(im).crop(box),(352,300),Image.Resampling.NEAREST);canvas.paste(crop,(col*352+(352-crop.width)//2,row*340+40));draw.text((col*352+4,row*340+4),['Reference','B0 unequal budget','E0','E1'][col]+' / '+['upper-body + accessories','lower-body proxy'][row],fill='white')
   detail.add(np.array(canvas))
 videos.append(writer.close())
 if detail:videos.append(detail.close())
for name,source in [('03_motion_fixed_views.mp4',runs['E1']/'evaluation/motion_fixed_views.mp4'),('04_geometry_orbit.mp4',runs['E1']/'evaluation/geometry_orbit.mp4')]:
 dest=out/name;shutil.copyfile(source,dest);assert sha(source)==sha(dest);videos.append({'path':str(dest),'sha256':sha(dest),'copied_from_validated_video':str(source),'frames':61 if name.startswith('03') else 120,'fps':25})
# All61 output timestamps; unsupervised odd frames deliberately show no weight map.
conf=np.load(clip/'05_reconstruction/confidence_S1/cam01.npz');lookup={int(f):w.astype('float32') for f,w in zip(conf['frames'],conf['weights'])};writer=Video(out/'06_confidence.mp4',704,680)
for f in range(61):
 ref,mask=target(1,f)
 if f in lookup:
  w=lookup[f];heat=np.stack([1-w,.25*np.ones_like(w),w],-1);overlay=uint8(ref/255*.55+heat*.45);label='RGB confidence / training view01\nBlue high, red low; alpha unweighted'
 else:overlay=np.full_like(ref,128);label='Unsupervised time: no confidence map\nNo extra temporal observation used'
 writer.add(np.concatenate([panel(ref,f'Generated v01 / t={f/25:.2f}s'),panel(overlay,label)],1))
videos.append(writer.close())
metrics={v:json.loads((d/'evaluation/metrics.json').read_text()) for v,d in runs.items()};b0=json.loads((b0_root/'quality_metrics/metrics.json').read_text());report={'object_id':oid,'label':a.object,'paired_controls_verified':True,'runs':{},'B0':{'source':str(b0_root),'training_observations':'21x61','legacy_metrics':b0['aggregates']['validation_generated']['0.5'],'role':'unequal-observation-budget reference; not a pure backend ranking','video':'Previously encoded sigma8/15k reference; quantitative metrics from saved original evaluator, not decoded video'},'comparison_limits':['E0/E1 use12x31 and48k rendered samples; B0 uses21x61,15k optimizer steps with its own recipe','Reference is heldout generated observations, already used for tuning, not independent real unseen-view GT','White SSIM implementation specified for E0/E1 only; do not compare incompatible legacy SSIM','Bottom35% is image-region coverage proxy, not annotated foot accuracy'],'videos':videos,'source_hashes':source_hashes,'production_approved':False}
for v,d in runs.items():
 asset=clip/'06_assets'/v;asset.mkdir(exist_ok=True)
 for name in ['checkpoint.pt','model_spec.json','config.json','replay.json']:
  shutil.copyfile(d/name,asset/name);assert sha(asset/name)==sha(d/name)
 shutil.copyfile(clip/'manifest.json',asset/'manifest.json')
 shutil.copyfile(Path(m['coordinates']['root_transform_source']),asset/'root_motion.npz')
 save_json(asset/'asset.json',{'format':'Native Fudan GaussianModel.capture with model_spec + optimizer/RNG','checkpoint_sha256':sha(asset/'checkpoint.pt'),'manifest':str(clip/'manifest.json'),'coordinates':m['coordinates'],'time_duration':[0,1],'seconds_per_model_time':2.4,'replay_command':f'CUDA_VISIBLE_DEVICES=4 python {ROOT}/adapters/train.py --replay {d}/checkpoint.pt','render_command':f'CUDA_VISIBLE_DEVICES=4 python {ROOT}/adapters/render_asset.py --asset {asset} --output {asset}/standalone_motion.mp4','no_point_filtering':True,'snapshot_ply':'Not exported; full dynamic checkpoint is authoritative'})
 report['runs'][v]={'metrics':metrics[v]['aggregates'],'steps':complete[v]['steps'],'rendered_samples':complete[v]['rendered_samples'],'training_seconds':complete[v]['training_seconds'],'peak_allocated_bytes':complete[v]['peak_allocated_bytes'],'points':complete[v]['points'],'confidence_gradient_test':complete[v]['confidence_gradient_test']}
for pth,h in source_hashes.items():assert sha(pth)==h
save_json(clip/'08_evaluation/comparison.json',report);save_json(clip/'09_records/packaging.json',{'state':'complete','videos':videos,'source_hashes_rechecked':True})
# Plain local report entry; never touches previous object's browser or publishes remotely.
rows=''.join(f'<tr><td>{v}</td><td>{report["runs"][v]["metrics"]["all61"]["gray_foreground_psnr"]:.3f}</td><td>{report["runs"][v]["metrics"]["all61"]["gray_crop_alex_lpips"]:.4f}</td><td>{report["runs"][v]["metrics"]["all61"]["silhouette_iou"]:.4f}</td></tr>' for v in ['E0','E1'])
sections=''.join(f'<h2>{title}</h2><video controls preload="metadata" src="07_demo/{name}"></video>' for name,title in [('01_pipeline.mp4','真实输入 → 生成参考 → E1'),('02_backend_comparison.mp4','生成参考 / B0参考（21×61） / E0 / E1（12×31）'),('03_motion_fixed_views.mp4','固定相机：正、侧、背完整动作'),('04_geometry_orbit.mp4','固定动作时刻1.2秒：相机环绕'),('05_details.mp4','上身与附件、下肢区域对照'),('06_confidence.mp4','实际训练置信度；奇数时刻未使用')])
page=f'''<!doctype html><meta charset="utf-8"><title>{a.object} {oid} native4D</title><style>body{{background:#111827;color:#e5e7eb;font:16px system-ui;max-width:1440px;margin:30px auto;padding:20px}}video{{width:100%;max-height:760px;background:#333}}a{{color:#7dd3fc}}td,th{{padding:10px;border:1px solid #475569}}p{{line-height:1.7}}</style><h1>{a.object} · {oid} · 独立原生4D后端</h1><p>24路已有素材，61时刻、2.4秒。训练12路×31个偶数时刻；输出61时刻。E0/E1各12k优化、48k图像渲染，同初值/采样/优化预算。奇数时刻是未直接监督查询。留出相机0/8/16；参考为生成观测，非真实新视角真值。</p><p>B0为旧sigma8/15k、21×61观测，视频为旧结果编码参考，不能据此做纯后端排名。原生未滤点结果。视觉复核尚需结合报告；生产批准：否。</p><table><tr><th>运行</th><th>灰背景前景PSNR↑</th><th>裁剪Alex LPIPS↓</th><th>IoU↑</th></tr>{rows}</table><p><a href="08_evaluation/comparison.json">完整指标与比较边界</a> · <a href="manifest.json">相机、时间、来源哈希</a> · <a href="06_assets/E0/asset.json">E0动态资产</a> · <a href="06_assets/E1/asset.json">E1动态资产</a></p>{sections}'''
(clip/'index.html').write_text(page)
print(clip/'index.html',flush=True)
