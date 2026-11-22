from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];W=R.parent;B=R/'benchmark'
def save(name,obj):
 p=B/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
labels=['walk063_rootlocal24','batch_0084_twentyfour','phone_full','pilot_043_twentyfour','batch_0073_twentyfour','batch_0086_twentyfour','batch_0055_twentyfour','batch_0062_twentyfour']
actions=['真实行走，root-local 改动分支','迈步／转身候选，需全片复核','站立／看手机／上肢动作','站立／上肢小动作','站立／手机／肩包','站立／手机／背包','安全帽／反光背心／栅栏遮挡','重心变化／生成头发外观失败']
tracks=[];jobs=[]
def job(label,recipe,metric):
 m=json.loads(metric.read_text());cp=Path(m['checkpoint']);data=Path(m['inputs'][0]['path']).parents[2]
 if not (data/'images/cam00/000000.png').exists():raise ValueError(data)
 jobs.append({'id':label+'_'+recipe,'label':label,'recipe':recipe,'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'data':str(data)})
for label,action in zip(labels,actions):
 walking=label.startswith('walk063');out=(R/'reconstruction/outputs' if walking else W/'reconstruction/outputs')/label
 m=out/'quality_metrics/metrics.json' if walking else R/'quality'/f'baseline_{label}'/'quality_metrics/metrics.json'
 job(label,'q0',m)
 tracks.append({'label':label,'behavior':action,'branch':'root_local_modified' if walking else 'current_static_camera_waymo_preprocessing','role':'tuning' if label in labels[2:4] else 'validation' if label in labels[4:6] else 'coverage_only','demo':str(out/'demo/four_panel.mp4' if walking else R/'demos'/label/'four_panel.mp4'),'asset':str(out/'asset.json'),'unmodified_official_end_to_end':False})
for label in labels[2:6]:
 base='Q50' if label in labels[2:4] else 'budget_half_50k'
 job(label,'budget_half_50k',R/'quality'/f'{label}_{base}'/'quality_metrics/metrics.json')
 job(label,'capacity50k',R/'quality'/f'{label}_capacity50k'/'quality_metrics/metrics.json')
protocol={'id':'waymo-generated-consistency-v1','scope':'Feasible local comparison only, not exact paper reproduction','reference':'saved generated-view RGBA observations','views':[0,8,16],'training_views':[v for v in range(24) if v not in (0,8,16)],'frames':list(range(121)),'resolution_wh':[352,640],'background_rgb':[1,1,1],'target_composition':'straight RGB * saved predicted alpha + 1 - alpha','prediction_composition':'premultiplied raster RGB + 1 - raster alpha; clamp[0,1]','crop':'none','resize':'none','primary_metrics':{'PSNR':'per-image RGB MSE, data_range1, MSE floor1e-12, arithmetic mean','SSIM':'torchmetrics functional, Gaussian11 sigma1.5 k1.01 k2.03 data_range1 RGB','LPIPS':'torchmetrics AlexNet pretrained, normalize=True, whole frame'},'aggregation':'image->view->track; cross-track macro mean; paired deltas require identical input SHA and protocol SHA','diagnostics':['foreground_psnr','silhouette_iou','outside_alpha_per_foreground_pixel'],'mask_source':'predicted SAM2/BiRefNet, not real ground truth','excluded':['generation versus real novel-view ground truth','GS versus real novel-view ground truth','paper-table numerical reproduction','new experiments solely to match paper cameras or dataset'],'paper_reference':'https://arxiv.org/html/2608.20335#S4','paper_differences':['Waymo not DNA-Rendering/DyMVHumans','24 total,21 train,3 heldout not16/12/4','121 frames not98 main-evaluation frames','Turbo4 notpaper20','independent FreeTimeGsVanilla, not unpublished author backend','local foreground background/resolution/metric implementation fixed; author metric background/crop/LPIPS network not identified in inspected material'],'interpretation':'Consistency improvement may fit hallucinations more accurately. Does not establish true unseen-view appearance.','uncertainty':'Single seed per current recipe: descriptive results, not statistical significance. Small improvements require paired repeats on2 tracks with3 seeds. Correlated frames are not independent subjects.'}
save('protocol.json',protocol)
save('manifest.json',{'state':'frozen_before_whole_frame_scoring','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tracks':tracks,'distinct_tracks':8,'jobs':jobs,'paired_50k_tracks':4,'selection':'Existing assets selected for behavior/failure coverage, not random sampling. Two tuning and two independent validation tracks distinguished.'})
print('Frozen8 tracks and16 evaluations')
