from pathlib import Path
import json,sys,shutil,subprocess,csv
R=Path(__file__).resolve().parents[1];D=R/'deliveries/nonvehicle10_and_scene_validation_20261023';D.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(R/'tools/report_deps'))
from docx import Document
from docx.shared import Inches
import imageio_ffmpeg
from concurrent.futures import ThreadPoolExecutor
N=R/'nonvehicle10';S=R/'scene_validation';cases=json.loads((N/'inputs/cases.json').read_text());jobs=[]
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def report_docs(root,figs):
 doc=Document(root/'REPORT.docx');doc.add_heading('关键可视化对比',1)
 for label,p in figs:doc.add_paragraph(label);doc.add_picture(str(p),width=Inches(6.5))
 doc.save(root/'REPORT.docx')
report_docs(N,[(name,N/'results'/cid/'comparison_orbit.jpg') for name,cid in [('绿灯：状态与几何的不同失败','nonvehicle_01_green_light_urban'),('消防栓：正面保色与背面漂移','nonvehicle_07_white_hydrant'),('路锥：外观与立体结构','nonvehicle_10_orange_cone')]])
report_docs(S,[(scene,S/'results'/scene/'random10_04_white_suv/comparison_frame_030.jpg') for scene in ['scene-0103','scene-0104','scene-0277']])
for root,sub in [(N,'nonvehicle10'),(S,'scene_validation')]:
 for name in ['START_HERE.html','REPORT.md','REPORT.txt','REPORT.html','REPORT.docx']:copy(root/name,D/sub/name)
 for p in (root/'docs').glob('*'):
  if p.suffix in ['.json','.csv']:copy(p,D/sub/'docs'/p.name)
for c in cases:
 cid=c['case_id']
 for name in ['source_full.jpg','real_00.png','real_00_mask.png','flux_00.png','flux_00_rgba.png','manifest.json','segmented_manifest.json','flux_generation.json']:copy(N/'inputs'/cid/name,D/'nonvehicle10/inputs'/cid/name)
 for name in ['comparison_video.mp4','comparison_orbit.jpg']:copy(N/'results'/cid/name,D/'nonvehicle10/results'/cid/name)
for name in ['cases.json','fresh_flux_manifest.json']:copy(N/'inputs'/name,D/'nonvehicle10/inputs'/name)
for scene in ['scene-0103','scene-0104','scene-0277']:
 for cid in ['random10_04_white_suv','random10_02_white_pickup']:
  base=S/'results'/scene/cid;dest=D/'scene_validation/results'/scene/cid
  for name in ['comparison_frame_000.jpg','comparison_frame_030.jpg','comparison_frame_059.jpg','comparison_crop.json']:copy(base/name,dest/name)
  jobs.append((base/'comparison_scene.mp4',dest/'comparison_scene.mp4',1280))
 bg=S/'results'/scene/'random10_04_white_suv/background/insert_splatad_6cam.mp4';jobs.append((bg,D/'scene_validation/background'/f'{scene}.mp4',1200))
 for name in [f'{scene}_traj.npz',f'{scene}_plan.json']:copy(S/'inputs'/name,D/'scene_validation/inputs'/name)
copy(S/'inputs/scenes.json',D/'scene_validation/inputs/scenes.json')
def transcode(job):
 src,dst,width=job;dst.parent.mkdir(parents=True,exist_ok=True);cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(src),'-vf',f'scale={width}:-2','-c:v','libx264','-crf','25','-preset','medium','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(dst)];subprocess.run(cmd,check=True);print('COMPRESSED',dst.name,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(transcode,jobs))
index=[]
for c in cases:
 for v in ['B0','R','H','B0_real1']:
  base=N/'results'/c['case_id']/f'{v}_seed42';index.append(dict(case_id=c['case_id'],variant=v,gaussian_ply=str(base/'asset/canonical.ply'),metadata=str(base/'asset/metadata.json'),orbit_video=str(base/'orbit/turntable.mp4')))
(D/'GAUSSIAN_PATHS.json').write_text(json.dumps(index,indent=2));fullvideos=[str(S/'results'/scene/cid/v/'insert_splatad_6cam.mp4') for scene in ['scene-0103','scene-0104','scene-0277'] for cid in ['random10_04_white_suv','random10_02_white_pickup'] for v in ['B0','R','H','B0_real4']];(D/'FULL_SCENE_VIDEO_PATHS.json').write_text(json.dumps(fullvideos,indent=2))
for p in (R/'scripts/nonvehicle').glob('*.py'):copy(p,D/'reproducibility/nonvehicle'/p.name)
for name in ['detect_nonvehicle_candidates.py','extract_nonvehicle_waymo.py','extract_nonvehicle_nuscenes.py','freeze_nonvehicle_inputs.py','insert_scene_validation.py','insert_scene_grounded.py','run_scene_validation_batch.py','plan_scene_validation.py','evaluate_scene_validation.py','audit_scene_validation.py','evaluate_asset_metrics.py','evaluate_pixel_metrics.py','asset_adapter.py','runtime.py','preprocessing.py','compose_scene_validation.py','build_nonvehicle_report.py','build_scene_validation_report.py']:copy(R/'scripts'/name,D/'reproducibility'/name)
css='body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:auto;padding:32px;line-height:1.7}a{color:#2058a0}section{padding:20px;background:#f3f6fa;margin:24px 0}video{max-width:100%}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}'
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>非车辆十例与三场景插入验证</title><style>'''+css+'''</style><h1>非车辆10例与三场景插入验证</h1><p>解压后双击此文件即可离线展示。10个非车辆 × 4路线 = 40个Gaussian结果；3场景 × 2车辆 × 4路线 = 24组插入。包内包含16个对比视频、3个背景参考，以及源图、图片和可编辑报告。</p><section><h2>先看非车辆实验</h2><p><a href="nonvehicle10/START_HERE.html">十个demo与指标总览</a> · <a href="nonvehicle10/REPORT.html">完整报告</a> · <a href="nonvehicle10/REPORT.docx">Word报告</a> · <a href="nonvehicle10/docs/summary.csv">指标表CSV</a></p><p>R在源外观指标上较强，H在部分实体几何上更完整；路牌文字、灯态和薄片等失败必须单独看。AOR100%不代表可直接入库。</p></section><section><h2>再看车辆插入场景</h2><p><a href="scene_validation/START_HERE.html">六组四路线插入对比</a> · <a href="scene_validation/REPORT.html">验证报告</a> · <a href="scene_validation/REPORT.docx">Word报告</a> · <a href="scene_validation/docs/insertion_diagnostics.csv">逐组诊断CSV</a></p><p>24组实际调用原SplatAD渲染器。仍可见悬浮感、接触阴影缺失和偏色；不能只凭CLIP/DINO选定生产方案。这里插入的是车辆，新增非车辆尚未做场景试插。</p></section><h2>无插入背景参考</h2>'''
for scene in ['scene-0103','scene-0104','scene-0277']:html+=f'<p>{scene}</p><video controls preload="none" src="scene_validation/background/{scene}.mp4"></video>'
html+='''<h2>高斯与完整数据位置</h2><p>为保持压缩包轻量，不包含40份PLY、场景checkpoint和24个原始六相机高清视频；它们已留在服务器。<a href="GAUSSIAN_PATHS.json">高斯路径清单</a> · <a href="FULL_SCENE_VIDEO_PATHS.json">完整场景视频清单</a>。包内场景视频仅做分辨率/码率压缩，帧数和时间不变；非车辆对比视频原样复制。</p><p>计算口径、失败说明、来源与脚本都在两个报告及reproducibility目录；脚本依赖服务器工程与权重，不是独立模型分发包。</p></html>''';(D/'START_HERE.html').write_text(html)
(D/'README.txt').write_text('解压后打开 START_HERE.html。\n10个非车辆四路线视频 + 6个车辆场景四路线视频 + 3个背景参考。\n非车辆指标：AOR、CLIP-I16、DINO-I16；PSNR/SSIM/LPIPS为环绕外观诊断，不是之前matched-source表A/B。ED-R因缺可靠对齐未计算。\n场景中的对象是车辆，非车辆尚未试插。\n重要限制与最终选择见两份REPORT.docx/html。\n高斯PLY和完整六相机视频在服务器，路径见GAUSSIAN_PATHS.json和FULL_SCENE_VIDEO_PATHS.json。\n')
print('PACKAGE STAGED',D,flush=True)
