"""Verify publication and add a small source/render preview next to the asset bank."""
import json,sys,shutil,html
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
P=Path('/mnt/ojc/pre_flux_pipeline');B=Path(__file__).resolve().parent;L=Path('/mnt/ojc/workplace1/asset_library')
sys.path.insert(0,str(P));sys.path.insert(0,'/mnt/wmj/workspace4/pipeline/traj_pipeline')
from asset_extraction.export import inspect_ply,_sha
from asset_extraction.io_utils import read_json,write_json
from corner_case.assets import AssetLibrary,bind_rigid
status=read_json(B/'status.json');assert status['stage']=='completed',status
rows=read_json(L/'manifests/asset_library_rigid.json')['assets'];before=read_json(B/'central_manifest_before.json')['assets']
byid={r['uid']:r for r in rows}
assert all(byid[r['uid']]==r for r in before), 'Existing bank entries changed'
new=[r for r in rows if r['uid'].startswith('waymo_bus_20261025:')];assert len(new)==2
OUT=L/'previews/bus/waymo_bus_20261025';OUT.mkdir(parents=True,exist_ok=True)
f=ImageFont.truetype('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',23)
canvas=Image.new('RGB',(2000,1060),'#172331');draw=ImageDraw.Draw(canvas)
gen={r['obj_id']:r for r in read_json(B/'generation/generation.json')['assets']}
accepted={r.uid:r for r in AssetLibrary(L/'manifests/asset_library_rigid.json',L).entries}
records=[]
page='<meta charset="utf-8"><title>Waymo 公交车资产</title><style>body{font:18px sans-serif;background:#152130;color:#eef;max-width:1500px;margin:30px auto}a{color:#9dd7ff}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #456;padding:10px}</style><h1>Waymo 公交车 / 大巴资产</h1>'
page+='<p>使用原来的 H 路线和严格筛选规则；复用可追溯的检测、SAM 分割与真实参考图，重新执行源图审核、FLUX、H、标准化和 16 视角审核。没有改动已有资产。估计尺寸不是传感器实测尺寸。</p>'
page+='<img src="comparison.jpg"><table><tr><th>对象</th><th>路线</th><th>真实图</th><th>状态</th><th>PLY / 配置</th></tr>'
for i,row in enumerate(new):
    oid=row['provenance']['source_obj_id'];g=gen[oid];dest=OUT/oid;dest.mkdir(exist_ok=True)
    p=L/row['canonical']['canonical_path'];v=inspect_ply(p);assert _sha(p)==row['canonical']['sha256']
    rendering=row['quality']['rendering'];assert rendering['all_16_nonempty'] and len(rendering['foreground_pixels'])==16
    write_json(dest/'metadata.json',row)
    write_json(dest/'source_camera.json',dict(source=row['provenance']['source'],reference_images=row['provenance']['real_images'],native_object_pose_included=False,note='Source camera/provenance; not an object pose or target insertion camera.'))
    shutil.copyfile(row['provenance']['source_image'],dest/'source.jpg')
    shutil.copyfile(row['provenance']['source']['source_image'],dest/'source_frame.jpg')
    shutil.copyfile(rendering['evidence'],dest/'orbit_16.jpg')
    shutil.copyfile(row['quality']['audit_image'],dest/'source_and_renders.jpg')
    for j,img in enumerate(g['stages']['flux']['images']):shutil.copyfile(img['path'],dest/f'flux_{j}.png')
    for j,img in enumerate(row['provenance']['real_images']):shutil.copyfile(img['rgba_path'],dest/f'real_{j}.png')
    binding=None
    if row['uid'] in accepted:
        bound=bind_rigid(accepted[row['uid']],np.random.default_rng(42))
        assert bound.asset_path==p and min(bound.size_lwh_m)>0
        binding=dict(uid=bound.uid,asset_path=str(bound.asset_path),scale=float(bound.scale),size_lwh_m=list(bound.size_lwh_m),category=row['category'],canonical_up='+Z',canonical_forward='+X',canonical_transform_already_applied=True,metric_scale_baked=False,scale_source='Original binder, estimated length with seed42 +/-6% scale jitter')
        write_json(dest/'binding.json',binding)
    record=dict(uid=row['uid'],obj_id=oid,status=row['status'],real_count=len(row['provenance']['real_images']),category=row['category'],subtype=row['label']['sub_type'],gaussian_count=v['gaussian_count'],ply=str(p),sha256=row['canonical']['sha256'],metadata=str(dest/'metadata.json'),source_camera=str(dest/'source_camera.json'),heading_verified=row['canonical']['heading_verified'],estimated_dimensions=row['scale'],binding=binding,quality_audit=row['quality']['audit'])
    records.append(record)
    ren=Image.open(rendering['evidence']);s=rendering['resolution'];imgs=[Image.open(dest/'source.jpg').convert('RGB')]
    for angle in [2,6,10,14]:
        x=angle%4*s;y=angle//4*(s+24)+24;imgs.append(ren.crop((x,y,x+s,y+s)).convert('RGB'))
    y=i*520;draw.text((15,y+10),f'{oid} | H + {len(row["provenance"]["real_images"])} real | {row["status"].upper()} | {v["gaussian_count"]:,} Gaussians',font=f,fill='white')
    for j,img in enumerate(imgs):
        x=j*400;draw.text((x+12,y+48),'SOURCE' if j==0 else f'{[45,135,225,315][j-1]} deg',font=f,fill='#bccbdd');canvas.paste(ImageOps.pad(img,(385,425),color='white' if j==0 else '#808080'),(x+7,y+80))
    page+=f'<tr><td>{html.escape(oid)}</td><td>H</td><td>{len(row["provenance"]["real_images"])}</td><td>{row["status"]}</td><td><a href="{p}">Gaussian PLY</a> / <a href="{oid}/metadata.json">metadata</a> / <a href="{oid}/source_and_renders.jpg">原图与 16 视角</a></td></tr>'
canvas.save(OUT/'comparison.jpg',quality=94)
page+='</table><p>PLY 位于中央资产库 payloads/candidate/ply/bus；metadata 中相对路径以 /mnt/ojc/workplace1/asset_library 为根目录。原 loader 只接纳 keep 条目；review/reject 仍保存为候选记录，不代表合格入库。</p>'
for r in records:page+=f'<h2>{r["obj_id"]}</h2><p>审核状态：{r["status"]}；原始审核说明：{html.escape(json.dumps(r["quality_audit"],ensure_ascii=False))}</p><img loading="lazy" src="{r["obj_id"]}/source_and_renders.jpg">'
(OUT/'START_HERE.html').write_text(page)
write_json(OUT/'manifest.json',dict(assets=records,counts=dict(Counter(r['status'] for r in new)),original_bank_entries_unchanged=len(before)))
index=L/'rigid_index.html';content=index.read_text() if index.exists() else '<meta charset="utf-8">'
marker='id="waymo_bus_20261025"'
if marker not in content:
    content=f'<section {marker}><a href="previews/bus/waymo_bus_20261025/START_HERE.html">2026-10-25 新增两辆 Waymo 公交车：原图、配置、PLY 与审核状态</a></section>'+content
    index.write_text(content)
write_json(B/'delivery.json',dict(preview=str(OUT/'START_HERE.html'),comparison=str(OUT/'comparison.jpg'),assets=records,previous_entries_unchanged=len(before)))
print(json.dumps(dict(preview=str(OUT/'START_HERE.html'),counts=dict(Counter(r['status'] for r in new)),gaussians=[r['gaussian_count'] for r in records]),ensure_ascii=False,indent=2))
