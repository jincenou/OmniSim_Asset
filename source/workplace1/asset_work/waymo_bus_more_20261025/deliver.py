"""Portable standard bus asset, source evidence and a true 16-view preview."""
import json,sys,shutil,copy,zipfile,subprocess,tempfile,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
B=Path(__file__).resolve().parent;L=Path('/mnt/ojc/workplace1/asset_library')
sys.path.insert(0,'/mnt/ojc/workplace1/asset_work/vehicle1000_20261025')
sys.path.insert(0,'/mnt/wmj/workspace4/pipeline/traj_pipeline')
from runtime import read_json,write_json,sha256
from corner_case.assets import AssetLibrary,bind_rigid
from corner_case.planning.objects import AssetQuery
from asset_extraction.export import inspect_ply

def main():
    done=read_json(B/'completed.json')
    rows=[r for r in read_json(L/'manifests/asset_library_rigid.json')['assets'] if r['uid'].startswith(B.name+':') and r['status']=='keep']
    out=L/'previews/bus'/B.name;out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',22)
    packaged=[];page='<meta charset="utf-8"><title>新增 Waymo 公交车</title><style>body{max-width:1400px;margin:30px auto;background:#172331;color:white;font:18px sans-serif}a{color:#9dd7ff}img{max-width:100%}video{width:640px;max-width:100%}</style><h1>Waymo 新增公交车 / 客车</h1>'
    page+='<p>本批生成 5 辆，原严格筛选保留 1 辆、淘汰 4 辆。淘汰输出已清理，审核记录保留。下面展示的是保留资产。</p>'
    for row in rows:
        oid=row['provenance']['source_obj_id'];dest=out/oid;dest.mkdir(exist_ok=True)
        mapping={};local=copy.deepcopy(row);source=L/row['canonical']['canonical_path']
        assert sha256(source)==row['canonical']['sha256'];validation=inspect_ply(source)
        payload=dest/'asset.ply'
        if not payload.exists():
            import os
            os.link(source,payload)
        assert sha256(payload)==row['canonical']['sha256']
        def include(path,name):
            target=dest/name;shutil.copyfile(path,target);mapping[str(path)]=oid+'/'+name
        include(row['provenance']['source_image'],'source_foreground.jpg')
        include(row['provenance']['source']['source_image'],'source_frame.jpg')
        include(row['quality']['rendering']['evidence'],'orbit_16.jpg')
        include(row['quality']['audit_image'],'source_and_renders.jpg')
        for i,ref in enumerate(row['provenance']['real_images']):
            include(ref['rgba_path'],f'real_{i}.png');include(ref['source_image'],f'real_context_{i}.jpg')
        generated=next(r for r in read_json(B/'generation/generation.json')['assets'] if r['obj_id']==oid)
        for i,img in enumerate(generated['stages']['flux']['images']):include(img['path'],f'flux_{i}.png')
        def remap(value):
            if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
            if isinstance(value,list):return [remap(v) for v in value]
            return mapping.get(value,value) if isinstance(value,str) else value
        local=remap(local);local['source_path']=oid+'/asset.ply';local['canonical']['canonical_path']=oid+'/asset.ply'
        local['delivery']=dict(source_observations_are_provenance=True,source_camera_is_not_target_insertion_camera=True,
            visual_limitations='Source has tandem rear axles; the generated rear axle configuration differs. Minor rear texture artifacts remain despite automatic keep.',
            protocol='metadata-assisted identity selection; generated geometry and metric scale do not use sensor cuboids')
        write_json(dest/'metadata.json',local);packaged.append(local)
        write_json(dest/'source_camera.json',dict(source=local['provenance']['source'],real_images=local['provenance']['real_images'],
            note='Original scene camera calibration and timestamps only; insertion camera/trajectory must come from the target scene.'))
        sheet=Image.open(dest/'orbit_16.jpg');size=row['quality']['rendering']['resolution']
        frames=[sheet.crop((i%4*size,i//4*(size+24)+24,i%4*size+size,i//4*(size+24)+24+size)) for i in range(16)]
        with tempfile.TemporaryDirectory(prefix='bus_preview_',dir='/tmp') as tmp:
            for i,frame in enumerate(frames):frame.save(Path(tmp)/f'{i:02d}.png')
            subprocess.run(['/mnt/ojc/miniconda3/envs/splatad/bin/ffmpeg','-v','error','-y','-framerate','4','-i',str(Path(tmp)/'%02d.png'),
                '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'orbit_16.mp4')],check=True)
        canvas=Image.new('RGB',(1600,390),'#172331');draw=ImageDraw.Draw(canvas)
        pics=[Image.open(dest/'source_foreground.jpg'),frames[2],frames[6],frames[10],frames[14]]
        for i,im in enumerate(pics):
            draw.text((i*320+8,8),'SOURCE' if i==0 else f'{[45,135,225,315][i-1]} deg',font=font,fill='white')
            canvas.paste(ImageOps.pad(im.convert('RGB'),(312,330),color='white' if i==0 else '#808080'),(i*320+4,45))
        canvas.save(dest/'comparison.jpg',quality=94)
        page+=f'<h2>白色长途客车 · H + {len(row["provenance"]["real_images"])} 张真实图</h2><p>状态 keep；{validation["gaussian_count"]:,} 个高斯，16/16 视角非空，车头 +X、竖直 +Z。尺寸来自视觉估计，未烘焙到 PLY。</p><img src="{oid}/comparison.jpg"><p><video controls loop muted src="{oid}/orbit_16.mp4"></video></p>'
        page+=f'<p><a href="{oid}/asset.ply">Gaussian PLY</a> · <a href="{oid}/metadata.json">完整配置</a> · <a href="{oid}/source_camera.json">源相机与参考图记录</a> · <a href="{oid}/source_and_renders.jpg">原图与 16 视角</a></p><p>可见局限：原图双后轴与生成资产的后轴配置存在差异，尾部仍有局部纹理瑕疵。自动通过不代表完全复刻。</p>'
    write_json(out/'asset_library_rigid.json',dict(schema_version=2,total=len(packaged),assets=packaged))
    library=AssetLibrary(out/'asset_library_rigid.json',out);candidates=library.candidates(AssetQuery(category='bus'))
    assert len(candidates)==len(packaged)>0
    for entry in candidates:
        bound=bind_rigid(entry,np.random.default_rng(42));assert bound.asset_path.is_file()
        write_json(out/entry.raw['provenance']['source_obj_id']/'binding.json',dict(uid=bound.uid,
            canonical_path=entry.raw['canonical']['canonical_path'],scale=float(bound.scale),size_lwh_m=list(bound.size_lwh_m),
            coordinate=dict(up='+Z',forward='+X',ground_z=0,canonical_transform_already_applied=True,metric_scale_baked=False),
            scale_source='Original bind_rigid with seed 42; VLM estimated length and original +/-6% scale jitter',
            insertion_scene=None,insertion_trajectory=None))
    write_json(out/'validation.json',dict(queries=done['queries'],counts=done['counts'],legacy_loader='corner_case.assets.AssetLibrary',
        bus_category_query_count=len(candidates),binder_validated=True,packaged_uids=[r['uid'] for r in packaged]))
    (out/'README.txt').write_text('解压后从 START_HERE.html 查看。asset_library_rigid.json 可交给旧 AssetLibrary，pipeline_root 指向解压目录。\nasset.ply 已完成朝向、居中、接地；不要再次应用 canonical.rotation/translation。\n插入时按 binding.json 施加一次各向同性 scale，然后由目标场景提供位移、yaw 和轨迹。\nsource_camera.json 是来源相机，不是目标场景相机。尺寸为估计值。\n视频直接使用 16 个实渲染方位，各差 22.5 度；不是场景插入视频。\n保留资产存在后轴配置差异和尾部纹理瑕疵，详见页面。\n')
    (out/'START_HERE.html').write_text(page)
    archive=L/'deliveries'/f'{B.name}_kept.zip';archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,Path(out.name)/path.relative_to(out))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write_json(B/'delivery.json',dict(preview=str(out/'START_HERE.html'),archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha256(archive),kept=len(rows),legacy_query_verified=True))
    print(json.dumps(read_json(B/'delivery.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
