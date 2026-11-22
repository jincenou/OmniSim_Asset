"""Publish complete results only; no provisional side-by-side comparisons."""
import argparse,html,json,os
from pathlib import Path
import numpy as np
import av
from PIL import Image,ImageDraw,ImageOps
from prepare_person import ROOT,selected,encode,save,sha

def frames(path):
    with av.open(str(path)) as con:
        con.streams.video[0].thread_count=2
        images=[]
        for i,f in enumerate(con.decode(video=0)):
            assert abs(float(f.pts*f.time_base)-i/25)<1e-7
            images.append(f.to_image())
    assert len(images)==121
    return images

def tile(image,size,label):
    canvas=Image.new('RGB',size,(128,128,128));im=ImageOps.contain(image,(size[0],size[1]-28))
    canvas.paste(im,((size[0]-im.width)//2,28+(size[1]-28-im.height)//2))
    ImageDraw.Draw(canvas).text((7,7),label,fill='white');return canvas

def generated(ident):
    base=ROOT/'frontend'/ident
    assert json.loads((base/'research/generated_validation.json').read_text())['state']=='all24_videos_decoded_same_retained_cameras'
    output=ROOT/'previews'/ident;output.mkdir(parents=True,exist_ok=True)
    target=output/'generated_multiview.mp4';assert not target.exists()
    streams=[frames(base/'outputs/turbo/videos'/f'{v:02d}.mp4') for v in (0,6,12,18)]
    def images():
        for i in range(121):
            canvas=Image.new('RGB',(704,1336),(128,128,128))
            for j,v in enumerate((0,6,12,18)):
                canvas.paste(tile(streams[j][i],(352,668),f'GENERATED VIDEO yaw {v*15}; t={i/25:.2f}s'),((j%2)*352,(j//2)*668))
            yield canvas
    encode(target,images())

def comparison(ident):
    folder=ROOT/'previews'/ident
    assert json.loads((folder/'independent_replay.json').read_text())['state']=='asset_only_replay_passed_quality_unreviewed'
    original=frames(ROOT/'inputs'/ident/'source_clip.mp4');asset=frames(folder/'asset_view00.mp4')
    target=folder/'source_vs_asset.mp4';assert not target.exists()
    def images():
        for i in range(121):
            canvas=Image.new('RGB',(1312,668),(128,128,128))
            canvas.paste(tile(original[i],(960,668),f'REAL SOURCE t={i/25:.2f}s'),(0,0))
            canvas.paste(tile(asset[i],(352,668),'DYNAMIC ASSET; different camera'),(960,0))
            if i==60:canvas.save(folder/'source_vs_asset.jpg',quality=94)
            yield canvas
    encode(target,images())
    save(folder/'media_validation.json',dict(frames=121,fps=25,source_and_asset_camera_aligned=False,
        independent_replay_required=True,quality='unreviewed',files={n:sha(folder/n) for n in ['source_vs_asset.mp4','source_vs_asset.jpg','asset_fixed_views.mp4','asset_orbit.mp4']}))

def montage():
    from publish_status import collect
    _,live,people=collect()
    assert not live,'Do not publish a provisional batch montage while the queue is running'
    assert len(people)==8 and any(p['asset_complete'] for p in people)
    terminal={'asset_replayed_quality_unreviewed','segmentation_gate_failed','stage_failed','camera_interface_not_applicable_pending'}
    assert all(p['state'] in terminal for p in people),'Unfinished people cannot silently become failure cards'
    streams={}
    evidence=[]
    for person in people:
        ident=person['id']
        if person['asset_complete']:
            path=ROOT/person['preview_root']/'asset_fixed_views.mp4'
            replay=json.loads((path.parent/'independent_replay.json').read_text())
            assert sha(path)==replay['videos'][path.name]['sha256']
            streams[ident]=frames(path)
            evidence.append(dict(id=ident,kind='independently_replayed_asset',source=str(path),sha256=sha(path)))
        else:
            evidence.append(dict(id=ident,kind='explicit_failure_or_interface_card',state=person['state'],reason=person['issue']))
    def images():
        import textwrap
        for index in range(121):
            canvas=Image.new('RGB',(1408,1336),(128,128,128))
            for slot,person in enumerate(people):
                ident=person['id']
                if ident in streams:
                    quality='known quality issues' if person['quality']=='completed_with_quality_issues' else 'quality unreviewed'
                    cell=tile(streams[ident][index],(352,668),f'{ident} FINAL ASSET; {quality}')
                else:
                    cell=Image.new('RGB',(352,668),(45,45,45));draw=ImageDraw.Draw(cell)
                    text=f'{ident}\nNO FINAL ASSET\n{person["state"]}\n{person["issue"]}'
                    draw.multiline_text((12,30),'\n'.join(textwrap.fill(line,42) for line in text.splitlines()),fill='white',spacing=8)
                canvas.paste(cell,((slot%4)*352,(slot//4)*668))
            yield canvas
    target=ROOT/'gallery/montage_current.mp4'
    encode(target,images())
    save(ROOT/'gallery/montage_current_validation.json',dict(frames=121,fps=25,sha256=sha(target),
        scope='All eight frozen identities; real asset renders only; no source-video substitution',people=evidence))

def gallery():
    manifest=json.loads((ROOT/'SELECTED_TRACKS.json').read_text());cards=[]
    status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').is_file() else {}
    statuses={p['id']:p for p in status.get('people',[])}
    for row in manifest['tracks']:
        ident=row['id'];elements=[]
        actual=statuses.get(ident,{})
        prefix=actual.get('preview_root',f'previews/{ident}')
        branch=Path(prefix).parents[1]
        replay_path=ROOT/prefix/'independent_replay.json'
        replay_ok=replay_path.is_file() and json.loads(replay_path.read_text()).get('state')=='asset_only_replay_passed_quality_unreviewed'
        for label,relative in [('真实源片段',f'inputs/{ident}/source_clip.mp4'),('实际外观输入',f'inputs/{ident}/source_crop.mp4'),
                ('原图骨架叠加',f'inputs/{ident}/pose_preview.mp4'),('生成多视角视频（非资产）',f'previews/{ident}/generated_multiview.mp4'),
                ('最终资产固定视角',f'previews/{ident}/asset_fixed_views.mp4'),('固定动作 360° 环绕',f'previews/{ident}/asset_orbit.mp4'),
                ('源视频与资产（不同相机）',f'previews/{ident}/source_vs_asset.mp4')]:
            if str(branch) != '.': relative=str(branch/relative)
            if ('asset_' in relative or 'source_vs_asset' in relative) and not replay_ok:continue
            if 'source_vs_asset' in relative and not (ROOT/prefix/'media_validation.json').is_file():continue
            if (ROOT/relative).is_file():elements.append(f'<p>{label}</p><video controls preload="none" src="../{relative}"></video>')
        state='未完成资产；尚未通过质量检查'
        mask_gate=ROOT/'inputs'/ident/'research/batch_gate_mask.json'
        if mask_gate.is_file() and not json.loads(mask_gate.read_text())['passed']:
            state='分割数值门槛未通过，未继续生成；保留源片段和失败输出供核查'
        if replay_ok:state='完整资产已独立回放；外观/动作质量待审查'
        if row['camera_class'] in ('moving','unknown'):state='移动/未知相机：接口不适用，单列待处理，未强行运行静态假设'
        if actual.get('asset_variant')=='accessory_repaired':
            elements.insert(0,'<p><strong>当前资产：携带物修复版；旧版模型已退役，仅保留对照媒体和历史记录。</strong></p>')
        if actual.get('generated_review_path'):
            elements.append(f'<p><a href="../{html.escape(actual["generated_review_path"])}">生成视频的视觉检查范围与问题记录</a></p>')
        if actual.get('known_frontend_observations'):
            elements.append('<p><strong>高斯重建前已发现的问题：</strong></p><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in actual['known_frontend_observations'])+'</ul>')
        if replay_ok and actual.get('quality')=='completed_with_quality_issues':
            state='完整资产已独立回放；已发现质量问题，具体检查范围见记录'
        if replay_ok:
            active_asset_path=actual.get('asset_path',f'assets/{ident}')
            elements.append(f'<p><a href="../{active_asset_path}/package_manifest.json">完整动态资产清单</a> · <a href="../{active_asset_path}/replay.sh">独立回放脚本</a></p>')
        if actual.get('asset_review_path'):
            elements.append(f'<p><a href="../{html.escape(actual["asset_review_path"])}">资产视觉检查范围与问题记录</a></p>')
        if actual.get('asset_observations'):
            elements.append('<ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in actual['asset_observations'])+'</ul>')
        repair=f'diagnostics/accessory_{ident}_v1'
        if ident in ('W02','W04') and (ROOT/repair/'input_comparison.mp4').is_file():
            elements.append(f'<p><a href="../{repair}/REPORT.md">独立随身物修复测试：原因与进度</a> · <a href="../{repair}/input_comparison.mp4">输入修复前后视频（不是最终资产）</a></p>')
            comparison=ROOT/repair/'COMPARISON.json'
            if comparison.is_file() and (comparison.parent/'generation_comparison.mp4').is_file():
                elements.append(f'<p>相同骨架、相机、seed 的原输入／随身物修复输入生成对照（非资产；质量见检查记录）</p><video controls preload="none" src="../{repair}/generation_comparison.mp4"></video>')
            if (comparison.parent/'ASSET_COMPARISON.json').is_file():
                elements.append(f'<p>原输入／随身物修复输入的最终高斯资产对照（均已独立回放；质量仍需视觉核查）</p><video controls preload="none" src="../{repair}/asset_comparison.mp4"></video>')
        progress=html.escape(f'{actual.get("stage", "not_started")} · {actual.get("state", "pending")} · {actual.get("issue", "")}')
        cards.append(f'<section><h2>{ident} · {html.escape(row["source_id"])}</h2><p>{html.escape(row["reason"])}</p><p>{row["camera_class"]} · {state}</p><p>{progress}</p>'+''.join(elements)+'</section>')
    folder=ROOT/'gallery';folder.mkdir(exist_ok=True)
    overview='<p><a href="../RESULTS.csv">逐人结果 CSV</a> · <a href="../REPORT.md">报告</a> · <a href="../README_RUN.md">实际运行与回放命令</a> · <a href="../sources/ATTRIBUTION.md">来源与许可</a></p>'
    if (folder/'montage_current_validation.json').is_file():overview+='<p>八人终态总览：失败与接口不适用对象保留文字位置。</p><video controls preload="none" src="montage_current.mp4"></video>'
    content='<!doctype html><meta charset="utf-8"><title>网络行人 G0 / Turbo / Grid4D A</title><style>body{background:#202124;color:#eee;font:16px sans-serif;margin:32px}section{border-top:1px solid #555;padding:16px}video{max-width:100%;max-height:680px}a{color:#8cf}</style><h1>固定八人：真实输入 → 多视角生成 → 动态资产</h1><p>前端输入适配 v2；静态相机假设不等于世界轨迹恢复。所有冻结对象均保留。只展示已存在且完成阶段验证的媒体。</p>'+overview+''.join(cards)
    temporary=folder/f'index.{os.getpid()}.tmp';temporary.write_text(content);temporary.replace(folder/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','comparison','gallery','montage']);p.add_argument('id',nargs='?');a=p.parse_args()
    if a.stage=='gallery':gallery()
    elif a.stage=='montage':montage();gallery()
    else:selected(a.id);{'generated':generated,'comparison':comparison}[a.stage](a.id);gallery()
