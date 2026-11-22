"""Local comparison entry for moving-camera inputs and retained failures."""
import html
import json
import os
from pathlib import Path
from datetime import datetime,timezone

R=Path(__file__).resolve().parents[1]
OUT=R/'moving_camera'
def read(p):return json.loads(p.read_text())
def main():
    OUT.mkdir(exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><title>移动相机行走实验</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;line-height:1.6}.row{display:flex;gap:16px;flex-wrap:wrap}video{width:280px;max-height:520px}section{margin:36px 0}a{overflow-wrap:anywhere}</style>',
        '<h1>真实行走输入与相机条件对照</h1><p>R07、M02已完成61帧动态高斯技术验收，画面质量未通过；M02有明显腿脚缺失。R01所选窗口相机移动1.159米；R07清晰后段相机仅移动0.074米，属于近停住对照，不能用于证明持续移动相机适配。M0使用估计静态相机；M2使用校准内参与相机旋转。M02实际曝光中心相机在所选窗口移动2.94米。M2没有使用相机平移，不代表恢复了准确世界轨迹。</p>',
        '<p><a href="../REPORT.md">进度报告</a> · <a href="../../objects/index.html">已完成资产</a></p>']
    cases=[]
    for name in ('R01','R07','M02'):
        source=R/'experiments'/f'moving_{name}_short61_v1'
        mapping=read(source/'inputs/frame_mapping.json')
        rel=lambda p:html.escape(os.path.relpath(p,OUT),quote=True)
        state={'R01':'源遮挡传入生成；不扩展重建','R07':'真实行走、车辆近停住；24视角与5k短片段资产完成，重建模糊','M02':'持续移动相机2.94米＋真实行走；24视角及5k资产完成，多时刻腿脚重建失败'}[name]
        case={'id':name,'source':str(source),'state':state,'frames':61,'variants':{}}
        parts.append(f'<section><h2>{name} · {state}</h2><p>25张真实观测 → 61个25fps时间槽：13个原始时刻、48个插值，采样跨度2.4秒。视频中的新视角由模型推断。</p><div class="row">')
        parts.append(f'<div>实际人物输入<br><video controls preload="metadata" src="{rel(source/"inputs/main.mp4")}"></video></div>')
        for variant in ('M0','M2'):
            output=R/'experiments'/f'moving_{name}_short61_{variant}_v1'/'outputs/smoke'
            validation=output/'preview/validation.json'
            if validation.exists() and read(validation).get('status')=='validated':
                video=output/'preview/multiview.mp4'
                if not video.is_file():raise ValueError('Validated preview is missing')
                case['variants'][variant]={'state':'generated_decoded_visual_review_separate','video':str(video)}
                parts.append(f'<div>{variant} 六视角<br><video controls preload="metadata" src="{rel(video)}"></video><br><a href="{rel(output/"short_preview/contact.jpg")}">首中末检查图</a></div>')
            else:
                case['variants'][variant]={'state':'not_yet_validated'}
                parts.append(f'<div>{variant}：尚未完成验收</div>')
        parts.append('</div><ul>')
        for title,path in [('真实帧及校准裁剪记录',source/'native.json'),('真实/插值时间与相机映射',source/'inputs/frame_mapping.json'),('运动投影复核',source/'motion_pair/review.json'),('运动投影图',source/'motion_pair/projection_contact.jpg')]:
            if path.is_file():parts.append(f'<li><a href="{rel(path)}">{title}</a></li>')
        if name=='R01':parts.append(f'<li><a href="{rel(R/"reviews/moving61_generation.json")}">已确认的遮挡失败及证据</a></li>')
        if name=='R07':
            for title,path in [('61帧四联屏',R/'demos/r07_short61_24/four_panel.mp4'),('固定视角动作',R/'demos/r07_short61_24/fixed_views.mp4'),('动态高斯PLY',R/'reconstruction/outputs/r07_short61_24/ckpt_4999.ftgs.ply'),('生成/高斯脚与包的细节差异',R/'demos/r07_short61_24/details_view06_030.jpg'),('技术指标及视觉问题',R/'reviews/r07_short61_24_asset.json')]:
                assert path.is_file()
                parts.append(f'<li><a href="{rel(path)}">{title}</a></li>')
        if name=='M02':
            for title,path in [('61帧四联屏（包含失败）',R/'demos/m02_short61_24/four_panel.mp4'),('动态高斯PLY',R/'reconstruction/outputs/m02_short61_24/ckpt_4999.ftgs.ply'),('技术指标与视觉失败',R/'reviews/m02_short61_24_asset.json')]:
                assert path.is_file()
                parts.append(f'<li><a href="{rel(path)}">{title}</a></li>')
        parts.append('</ul></section>');cases.append(case)
    stamp=datetime.now(timezone.utc).isoformat()
    parts.append(f'<p>文件快照更新：{stamp}。技术解码通过与视觉通过分开记录；没有计算真实新视角指标。</p>')
    (OUT/'index.html').write_text('\n'.join(parts))
    (OUT/'manifest.json').write_text(json.dumps({'updated_at':stamp,'cases':cases,'production_approved':False},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'state':'local_page_written','cases':len(cases),'path':str(OUT/'index.html')}))

if __name__=='__main__':main()
