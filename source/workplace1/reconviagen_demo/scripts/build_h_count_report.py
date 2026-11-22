"""Local report with all ten identities, fixed-source comparisons and orbit videos."""
from pathlib import Path
import html,json
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from h_count_ablation import ROOT,WORK
from evaluate_asset_metrics import foreground

def main():
    import imageio.v2 as imageio
    metrics=json.loads((WORK/'metrics.json').read_text());summary=metrics['summary']
    registry=json.loads((WORK/'registry.json').read_text());cases=list(dict.fromkeys(t['case_id'] for t in registry['tasks']))
    media=WORK/'media';media.mkdir(exist_ok=True)
    font=ImageFont.load_default(size=20)
    small=ImageFont.load_default(size=15)
    sections=[]
    for cid in cases:
        cam=json.loads((ROOT/'inputs'/cid/'evaluation/camera.json').read_text())
        source=foreground(cam['source_rgb'],cam['source_mask'])[0]
        panels=[source];outdirs=[]
        for n in range(1,5):
            v='H' if n==4 else f'H_real{n}';out=ROOT/'results'/cid/f'{v}_seed42'/'evaluation';outdirs.append(out)
            panels.append(foreground(out/'matched_source.png',out/'matched_source_alpha.png')[0])
        sheet=Image.new('RGB',(2000,500),'white');draw=ImageDraw.Draw(sheet)
        for i,im in enumerate(panels):
            sheet.paste(im.resize((400,400)),(400*i,40));draw.text((400*i+12,10),'Real source' if i==0 else f'H + {i} real + 2 FLUX',font=font,fill='black')
            if i:
                row=next(r for r in metrics['rows'] if r['case_id']==cid and r['real_count']==i)
                draw.text((400*i+12,450),f"DINO-I16 {row['DINO_I16']:.4f} | ED-R {row['ED_R']:.4f}",font=small,fill='black')
                draw.text((400*i+12,475),f"PSNR {row['PSNR']:.2f} | LPIPS {row['LPIPS']:.4f}",font=small,fill='black')
        sheet.save(media/f'{cid}_matched.jpg',quality=94)
        Image.open(cam['source_rgb']).save(media/f'{cid}_source.png')
        # Fixed orbit views shared by all counts; no best-view selection.
        frames=[]
        for j in range(16):
            frame=Image.new('RGB',(1600,350),'white');d=ImageDraw.Draw(frame)
            for i,out in enumerate(outdirs):
                im=foreground(out/f'orbit_{j:02}.png',out/f'orbit_{j:02}_alpha.png')[0]
                frame.paste(im.resize((320,320)),((i+1)*320,30));d.text(((i+1)*320+10,8),f'H + {i+1} real | view {j:02}',font=small,fill='black')
            frame.paste(source.resize((320,320)),(0,30));d.text((10,8),'Real source',font=small,fill='black')
            frames.append(np.array(frame))
        imageio.mimwrite(media/f'{cid}_orbit.mp4',frames,fps=4,codec='libx264',quality=8,macro_block_size=2)
        # Relative 180 degree evidence, not a source-image ground-truth comparison.
        Image.fromarray(frames[8]).save(media/f'{cid}_opposite.jpg',quality=94)
        selected=next(t['selection_order'] for t in registry['tasks'] if t['case_id']==cid)
        sections.append(f'<article><h2>{html.escape(cid)}</h2><p>逐步加入的真实观测：{html.escape(" → ".join(selected))}。模型输入保留原清单顺序。</p><a href="media/{cid}_source.png">原始评估源图</a><img loading="lazy" src="media/{cid}_matched.jpg" alt="源图与四种数量的匹配源视角对比"><details><summary>查看共同相对 180° 视图</summary><img loading="lazy" src="media/{cid}_opposite.jpg"></details><video controls loop preload="none" src="media/{cid}_orbit.mp4"></video></article>')
    cols=[('real_count','真实图数'),('AOR','AOR % ↑'),('CLIP_I16','CLIP-I16 ↑'),('DINO_I16','DINO-I16 ↑'),('ED_R','ED-R ↓'),('PSNR','PSNR ↑'),('SSIM','SSIM ↑'),('LPIPS','LPIPS ↓'),('FID','FID ↓')]
    header=''.join(f'<th>{label}</th>' for _,label in cols)
    body=''.join('<tr>'+''.join(f'<td>{r[k]:.4f}</td>' if k not in ('real_count','AOR') else f'<td>{r[k]:g}</td>' for k,_ in cols)+'</tr>' for r in summary)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>H 路线：真实参考图数量实验</title><style>body{{font:16px/1.7 system-ui,sans-serif;margin:auto;max-width:1500px;padding:28px;color:#172536;background:#f5f7fa}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border:1px solid #d7dee6;text-align:right}}article,section{{background:white;padding:24px;margin:24px 0;border-radius:10px}}img,video{{display:block;width:100%;height:auto;margin:18px 0}}a{{color:#1263b4}}.scroll{{overflow:auto}}</style>
    <h1>H 路线：1／2／3／4 张真实图对比</h1><p>10 个固定车辆身份 × 4 组；每组额外固定两张 FLUX 图。seed=42；结构与 latent 各 25 步。四图组复用原 H，其余 30 组重新生成。</p>
    <section><h2>实验结论与建议</h2><p><strong>建议保持 3–4 张为目标，2 张作为有效观测不足时的回退，1 张标记为受限输入并加强质量检查。</strong>1→2 张的 DINO-I16 提升约 0.0447、FID 下降约 3.172；2→3 张 DINO-I16 再提升约 0.0079、LPIPS 下降约 0.0066；3→4 张 FID 仅下降约 0.157、DINO-I16 提升约 0.0083，而 PSNR 和 LPIPS 的均值略偏向三张。三张对四张的 DINO、ED-R、PSNR、LPIPS 配对 bootstrap 95% 区间均跨零，十个对象不足以证明四张普遍更优。</p><p>第四张应提供清晰、未明显遮挡、且有补充信息的观测；不能为了凑四张降低质量。两张是可用回退，不应直接等同于三至四张的平均质量；单张的 DINO-I16 和 LPIPS 相对四张明显偏弱。个别模型会出现跨视角幻觉，例如灰色 MPV 的三图和四图结果尾部出现车头式格栅，参考图更多并不保证消除伪影。</p><p>前三组推理均值约 18.64／20.55／21.36 秒，旧四图组约 15.19 秒；GPU 并行负载不同，不能据此宣称四图更快。推理峰值分配显存约 6.89／7.20／7.50／7.78 GiB。首次预览因 ninja 路径失败、首次特征评估因 Python 环境选择错误失败，修正环境后补跑成功；30 次重建均已成功，最终 40 组无缺失对象。</p><p>前端已有自适应真实图与独立路线清单的待验证修改，原 FLUX 接口不变。nuScenes 端到端生成、跨相机去重和统一插入验证仍待后续完成；按用户要求，先查看本报告并确定最终策略，再启动后续场景验证。</p></section><section><h2>汇总指标</h2><div class="scroll"><table><tr>{header}</tr>{body}</table></div><p><a href="summary.csv">均值 CSV</a> · <a href="per_asset.csv">逐对象 CSV</a> · <a href="metrics.json">完整指标与配对 bootstrap 区间</a> · <a href="registry.json">冻结输入</a> · <a href="fid_protocol.json">FID 输入与协议</a></p>
    <p>所有 40 组都必须通过 PLY 加载和 16 视图非空检查；AOR 不等于视觉质量合格率。CLIP/DINO 先在对象内平均 16 个视图，再对十个身份平均。ED-R 为固定源视角、前景 patch 均值的 DINOv3 距离；没有事后挑选最佳视角。PSNR/SSIM/LPIPS 使用固定源视角的独立前景居中、等比缩放、白底 512×512，包含白背景；原画布与前景并集敏感性指标另有保留。</p>
    <p>FID 使用官方 Inception-v3 2048 维特征，10 张真实源图与每组 160 张轨道渲染，整组计算，越低越好。十个身份的小样本结果有明显统计局限，不应把很小的差值当作稳定优劣。所有指标都是训练源图条件下的诊断，不是独立新视角几何真值，也不是插入场景效果分数。</p>
    <p>下方静态图与视频为了比较外观，均做独立前景居中缩放；不用于判断绝对尺度。180° 图没有对应真实背面真值，只展示生成完整性和合理性。数据来自已有 Waymo 十车试验，不代表已完成 nuScenes 全流程验证。</p></section>
    {''.join(sections)}</html>'''
    (WORK/'REPORT.html').write_text(report)
    print(WORK/'REPORT.html')

if __name__=='__main__':main()
