from pathlib import Path
import json,csv,sys,hashlib
R=Path(__file__).resolve().parents[1];D=R/'docs';variants=['H','R','B0']
fid=json.loads((D/'random10_fid_metrics.json').read_text());fids={r['variant']:r['FID_2048'] for r in fid['results']}
vm=json.loads((D/'random10_metrics.json').read_text());pm=json.loads((D/'random10_pixel_metrics.json').read_text());nm=json.loads((R/'nonvehicle10/docs/metrics.json').read_text());rows=[]
for v in variants:
 m=next(s for s in vm['summary'] if s['variant']==v);p=next(s for s in pm['summary'] if s['variant']==v and s['alignment']=='normalized')
 rows.append(dict(cohort='vehicles10',variant=v,N=10,AOR_percent=m['AOR_percent'],FID_2048=fids[v],CLIP_I16=m['CLIP_I16']['mean'],DINO_I16=m['DINO_I16']['mean'],ED_R=m['ED_R']['mean'],PSNR_dB=p['PSNR_dB']['mean'],SSIM=p['SSIM']['mean'],LPIPS=p['LPIPS_alex']['mean'],pixel_protocol='fixed source view, independent foreground centering/isotropic scale normalization',FID_reference_images=10,FID_rendered_images=160))
for v in variants:
 m=next(s for s in nm['summary'] if s['variant']==v)
 rows.append(dict(cohort='other_objects10',variant=v,N=10,AOR_percent=100*m['AOR'],FID_2048=m['FID_2048'],CLIP_I16=m['CLIP_I16']['mean'],DINO_I16=m['DINO_I16']['mean'],ED_R=None,PSNR_dB=m['PSNR_I16']['mean'],SSIM=m['SSIM_I16']['mean'],LPIPS=m['LPIPS_I16']['mean'],pixel_protocol='source foreground versus16 orbit views, average within identity then across identities',FID_reference_images=10,FID_rendered_images=160))
for name,data in [('random10_H_R_B0_summary',rows[:3]),('assets20_H_R_B0_metrics',rows)]:
 with (D/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(D/'assets20_H_R_B0_metrics.json').write_text(json.dumps(dict(rows=rows,vehicle_fid_protocol='random10_fid_protocol.json',vehicle_fid_numerical_audit=fid['numerical_audit'],nonvehicle_fid_protocol='../nonvehicle10/docs/fid_protocol.json',note='Existing appearance and pixel values unchanged; vehicle FID newly computed. No new asset generation.'),indent=2))
def table(group,vehicle):
 headers=['路线','AOR ↑','FID ↓','CLIP-I16 ↑','DINO-I16 ↑','ED-R ↓','PSNR'+('' if vehicle else '-I16')+' ↑','SSIM'+('' if vehicle else '-I16')+' ↑','LPIPS'+('' if vehicle else '-I16')+' ↓']
 body=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
 for r in group:body.append('| '+' | '.join([r['variant'],f"{r['AOR_percent']:.0f}%",f"{r['FID_2048']:.2f}",f"{r['CLIP_I16']:.4f}",f"{r['DINO_I16']:.4f}",f"{r['ED_R']:.5f}" if r['ED_R'] is not None else 'N/A',*[f"{r[k]:.4f}" for k in ['PSNR_dB','SSIM','LPIPS']]])+' |')
 return '\n'.join(body)
report='''# 20个对象：H / R / B0 指标对比（2026-10-23）

车辆FID已补测；其余既有指标保持原值。两批各10个对象，都是seed42。只列本次资产包中的H、R、B0。

## 10个车辆

'''+table(rows[:3],True)+'''

PSNR/SSIM/LPIPS采用固定源视角、前景居中与等比尺度归一化，属于训练源外观诊断，不是held-out评估。这三项R均值最好；H的FID、CLIP-I16、DINO-I16、ED-R均值最好。只是本批均值排序，不代表总体显著优势。

## 10个其他物体

'''+table(rows[3:],False)+'''

此表像素指标是源图与16个环绕图的平均诊断，与车辆表固定源视角的口径不同，不能跨表横比。ED-R缺可靠源视角对齐，记为N/A。R的FID/CLIP/SSIM/LPIPS均值最好；H的DINO/PSNR均值最好。

## FID实际口径

两批均用pytorch-fid 0.3.0、FID兼容Inception v3 pool3 2048维特征，float32、关闭TF32。每对象固定一张真实源前景，10张参考图；每路线16视角×10对象=160张渲染前景。与既有I16共用前景处理：alpha>127、紧框、每边5%留白、白底512，然后Inception双线性缩到299并映射[-1,1]，特征不做L2归一化。

分别计算两组特征的均值与无偏协方差（ddof=1），FID=均值差平方+两协方差迹−2倍协方差乘积平方根的迹。实际用float64低秩SVD等价公式，并在实际2048维数据上与上游SciPy实现交叉核验。它是整个集合的分布分数，不是单对象FID的平均。

仅10个真实身份，参考协方差秩至多9；160张环绕图互相关联，存在强小样本偏差和源视角/环绕视角分布差异。因此只作每批相同设置下的探索性比较。车辆与其他物体的FID不直接横比；不据此认定几何、文字、灯态、接地或插入效果最好。没有计算插入场景FID，也没有重新生成PLY。

- [车辆FID原始结果](random10_fid_metrics.json) / [协议与输入哈希](random10_fid_protocol.json) / [FID CSV](random10_fid_summary.csv)
- [车辆完整指标CSV](random10_H_R_B0_summary.csv) / [两批对比CSV](assets20_H_R_B0_metrics.csv)
- [非车辆协议](../nonvehicle10/docs/fid_protocol.json)
- [pytorch-fid官方实现](https://github.com/mseitzer/pytorch-fid)
'''
(D/'assets20_H_R_B0_metrics.md').write_text(report)
sys.path.insert(0,str(R/'tools/report_deps'));import markdown
(D/'assets20_H_R_B0_metrics.html').write_text('<!doctype html><meta charset="utf-8"><title>H/R/B0指标对比</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:auto;padding:30px;line-height:1.7}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:9px}th{background:#edf3f9}</style>'+markdown.markdown(report,extensions=['tables']))
# Put current summary ahead of historical results, without rewriting those results.
section='''## 车辆FID已补测（2026-10-23）

[H/R/B0车辆与其他物体完整对比表](docs/assets20_H_R_B0_metrics.html) · [车辆CSV](docs/random10_H_R_B0_summary.csv) · [两批CSV](docs/assets20_H_R_B0_metrics.csv)。车辆FID：H **87.25**、R **95.01**、B0 **95.98**；每路线10个对象、10张真实参考与160张环绕图。使用与非车辆相同的Inception权重、预处理、协方差定义，完成数值交叉核验。仅作小样本探索性比较；不与非车辆的数值跨类别横比，不替代插入效果验证。原CLIP/DINO/ED-R/像素指标未改动。

'''
for name in ['REPORT.md','DEMOS.md']:
 p=R/name;s=p.read_text()
 if '## 车辆FID已补测（2026-10-23）' not in s:
  head,body=s.split('\n',1);p.write_text(head+'\n\n'+section+body.lstrip('\n'))
print(json.dumps(rows[:3],indent=2))
