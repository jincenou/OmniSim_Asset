import os,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(R/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=R/'deliveries/vehicle_experiments_20261023/images/overview';d=json.loads((R/'docs/random10_pixel_metrics.json').read_text());vs=['B0','R','F','H','B0_real1','B0_real4'];labels=['B0','R','F','H','B0+1 real','B0+4 real'];colors=['#94a3b8','#15803d','#c084fc','#2563eb','#f59e0b','#ea580c']
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for row,mode in enumerate(['normalized','native']):
 for col,(k,title) in enumerate([('PSNR_dB','PSNR (dB), higher is better'),('SSIM','SSIM, higher is better'),('LPIPS_alex','LPIPS (Alex), lower is better')]):
  ax=axes[row,col];s=[next(s for s in d['summary'] if s['variant']==v and s['alignment']==mode)[k] for v in vs];means=np.array([x['mean'] for x in s]);ci=np.array([x['bootstrap95'] for x in s]);ax.bar(range(6),means,color=colors,width=.65);ax.errorbar(range(6),means,yerr=[means-ci[:,0],ci[:,1]-means],fmt='none',ecolor='#334155',capsize=3,lw=1)
  ax.set_xticks(range(6),labels,rotation=30,ha='right');ax.set_title(('Foreground-normalized' if mode=='normalized' else 'Camera-projection preserved')+'\n'+title,fontsize=11);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  for i,y in enumerate(means):ax.text(i,y*.45,f'{y:.2f}' if k=='PSNR_dB' else f'{y:.3f}',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
fig.suptitle('Matched-source diagnostics | 10 identities, six routes\nMean and identity-bootstrap 95% CI; fixed training source, not held-out reconstruction',fontsize=14)
fig.savefig(O/'pixel_metrics.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(14,6));ax.axis('off');ax.set_xlim(0,14);ax.set_ylim(0,6)
def box(x,y,w,h,text,color):
 from matplotlib.patches import FancyBboxPatch
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#cbd5e1'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#64748b',lw=1.5))
box(.2,2.1,2.2,1.4,'Waymo inputs\n10 fixed identities\n4 train + 2 held out','#f1f5f9');box(3,2.1,2,1.4,'SAM2 foregrounds\n4 real views\n1 fixed source','#e0f2fe');arrow((2.45,2.8),(2.85,2.8))
box(5.6,4.3,2.2,1,'Original FLUX.2 Klein\nvisible end + side\nverbatim prompts','#ede9fe');arrow((4.7,3.6),(5.6,4.5))
box(8.5,4.3,2.5,1,'Original TRELLIS\nB0 / B0+1 / B0+4\nequal-weight multidiffusion','#fff7ed');arrow((7.95,4.8),(8.35,4.8));arrow((5.1,3.1),(8.35,4.4))
box(8.5,1.6,2.5,1.5,'ReconViaGen v0.2\nR: 4 real\nF: 2 FLUX\nH: 4 real + 2 FLUX','#dbeafe');arrow((5.15,2.5),(8.35,2.5));arrow((7.8,4.3),(8.65,3.25))
box(11.65,2.3,2.0,1.6,'Gaussian PLY\ncanonicalize + reload\n16 orbit views\nfixed source view','#dcfce7');arrow((11.15,4.7),(11.9,4.05));arrow((11.15,2.6),(11.5,2.8))
ax.text(.3,.7,'Actual pilot: native labels/3D track metadata assist selection, scale and source camera.\nGroundingDINO/Qwen stages were not rerun. No held-out images enter generation. Main runs use seed 42.',fontsize=11,color='#475569');ax.set_title('Actual experiment pipeline and route differences',loc='left',fontsize=18,pad=20)
fig.savefig(O/'pipeline.png',dpi=150,bbox_inches='tight');plt.close(fig)
print('PLOTS COMPLETE')
