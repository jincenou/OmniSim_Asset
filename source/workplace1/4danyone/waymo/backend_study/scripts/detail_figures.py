"""Scientific comparison crops, same pixel regions for all methods; no enhancement."""
import hashlib,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_queue import ROOT,save,stamp
BASE=ROOT.parent/'random8/r20261015_113332'

def main():
    records=[]
    for label,view in [('N04',0),('N08',8)]:
        data=json.loads((BASE/'four_dg/data'/label/'fdanyone.json').read_text())
        result=ROOT/'results'/label
        assert (result/'results.json').exists()
        fig,axes=plt.subplots(6,4,figsize=(14,12),layout='constrained')
        for idx,frame in enumerate([0,(data['frame_count']-1)//2,data['frame_count']-1]):
            source=result/f'view{view:02d}_frame{frame:03d}.jpg'
            canvas=np.asarray(Image.open(source))
            rgba=__import__('pathlib').Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            mask=np.asarray(Image.open(rgba))[...,3]>127
            yy,xx=np.where(mask);y0,y1=int(yy.min()),int(yy.max())+1;h=y1-y0
            x0,x1=max(0,int(xx.min())-20),min(352,int(xx.max())+21)
            boxes=[(x0,max(0,y0+int(.18*h)),x1,min(640,y0+int(.68*h))),
                   (x0,max(0,y0+int(.65*h)),x1,min(640,y1+20))]
            for part,box in enumerate(boxes):
                a,b,c,d=box
                for method in range(4):
                    ax=axes[idx*2+part,method]
                    ax.imshow(canvas[b+32:d+32,a+352*method:c+352*method],interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    if method==0:ax.set_ylabel(f'f={frame}\n'+('torso / arms region' if part==0 else 'lower / feet region'))
                records.append({'label':label,'view':view,'frame':frame,'region':part,'box_xyxy':box,
                    'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        for ax,title in zip(axes[0],['Generated reference','HUST grid61','CEM stage1','CEM complete']):ax.set_title(title)
        fig.suptitle(f'{label} | heldout view {view:02d} | first / middle / last | same crops, no enhancement')
        fig.savefig(result/'details.png',dpi=140);plt.close(fig)
    save(ROOT/'research/detail_figures.json',dict(updated_at=stamp(),state='complete',crops=records,
        note='Crops from completed JPEG comparison renders. Fixed per-frame reference-bbox fractions, not semantic body-part annotations. Plot scaling does not add information. Metrics use raw renders, not these figures.'))

if __name__=='__main__':main()
