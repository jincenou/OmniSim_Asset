from media import *

def main():
 plan=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text());records=[]
 for key in ['N06_R2','N06_R3','N11_R2','N11_R3']:
  label=key[:3];cache=D/'cache'/key;win=plan['selections'][label]['window'];v=win['view']
  with np.load(cache/'static.npz') as z:ids=z['binding_ids'];bw=z['binding_weights']
  for f in [win['start'],(win['start']+win['end'])//2,win['end']]:
   with np.load(cache/f'state_{f:03d}.npz') as z:s={k:z[k] for k in z.files}
   with np.load(cache/f'weights{v:02d}_{f:03d}.npz') as z:w=z['weights'][0]
   nw=np.zeros(128);np.add.at(nw,ids.ravel(),(w[:,None]*bw).ravel());gi=np.argsort(w)[-8:];anchors=np.unique(ids[gi]);top=anchors[np.argsort(nw[anchors])[-4:]]
   for vi,sv in enumerate([1,13,7,19]):
    with np.load(D/'data'/label/f'support{sv:02d}.npz') as z:lab=z['labels'][list(z['frames']).index(f)]
    target,ra=ref(label,sv,f);im=Image.fromarray(np.uint8(np.clip(target,0,1)*255));dr=ImageDraw.Draw(im)
    for p,col in enumerate(['orange','cyan','magenta','purple','green','yellow'],1):
     contours,_=cv2.findContours((lab==p).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     for c in contours:
      xy=[tuple(x) for x in c[:,0]]
      if len(xy)>1:dr.line(xy,fill=col,width=1)
    base=s[f'support_uv_base_{sv}'];own=s[f'support_uv_final_{sv}']
    for i in gi:
     u,y=own[i];bu,by=base[i]
     if np.isfinite([u,y,bu,by]).all():dr.line((bu,by,u,y),fill='white',width=2);dr.ellipse((u-2,y-2,u+2,y+2),fill='red')
    for m in top:
     u,y=(s['anchor_ndc'][vi,m]+1)*[176,320]-.5;dr.ellipse((u-3,y-3,u+3,y+3),outline='blue',width=2)
     for level,scale in enumerate([2,4]):
      for k in range(4):
       token=vi*8+level*4+k
       if not s['valid'][m,token]:continue
       grid=s['grids'][vi,m,level,k];feat=(grid+1)*np.array([352/scale,640/scale])/2-.5;lo=np.floor(feat)*scale-.5;hi=(np.floor(feat)+2)*scale-.5
       if np.isfinite([*lo,*hi]).all():
        dr.rectangle((*lo,*hi),outline='yellow' if level==0 else 'magenta',width=1)
        records.append(dict(key=key,frame=f,support_view=sv,anchor=int(m),scale=scale,sample=k,rgb_alpha_footprint_xyxy=[*lo.tolist(),*hi.tolist()],sobel_receptive_bound_xyxy=[*(lo-scale).tolist(),*(hi+scale).tolist()],note='RGB/alpha exact 2x2 bilinear feature cells each avg-pools scale x scale; Sobel extends one feature pixel, boundary replication/zero padding apply'))
    dr.text((4,4),f'{key} f{f} s{sv} red=Gaussian blue=anchor',fill='white');im.save(D/'comparison'/key/f'read_f{f:03d}_s{sv:02d}.png')
 (D/'research/footprint_bounds.json').write_text(json.dumps(records,indent=2));print('exact plotted footprints',len(records))
if __name__=='__main__':main()
