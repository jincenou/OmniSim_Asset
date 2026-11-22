"""Contribution-weighted token footprints, proxy compatibility, no parameter updates."""
from diagnostic import *
from PIL import ImageDraw

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);a=ap.parse_args();key=a.key;label=key[:3];assert key.endswith(('R2','R3'));device=verify();torch.set_num_threads(2);cv2.setNumThreads(1);plan=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text());sel=plan['selections'][label];cache=D/'cache'/key;out=D/'results'/key;media=D/'comparison'/key;media.mkdir(parents=True,exist_ok=True)
 while not (out/'complete.json').exists():
  status=json.loads((D/'STATUS.json').read_text());assert status.get('jobs',{}).get(key,{}).get('state')!='failed';time.sleep(10)
 with np.load(cache/'static.npz') as z:ids=z['binding_ids'];bw=z['binding_weights'];L=float(z['L'])
 rows=[];shared=[];textures=[];checks=[];samples=[sel['window']['start'],(sel['window']['start']+sel['window']['end'])//2,sel['window']['end'],sel['frames']-1]
 proxies={}
 for v in [1,13,7,19]:
  with np.load(D/'data'/label/f'support{v:02d}.npz') as z:proxies[v]={int(f):lab for f,lab in zip(z['frames'],z['labels'])}
 for f in sel['support_times']:
  with np.load(cache/f'state_{f:03d}.npz') as z:s={k:z[k] for k in z.files}
  # Exact effective footprint: avg-pool source indicators then same bilinear grid sampler.
  footprint=[]
  for level,scale in enumerate([2,4]):
   maps=[]
   for v in [1,13,7,19]:
    target,ra=reference(label,v,f);lab=proxies[v][f];classes=np.stack([(ra<=.5)&(lab==0),(ra>.5)&(lab==0)]+[lab==i for i in range(1,7)]).astype('float32');maps.append(classes)
   pooled=torch.nn.functional.avg_pool2d(torch.as_tensor(np.stack(maps),device='cuda'),scale);grid=torch.as_tensor(s['grids'][:,:,level],device='cuda');vals=torch.nn.functional.grid_sample(pooled,grid,align_corners=False).permute(0,2,3,1);footprint.append(cpu(vals))
  # shape view,anchor,scale,sample,class, class0background,class1unknown,2..7body
  footprint=np.stack(footprint,2);tokenweights=s['attention'].mean(1)[:,:32].reshape(128,4,2,4).transpose(1,0,2,3)
  uv_tokens=(s['grids']+1)*np.array([352,640])/2-.5
  for targetview,tf in sel['observations']:
   if tf!=f:continue
   with np.load(cache/f'weights{targetview:02d}_{f:03d}.npz') as z:allw=z['weights'];axes=z['axes_sigma']
   # Shared anchor loads use distinct exact target-view alpha contributions.
   node=np.zeros((8,128),np.float64)
   for r in range(8):np.add.at(node[r],ids.ravel(),(allw[r,:,None]*bw).ravel())
   for ri,targetpart in [(0,5 if label=='N06' else 3),(1,6 if label=='N06' else 4)]:
    w=allw[ri];total=float(w.sum());nw=node[ri]
    if total<=0:continue
    torso=node[3] # region index 2+part torso index1
    share=float(np.minimum(nw/max(nw.sum(),1e-12),torso/max(torso.sum(),1e-12)).sum())
    shared.append(dict(key=key,frame=f,view=targetview,roi=f'expanded_{ri}',node_distribution_overlap_with_torso=share,target_contribution=total,torso_contribution=float(torso.sum()),note='2D contribution distributions, not material or semantic truth'))
    for vi,sv in enumerate([1,13,7,19]):
     own=s[f'support_uv_final_{sv}'];base=s[f'support_uv_base_{sv}'];depth=s[f'support_z_final_{sv}'];px=np.rint(own[:,0]).astype(int);py=np.rint(own[:,1]).astype(int);inside=(px>=0)&(px<352)&(py>=0)&(py<640)&(depth>0);cx=px.clip(0,351);cy=py.clip(0,639);lab=proxies[sv][f];ownlabel=lab[cy,cx];sa=s[f'support_alpha_{sv}'];ds=s[f'support_depthsum_{sv}'];expected=ds[cy,cx]/np.maximum(sa[cy,cx],1e-8);depth_ok=abs(depth-expected)<=.01*L
     # Label agreement and composited-depth proximity are necessary proxies, not proof of visibility.
     trusted=inside&(sa[cy,cx]>.5)&depth_ok&(ownlabel==targetpart)
     trustednw=np.zeros(128);np.add.at(trustednw,ids.ravel(),(w[:,None]*bw*trusted[:,None]).ravel())
     aweights=tokenweights[vi];mass=nw[:,None,None]*aweights;tmass=trustednw[:,None,None]*aweights;fp=footprint[vi];valid=s['valid'][:,:32].reshape(128,4,2,4)[:,vi];assert np.isfinite(fp).all() and fp.min()>=-1e-6 and fp.sum(-1).max()<=1+1e-5
     classification=(tmass[...,None]*fp).sum((0,1,2));allmass=float(mass.sum());trustedmass=float(tmass.sum());null=float((nw*s['attention'].mean(1)[:,-1]).sum())
     # Weighted distances evaluated for each contributing Gaussian and its four anchors.
     dsums=bsums=den=0.;dvalues=[];dweights=[]
     for st in range(0,len(w),2048):
      ix=ids[st:st+2048];ww=w[st:st+2048,None,None,None]*bw[st:st+2048,:,None,None]*aweights[ix];centers=uv_tokens[vi][ix];dist=np.linalg.norm(centers-own[st:st+2048,None,None,None,:],axis=-1);bd=np.linalg.norm(centers-base[st:st+2048,None,None,None,:],axis=-1);dsums+=float((ww*dist).sum());bsums+=float((ww*bd).sum());den+=float(ww.sum());dvalues.append(dist.ravel());dweights.append(ww.ravel())
     stats=weighted(np.concatenate(dvalues),np.concatenate(dweights));row=dict(key=key,frame=f,target_view=targetview,roi=f'expanded_{ri}',support_view=sv,target_part=PARTS[targetpart-1],actual_roi_contribution=total,token_mass=allmass,compatible_mass=float(classification[targetpart+1]),torso_mass=float(classification[3]),wrong_side_mass=float(classification[(6 if targetpart==5 else 5 if targetpart==6 else 4 if targetpart==3 else 3)+1]),background_mass=float(classification[0]),other_part_mass=float(classification[2:].sum()-classification[targetpart+1]),unknown_mass=max(0.,allmass-float(classification[[0]+list(range(2,8))].sum())),visibility_proxy_trusted_contribution=float(w[trusted].sum()),visibility_proxy_unknown_contribution=float(w[~trusted].sum()),null_mass_all_views_once=null,final_read_distance_mean=dsums/den if den else None,base_read_distance_mean=bsums/den if den else None,final_read_distance_p50=stats['p50'],final_read_distance_p90=stats['p90'],semantic_truth=False)
     rows.append(row)
     if ri==0 and targetview==sel['window']['view'] and f in samples[:3]:
      target,ra=reference(label,sv,f);im=Image.fromarray(np.uint8(np.clip(target,0,1)*255));draw=ImageDraw.Draw(im)
      for part in range(1,7):
       mask=(lab==part).astype('uint8');contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       for c in contours:
        xy=[tuple(p) for p in c[:,0]]
        if len(xy)>1:draw.line(xy,fill=['orange','cyan','magenta','purple','green','yellow'][part-1],width=1)
      gi=np.argsort(w)[-8:];anchors=np.unique(ids[gi]);top=anchors[np.argsort(nw[anchors])[-4:]]
      for i in gi:
       u,v=own[i];bu,bv=base[i]
       if np.isfinite([u,v,bu,bv]).all():draw.line((bu,bv,u,v),fill='white',width=2);draw.ellipse((u-2,v-2,u+2,v+2),fill='red')
      for m in top:
       u,v=(s['anchor_ndc'][vi,m]+1)*np.array([352,640])/2-.5;draw.ellipse((u-3,v-3,u+3,v+3),outline='blue',width=2)
       for lv,scale in enumerate([2,4]):
        for k in range(4):
         u,v=uv_tokens[vi,m,lv,k]
         # Bounding footprint of two feature pixels each axis; includes average pooling.
         if np.isfinite([u,v]).all() and abs(u)<1e5 and abs(v)<1e5:draw.rectangle((u-scale,v-scale,u+scale,v+scale),outline='yellow' if lv==0 else 'magenta',width=1)
      draw.text((4,4),f'{key} f{f} support{sv} RED Gaussian BLUE anchor',fill='white');im.save(media/f'read_f{f:03d}_s{sv:02d}.png')
      # Same support camera/time only. Crops use frozen reference target-part bounds, no output selection.
      yy,xx=np.where(lab==targetpart)
      if len(xx)>0:
       x0=max(0,int(xx.min())-16);x1=min(352,int(xx.max())+17);y0=max(0,int(yy.min())-16);y1=min(640,int(yy.max())+17);crop=target[y0:y1,x0:x1];half=cv2.resize(cv2.resize(target,(176,320),interpolation=cv2.INTER_AREA),(352,640),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1];quarter=cv2.resize(cv2.resize(target,(88,160),interpolation=cv2.INTER_AREA),(352,640),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1];pred=s[f'support_rgb_{sv}'][y0:y1,x0:x1];imgs=[crop,half,quarter,np.clip(pred,0,1)];h,wc=crop.shape[:2];sheet=Image.new('RGB',(wc*4,h+22),'#333333');dr=ImageDraw.Draw(sheet)
       for j,(name,x) in enumerate(zip(['REF352','half176','quarter88','M11'],imgs)):sheet.paste(Image.fromarray(np.uint8(x*255)),(j*wc,22));dr.text((j*wc+3,3),name,fill='white')
       sheet.save(media/f'texture_f{f:03d}_s{sv:02d}.png');gray=cv2.cvtColor(crop.astype('float32'),cv2.COLOR_RGB2GRAY);freq=cv2.Laplacian(gray,cv2.CV_32F)
       textures.append(dict(key=key,frame=f,support_view=sv,roi=f'expanded_{ri}',reference_proxy_crop=[x0,y0,x1,y1],reference_laplacian_rms=float(np.sqrt((freq*freq).mean())),crop_pixels=h*wc,note='Descriptive image variation including edges; not proof of recoverable garment frequency'))
  print(key,'locality',f,flush=True)
 csvwrite(out/'locality.csv',rows);csvwrite(out/'node_sharing.csv',shared);csvwrite(out/'texture.csv',textures);save(out/'locality_complete.json',dict(state='complete',device=device,optimizer_calls=0,rows=len(rows),classes='conservative joint own-part/depth proxy; unsupported portions unknown',attention_weights='mean over two heads; noncausal effective value-path weights',bilinear_pool_footprints=True,occlusion_truth=False));print(key,'LOCALITY COMPLETE',flush=True)
if __name__=='__main__':main()
