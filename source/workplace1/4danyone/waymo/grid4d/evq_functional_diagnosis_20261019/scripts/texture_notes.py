from media import *
from summarize import load
# Descriptive spectra only; no threshold, pass/fail, or method selection.
def spectral(gray):
 h,w=gray.shape;win=np.outer(np.hanning(h),np.hanning(w));a=(gray-gray.mean())*win;p=abs(np.fft.rfft2(a))**2;fy=np.fft.fftfreq(h)[:,None];fx=np.fft.rfftfreq(w)[None];f=np.sqrt(fy**2+fx**2);valid=f>0;freq=f[valid];power=p[valid];order=np.argsort(freq);c=np.cumsum(power[order]);median=float(np.interp(.5*c[-1],c,freq[order])) if c[-1]>0 else None
 return dict(median_power_frequency_cycles_per_pixel=median,median_power_wavelength_px=1/median if median else None,scope='mean removed Hann-windowed 2D luminance spectrum,includes shading/boundary;not semantic garment-frequency truth')
def main():
 rows=[]
 for key in ['N06_R2','N06_R3','N11_R2','N11_R3']:
  label=key[:3];sel=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text())['selections'][label];win=sel['window'];fs=[win['start'],(win['start']+win['end'])//2,win['end']]
  for f in fs:
   with np.load(D/'cache'/key/f'state_{f:03d}.npz') as z:s={k:z[k] for k in z.files}
   for v in [1,13,7,19]:
    with np.load(D/'data'/label/f'support{v:02d}.npz') as z:lab=z['labels'][list(z['frames']).index(f)]
    for part,pn in [(2,'torso'),(5 if label=='N06' else 3,'target')]:
     yy,xx=np.where(lab==part)
     if len(xx)==0:rows.append(dict(key=key,frame=f,view=v,part=pn,status='unknown_no_core'));continue
     x0=max(0,int(xx.min())-8);x1=min(352,int(xx.max())+9);y0=max(0,int(yy.min())-8);y1=min(640,int(yy.max())+9);target,ra=ref(label,v,f);crop=target[y0:y1,x0:x1];pred=s[f'support_rgb_{v}'][y0:y1,x0:x1];h,w=crop.shape[:2];hal=cv2.resize(cv2.resize(target,(176,320),interpolation=cv2.INTER_AREA),(352,640),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1];qua=cv2.resize(cv2.resize(target,(88,160),interpolation=cv2.INTER_AREA),(352,640),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1]
     im=canvas([crop,hal,qua,pred],['REF352','half176','quarter88','M11'],f);im.save(D/'comparison'/key/f'{pn}_texture_f{f:03d}_s{v:02d}.png');gray=cv2.cvtColor(crop.astype('float32'),cv2.COLOR_RGB2GRAY);rows.append(dict(key=key,frame=f,view=v,part=pn,status='reference_proxy_crop',crop=[x0,y0,x1,y1],**spectral(gray)))
 write(D/'texture_scale.csv',rows)
 # Pool projected support means by actual contribution, never mean of quantiles.
 disp=load(D/'displacement_support.csv');summary=[]
 for key in sorted(set(r['key'] for r in disp)):
  sel=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text())['selections'][key[:3]];win=sel['window']
  for roi in ['expanded_0','torso']:
   x=[r for r in disp if r['key']==key and r['view']==win['view'] and win['start']<=r['frame']<=win['end'] and r['roi']==roi];mass=sum(r['contribution_mass'] for r in x)
   summary.append(dict(key=key,roi=roi,contribution_mass=mass,**{n:sum((r[n] or 0)*r['contribution_mass'] for r in x)/mass for n in ['ellipse_minor_sigma_px_mean','ellipse_major_sigma_px_mean','ellipse_3sigma_area_px_mean','delta_norm_L_mean','projected_shift_px_mean','mahalanobis_mean','saturated_any_component_mean']}))
 write(D/'displacement_window_summary.csv',summary)
 print('texture descriptors',len(rows))
if __name__=='__main__':main()
