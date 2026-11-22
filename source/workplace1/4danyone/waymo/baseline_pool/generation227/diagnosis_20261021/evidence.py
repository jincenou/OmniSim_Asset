import asyncio,json,hashlib,sys,os
from pathlib import Path
import av
from PIL import Image,ImageDraw,ImageOps
import numpy as np
R=Path(__file__).resolve().parent;rows=json.loads((R/'overview_index.json').read_text())
SELECTED=[1,3,8,18,42,44,64,85,118,123,145,159,199,225,0,10,14,47,56,59,88,119,153,160,189,224]
def decode(path,indices):
 wanted=set(indices);out={}
 with av.open(str(path)) as c:
  c.streams.video[0].thread_count=1
  for i,f in enumerate(c.decode(video=0)):
   if i in wanted:out[i]=f.to_image()
   if i>=max(wanted):break
 assert len(out)==len(wanted),(path,len(out),len(wanted))
 return out

def one(index):
 row=rows[index];b=Path(row['output']);p=Path(row['package']);n=row['frames'];ts=np.rint(np.linspace(0,n-1,7)).astype(int).tolist();mapping=json.loads((p/'clip0/inputs/frame_mapping.json').read_text())['frames'];native=[mapping[t]['left_native_index'] if mapping[t]['interpolation_fraction']<=.5 else mapping[t]['right_native_index'] for t in ts]
 sources=[('observed',p/'clip0/observed_native.mp4',native,120),('model input',p/'clip0/inputs/main.mp4',ts,120),('source pose overlay',p/'clip0/motion_pair/projection_comparison.mp4',ts,240),('target skeleton00',b/'skeletons/00.mp4',ts,120),('generated00',b/'videos/00.mp4',ts,120),('generated12',b/'videos/12.mp4',ts,120)]
 canvas=Image.new('RGB',(840,7*210+40),'#202020');draw=ImageDraw.Draw(canvas);draw.text((4,3),f'{index:03d} {row["id"]} | columns are different cameras; compare skeleton00 with generated00',fill='white');x=0
 for label,path,inds,width in sources:
  decoded=decode(path,inds);draw.text((x+3,22),label,fill='#eee')
  for j,(t,i) in enumerate(zip(ts,inds)):
   tile=ImageOps.pad(decoded[i],(width,190),color='#202020');canvas.paste(tile,(x,40+j*210));draw.text((x+3,40+j*210+191),f't={t/25:.2f}s frame={i}',fill='#eee')
  x+=width
 canvas.save(R/f'evidence_{index:03d}.jpg',quality=93)
 audit=json.loads((p/'skeleton_audit.json').read_text());mask=json.loads((p/'clip0/research/batch_gate_mask.json').read_text())
 result={'index':index,'id':row['id'],'slots':ts,'nearest_native_indices':native,'native_alignment':'Nearest bracketing native exposure via recorded mapping; not same frame count or exact same timestamp','sources':[{'label':l,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for l,path,_,_ in sources],'skeleton_audit':audit,'mask_risk':{k:v for k,v in mask.items() if not isinstance(v,list)},'sheet':f'evidence_{index:03d}.jpg','viewed':False}
 (R/f'evidence_{index:03d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(index,flush=True)
async def main():
 sem=asyncio.Semaphore(2)
 async def run(i):
  async with sem:
   if (R/f'evidence_{i:03d}.json').exists():return
   proc=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__)),str(i),env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
   code=await proc.wait()
   if code:raise RuntimeError(f'evidence {i}: exit{code}')
 await asyncio.gather(*(run(i) for i in SELECTED))
if __name__=='__main__':
 if len(sys.argv)>1:one(int(sys.argv[1]))
 else:asyncio.run(main())
