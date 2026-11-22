"""Find large vehicle observations in new Waymo clips; annotations guide search only."""
import asyncio,json,struct,sys,hashlib
import faulthandler
faulthandler.dump_traceback_later(30, repeat=True)
from pathlib import Path
sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
from waymo_open_dataset import dataset_pb2
from PIL import Image,ImageOps,ImageDraw,ImageFont
from io import BytesIO
Image.init()
B=Path(__file__).resolve().parent;O=B/'search';O.mkdir(exist_ok=True)
records=sorted(Path('/mnt/public_data/waymo_raw/validation').glob('*.tfrecord'))
old=json.loads(Path('/mnt/ojc/workplace1/asset_work/batch20_20261024/registry.json').read_text());excluded={r['raw_tfrecord'] for r in old['waymo']}
async def scan(path):
 def work():
  with path.open('rb') as f:
   head=f.read(12);n=struct.unpack('<Q',head[:8])[0];data=f.read(n)
  fr=dataset_pb2.Frame();fr.ParseFromString(data)
  large={l.id:l for l in fr.laser_labels if l.type==1 and l.box.length>=7.5 and l.box.height>=2.4}
  if not large:return []
  groups=[]
  for group in fr.projected_lidar_labels:
   for l in group.labels:
    track=l.id.rsplit('_',1)[0]
    # The camera suffix can contain an underscore; test the known object IDs directly.
    ids=[oid for oid in large if l.id.startswith(oid+'_')]
    if not ids:continue
    oid=ids[0];b=l.box
    if b.length<180 or b.width<95:continue
    image=next(x for x in fr.images if x.name==group.name)
    im=Image.open(BytesIO(image.image));w,h=im.size
    bbox=[b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2]
    if bbox[0]<5 or bbox[1]<5 or bbox[2]>w-5 or bbox[3]>h-5:continue
    pad=max(b.length,b.width)*.08;crop=[max(0,int(bbox[0]-pad)),max(0,int(bbox[1]-pad)),min(w,int(bbox[2]+pad)),min(h,int(bbox[3]+pad))]
    key=hashlib.sha256((str(path)+oid+str(group.name)).encode()).hexdigest()[:16]
    cp=O/f'{key}.jpg';im.crop(crop).save(cp,quality=93)
    groups.append(dict(key=key,tfrecord=str(path),context=fr.context.name,track_id=oid,camera=int(group.name),native_frame_index=0,timestamp_us=fr.timestamp_micros,bbox=bbox,image_wh=[w,h],crop=str(cp),area=b.length*b.width,search_length=large[oid].box.length,search_height=large[oid].box.height))
  print('found',path.stem,len(groups),flush=True)
  return groups
 async with sem:return work()
async def main():
 global sem
 sem=asyncio.Semaphore(1)
 results=await asyncio.gather(*(scan(p) for p in records if str(p) not in excluded))
 rows=[r for group in results for r in group];byid={}
 for r in sorted(rows,key=lambda r:-r['area']):byid.setdefault((r['context'],r['track_id']),r)
 rows=list(byid.values());(O/'candidates.json').write_text(json.dumps(rows,indent=2));print('clips',len(records),'new large distinct candidates',len(rows),flush=True)
 font=ImageFont.truetype('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',22)
 for start in range(0,len(rows),20):
  canvas=Image.new('RGB',(2000,1750),'#162331');draw=ImageDraw.Draw(canvas)
  for j,r in enumerate(rows[start:start+20]):
   x=j%4*500;y=j//4*350;draw.text((x+10,y+5),f'{start+j:03} {r["search_length"]:.1f}m / cam{r["camera"]}',font=font,fill='white');canvas.paste(ImageOps.pad(Image.open(r['crop']).convert('RGB'),(480,310),color='white'),(x+10,y+35))
  canvas.save(O/f'candidates_{start//20+1:02}.jpg',quality=94)
asyncio.run(main())
