"""Download an auditable, balanced first wave of Commons vehicle photographs."""
import concurrent.futures,hashlib,json,time,urllib.parse,urllib.request,io
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
UA='DriveLabVehicleResearch/1.0 (vehicle reconstruction research; source attribution retained)'
CATEGORIES={
 'bus':['Side views of buses','Three-quarter views of buses'],
 'truck':['Side views of trucks','Three-quarter views of trucks'],
 'van':['Side views of vans','Three-quarter views of vans','Panel vans','Ford Transit vans','Mercedes-Benz Sprinter vans']}

def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def get(url):
 for retry in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=40) as r:
    if int(r.headers.get('Content-Length',0))>25*1024**2:raise ValueError('Source exceeds 25 MiB download cap')
    b=r.read(25*1024**2+1)
    if len(b)>25*1024**2:raise ValueError('Source exceeds 25 MiB download cap')
    return b
  except Exception:
   if retry==2:raise
   time.sleep(3*(retry+1))

def catalogue():
 path=ROOT/'catalogue.json'
 if path.exists():return json.loads(path.read_text())
 selected=[];seen=set();counts={}
 for category,names in CATEGORIES.items():
  candidates=[]
  for name in names:
   params=dict(action='query',format='json',generator='categorymembers',gcmtitle='Category:'+name,gcmtype='file',gcmlimit=100,prop='imageinfo',iiprop='url|extmetadata|size',iiurlwidth=1280)
   try:
    raw=get('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params))
    out=ROOT/'search'/category/(hashlib.sha256(name.encode()).hexdigest()[:12]+'.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    data=json.loads(raw)
    for page in sorted(data.get('query',{}).get('pages',{}).values(),key=lambda x:x['pageid']):
     info=page.get('imageinfo',[{}])[0];url=info.get('url','');meta=info.get('extmetadata',{})
     if not url or page['pageid'] in seen or min(info.get('width',0),info.get('height',0))<300 or max(info.get('width',0),info.get('height',0))<800:continue
     if not url.lower().split('?')[0].endswith(('.jpg','.jpeg','.png')):continue
     seen.add(page['pageid']);candidates.append(dict(source_id='commons_'+str(page['pageid']),category=category,page_id=page['pageid'],title=page['title'],page_url=info['descriptionurl'],original_url=url,download_url=info.get('thumburl',url),metadata=meta,native_size=[info['width'],info['height']],search_category=name))
   except Exception as e:
    save(ROOT/'search'/category/(hashlib.sha256(name.encode()).hexdigest()[:12]+'.error.json'),dict(category=name,error=repr(e)))
   time.sleep(.3)
  # Bound each acquisition wave, not the lifetime library size.
  candidates=candidates[:120];selected.extend(candidates);counts[category]=len(candidates)
 result=dict(created=time.time(),wave='first',per_category_candidate_cap=120,counts=counts,sources=selected)
 save(path,result);return result

def download(a):
 folder=ROOT/'sources'/a['source_id'];record=folder/'acquisition.json'
 if record.exists():return json.loads(record.read_text())
 folder.mkdir(exist_ok=True);r=dict(a,retrieved=time.time())
 try:
  raw=get(a['download_url']);im=Image.open(io.BytesIO(raw));im.load();im=im.convert('RGB')
  (folder/'download.bin').write_bytes(raw);im.save(folder/'source.jpg',quality=95)
  r.update(status='downloaded',size=list(im.size),download_sha256=hashlib.sha256(raw).hexdigest(),image_path=str(folder/'source.jpg'),sha256=hashlib.sha256((folder/'source.jpg').read_bytes()).hexdigest())
 except Exception as e:r.update(status='failed',error=repr(e))
 save(record,r);print(a['source_id'],r['status'],flush=True);return r

if __name__=='__main__':
 save(ROOT/'acquisition_status.json',dict(status='searching',started=time.time()))
 doc=catalogue();save(ROOT/'acquisition_status.json',dict(status='downloading',counts=doc['counts'],started=time.time()))
 # Interleave categories so the consumer can start with all three classes.
 grouped={c:[a for a in doc['sources'] if a['category']==c] for c in CATEGORIES};order=[]
 for i in range(max(map(len,grouped.values()),default=0)):
  order.extend(a[i] for a in grouped.values() if i<len(a))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(download,order))
 save(ROOT/'acquisition_status.json',dict(status='completed',counts=doc['counts'],downloaded=sum(a['status']=='downloaded' for a in results),failed=sum(a['status']=='failed' for a in results),completed=time.time()))
