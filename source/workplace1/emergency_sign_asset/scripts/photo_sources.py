import concurrent.futures,datetime,hashlib,html,json,pathlib,re,urllib.request
from urllib.parse import urljoin,urlparse
W=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[('P01','https://www.thebannerhub.co.uk/road-closed-metal-sign-frame','road|closed|frame'),('P02','https://www.safesitefacilities.co.uk/products/diversion-left-temporary-road-sign-1050mm-750mm','diversion|rear.view|1050'),('P03','https://www.proteclight.com/3-faces-foldable-tripod-car-crash-381.html','381|tripod|accident|ak31|car.crash'),('D01','https://www.deflecto.com/wp-content/uploads/2022/06/EWT-Kit-Flyer_120921.pdf','')]
def fetch(u):
 with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:return r.read(),dict(r.headers),r.url

def one(source):
 sid,u,pattern=source;p=W/'sources'/sid;p.mkdir(exist_ok=True);record=dict(source_id=sid,page_url=u,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),license_status='unknown',media=[])
 try:
  raw,headers,final=fetch(u);record['page_response']=dict(headers=headers,final_url=final);(p/('spec.pdf' if sid=='D01' else 'page.html')).write_bytes(raw)
  if sid!='D01':
   txt=html.unescape(raw.decode(errors='replace')).replace('\\/','/')
   urls=set(urljoin(u,m) for m in re.findall(r'''(?:https?://|/)[^\s"'<>]+?\.(?:jpe?g|png|webp)(?:\?[^\s"'<>]*)?''',txt,re.I));urls=sorted(x for x in urls if re.search(pattern,x,re.I))
   record['candidate_image_urls']=urls
   for i,img in enumerate(urls[:18]):
    try:
     b,h,f=fetch(img);name=f'image_{i:02d}'+pathlib.Path(urlparse(img).path).suffix.lower();(p/name).write_bytes(b)
     from PIL import Image
     with Image.open(p/name) as im: im.load();size=im.size
     record['media'].append(dict(url=img,final_url=f,path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),size=size))
    except Exception as e:record['media'].append(dict(url=img,error=str(e)))
  record['status']='fetched'
 except Exception as e: record.update(status='failed',error=str(e))
 (p/'acquisition.json').write_text(json.dumps(record,indent=2));print(sid,record['status'],len(record['media']),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,SOURCES))
