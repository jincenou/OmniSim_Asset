import concurrent.futures,datetime,hashlib,json,re,urllib.parse,urllib.request
from pathlib import Path
from PIL import Image
W=Path(__file__).resolve().parents[1]
sources=[('P04','File:-2020-01-28 Diversion Sign, High Street, Mundesley.JPG'),('P05','File:Diversion sign, Charterhouse Street, EC1 - geograph.org.uk - 825234.jpg'),('P06','File:A road diversion sign at St Boswells - geograph.org.uk - 3312126.jpg')]
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RoadSignResearch/1.0 (local educational asset research)'}),timeout=30) as r:return r.read()
def one(s):
 sid,title=s;p=W/'sources'/sid;p.mkdir(exist_ok=True);url='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(title.replace(' ','_'));r=dict(source_id=sid,title=title,page_url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  raw=get(url);(p/'page.html').write_bytes(raw)
  text=raw.decode();link=re.search(r'<div class="fullImageLink".*?<a href="([^"]+)"',text,re.S)
  if not link:raise RuntimeError('No original image link')
  image_url=link.group(1).replace('&amp;','&');b=get(image_url);(p/'source.jpg').write_bytes(b)
  with Image.open(p/'source.jpg') as im:im.load();size=im.size
  licenses=list(set(re.findall(r'https?://creativecommons.org/licenses/[^"<>\s]+',text)))
  r.update(status='downloaded_decoded',media_url=image_url,sha256=hashlib.sha256(b).hexdigest(),size=size,license_links=licenses,attribution_source='page.html; transcribe author before asset use')
 except Exception as e:r.update(status='failed',error=str(e))
 (p/'acquisition.json').write_text(json.dumps(r,indent=2));print(sid,r['status'],r.get('size'),r.get('error'),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(one,sources))
