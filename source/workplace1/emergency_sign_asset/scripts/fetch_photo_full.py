import concurrent.futures,hashlib,json,re,urllib.request
from pathlib import Path
from PIL import Image
W=Path(__file__).resolve().parents[1];p=W/'sources/P01'
s=(p/'page.html').read_text().replace('\\/','/')
urls=list(dict.fromkeys(re.findall(r'"full":"([^"]+)"',s)))
def one(pair):
 i,u=pair;row=dict(url=u)
 try:
  with urllib.request.urlopen(u,timeout=20) as r:b=r.read()
  f=p/f'full_{i}.png';f.write_bytes(b)
  with Image.open(f) as im:im.load();size=im.size
  row.update(path=f.name,sha256=hashlib.sha256(b).hexdigest(),size=size)
 except Exception as e:row['error']=str(e)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,enumerate(urls)))
(p/'full_acquisition.json').write_text(json.dumps(dict(page_url='https://www.thebannerhub.co.uk/road-closed-metal-sign-frame',license_status='unknown',media=rows),indent=2));print(rows)
