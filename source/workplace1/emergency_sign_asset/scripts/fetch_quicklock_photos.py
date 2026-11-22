import urllib.request,json,hashlib,re
from pathlib import Path
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
W=Path(__file__).resolve().parents[1];d=W/'sources/P08';d.mkdir(exist_ok=True)
u='https://www.tennantsuk.com/highway-equipment/quick-lock-signs/'
b=urllib.request.urlopen(u,timeout=40).read();(d/'page.html').write_bytes(b)
urls=sorted(set(re.findall(r'https://www.tennantsuk.com/wp-content/uploads/[^"<> ]+?\.(?:jpg|jpeg|png|webp)',b.decode())))
rows=[]
for i,url in enumerate(urls):
 if not any(x in url.lower() for x in ['quick','lock','sign','diversion']):continue
 try:
  b=urllib.request.urlopen(url,timeout=30).read();im=Image.open(BytesIO(b));name=f'image_{i:03}{Path(url).suffix}';(d/name).write_bytes(b);rows.append(dict(url=url,path=name,size=im.size,sha256=hashlib.sha256(b).hexdigest()))
 except Exception as e:rows.append(dict(url=url,error=str(e)))
(d/'acquisition.json').write_text(json.dumps(dict(source_id='P08',page_url=u,publisher='Tennants UK',license_status='unknown',media=rows),indent=2));print(rows)
