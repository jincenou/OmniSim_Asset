import json,urllib.request,concurrent.futures,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];p=W/'sources/niu_official';rows=json.loads((p/'images.json').read_text())
def fetch(i):
 r=rows[i];dest=p/f'original_{i:02}.jpg';urllib.request.urlretrieve(r['url'],dest);return dict(index=i,path=str(dest),url=r['url'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),page='https://global.niu.com/en-us/product/NQi-GTS',license='No explicit reuse license identified; internal research only',temporal_evidence='single photograph; no temporal rigidity claim')
with concurrent.futures.ThreadPoolExecutor(3) as ex: result=list(ex.map(fetch,[3,6,10]))
(p/'selected_originals.json').write_text(json.dumps(result,indent=2));print('Downloaded',len(result),'originals')
