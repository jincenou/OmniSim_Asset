import hashlib,json,urllib.request
from pathlib import Path
import fitz
W=Path(__file__).resolve().parents[1];p=W/'sources/P07';p.mkdir(exist_ok=True)
url='https://www.tennantsuk.com/wp-content/uploads/2023/01/Quick-Lock-Signs-And-Deployment-Guide-Web.pdf'
with urllib.request.urlopen(url,timeout=40) as r:b=r.read()
(p/'source.pdf').write_bytes(b);doc=fitz.open(stream=b,filetype='pdf');rows=[]
for i,page in enumerate(doc):
 page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(p/f'page_{i}.png')
 for image in page.get_images(full=True):
  item=doc.extract_image(image[0]);name=f'image_{image[0]}.{item["ext"]}';(p/name).write_bytes(item['image']);rows.append(dict(path=name,width=item['width'],height=item['height'],sha256=hashlib.sha256(item['image']).hexdigest(),page=i))
(p/'acquisition.json').write_text(json.dumps(dict(source_id='P07',page_url=url,sha256=hashlib.sha256(b).hexdigest(),publisher='Tennants UK',license_status='unknown',media=rows),indent=2));print(rows)
