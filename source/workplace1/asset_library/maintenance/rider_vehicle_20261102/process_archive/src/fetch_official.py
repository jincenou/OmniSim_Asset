import re,json,urllib.parse,urllib.request,concurrent.futures
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(__file__).resolve().parents[1]/'sources/niu_official'
s=urllib.parse.unquote((p/'page.html').read_text());urls=list(dict.fromkeys(re.findall(r'https?[^\s\"<>\\]+?\.(?:jpg|png)',s)))
urls=[u for u in urls if (m:=re.search(r'-(\d+)x(\d+)\.',u)) and max(map(int,m.groups()))>1500]
def fetch(pair):
 i,u=pair; dest=p/f'{i:02}.jpg'
 urllib.request.urlretrieve(u+'?w=1000&auto=format',dest)
 return dict(path=str(dest),url=u)
with concurrent.futures.ThreadPoolExecutor(4) as pool: rows=list(pool.map(fetch,enumerate(urls)))
(p/'images.json').write_text(json.dumps(rows,indent=2));sheet=Image.new('RGB',(1000,220*((len(rows)+3)//4)),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 im=Image.open(r['path']);im.thumbnail((250,195));x=i%4*250;y=i//4*220;sheet.paste(im,(x,y));d.text((x,y+198),str(i),fill='black')
sheet.save(p/'sheet.jpg')
