import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
R=Path(__file__).resolve().parent;G=R.parent
rows=json.loads((G/'manifest.json').read_text())['rows'];out=[]
for page in range((len(rows)+15)//16):
 canvas=Image.new('RGB',(1760,800),'#252525');d=ImageDraw.Draw(canvas)
 for j,row in enumerate(rows[page*16:(page+1)*16]):
  index=page*16+j;c=row['candidate'];base=G/'scenes'/c['split']/c['scene']/c['review_id'];p=Path(row['package']);im=Image.open(base/'contact.jpg').convert('RGB');nat=Image.open(p/'clip0/native_contact.jpg').convert('RGB');w=nat.width//5
  tiles=[nat.crop((2*w,0,3*w,nat.height)),im.crop((0,20,264,500)),im.crop((792,20,1056,500)),im.crop((1584,20,1848,500)),im.crop((792,520,1056,1000))]
  x=(j%4)*440;y=(j//4)*200
  for k,tile in enumerate(tiles):canvas.paste(ImageOps.pad(tile,(88,160),color='#252525'),(x+k*88,y+40))
  d.text((x+3,y+2),f'{index:03d} {row["id"]}',fill='white');d.text((x+3,y+19),'source mid | v00: first/mid/end | v12 mid',fill='#bbbbbb')
  out.append({'index':index,'id':row['id'],'output':str(base),'package':str(p),'frames':c['output_slots'],'flags':row['skeleton_audit'].get('flags',[]),'overview':f'overview_{page:02d}.jpg','contact_sha256':hashlib.sha256((base/'contact.jpg').read_bytes()).hexdigest()})
 canvas.save(R/f'overview_{page:02d}.jpg',quality=92)
(R/'overview_index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('227cases in15 overview sheets; thumbnails are triage only, not full temporal review')
