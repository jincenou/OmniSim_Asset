"""Show fixed canonical views for manual front/back review; no score selection."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--name',required=True);p.add_argument('--limit',type=int,default=5);p.add_argument('--matrix',type=Path,default=ROOT/'inputs/random10_experiment_matrix.json');a=p.parse_args()
tasks=json.loads(a.matrix.read_text());rows=[]
reviewed={(r['case_id'],r['variant'],r['seed']) for r in json.loads((ROOT/'inputs/random10_heading_decisions.json').read_text())}
for t in tasks:
 if (a.variant!='all' and t['variant']!=a.variant) or t['status']!='rendered_heading_review_pending':continue
 if (t['case_id'],t['variant'],t['seed']) in reviewed:continue
 asset=ROOT/'results'/t['case_id']/f"{t['variant']}_seed{t['seed']}"/'asset'
 if (asset/'heading_review.json').exists():continue
 rows.append((t,asset))
 if len(rows)>=a.limit:break
if not rows:raise ValueError('No pending heading review')
sheet=Image.new('RGB',(1120,250*len(rows)),'#ddd');draw=ImageDraw.Draw(sheet)
for i,(t,asset) in enumerate(rows):
 for j,v in enumerate([0,8,2,10]):
  im=Image.open(asset/f'view_{v:02}.png');alpha=np.array(Image.open(asset/f'alpha_{v:02}.png'));yy,xx=np.where(alpha>127)
  if len(xx)==0:raise ValueError('Empty view')
  box=(max(0,xx.min()-12),max(0,yy.min()-12),min(im.width,xx.max()+13),min(im.height,yy.max()+13));tile=ImageOps.contain(im.crop(box),(275,215));sheet.paste(tile,(j*280,i*250+30));draw.text((j*280+3,i*250+4),f"{t['case_id'][9:]} {t['variant']} {v*22.5} deg",fill='black')
out=ROOT/'docs'/a.name;sheet.save(out.with_suffix('.jpg'),quality=95);out.with_suffix('.json').write_text(json.dumps([t for t,asset in rows],indent=2));print([(t['case_id'],t['variant']) for t,asset in rows])
