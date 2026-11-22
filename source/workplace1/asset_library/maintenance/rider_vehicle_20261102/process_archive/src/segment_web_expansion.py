from pathlib import Path
import os,json,hashlib
os.environ['U2NET_HOME']='/mnt/wmj/workspace4/pipeline'
import rembg
from PIL import Image,ImageOps,ImageDraw
W=Path(__file__).resolve().parents[1];src=W/'sources/web_expansion_20261103'
config={
'003':('mqi_gt_cream',[.21,.11,.44,.79],'black open-face helmet, cream jacket, pale blue jeans and dark shoes','dark grey MQi GT scooter with circular headlight'),
'011':('mqi_gt_blazer',[.27,.28,.85,.96],'black open-face helmet, grey blazer, white shirt, beige trousers and beige lace-up boots','dark grey MQi GT scooter with circular headlight'),
'043':('fqi_black_polo',[.46,.11,.88,.83],'white open-face helmet, sunglasses, black short-sleeve polo, brown trousers, black gloves and boots','black and silver FQi scooter with angular front fairing'),
'055':('mqi_sport_white',[.32,.35,.66,.96],'white helmet, light grey short-sleeve shirt, dark blue trousers and white high-top shoes','red MQi Sport electric scooter'),
'079':('nqi_sport_green',[.46,.12,.78,.94],'dark green helmet, long brown hair, black coat, ripped grey jeans and black boots','muted green NQi Sport electric scooter'),
'115':('nqi_sport_river',[.43,.35,.81,.85],'black open-face helmet, dark jacket, blue jeans and grey shoes','dark red and black NQi Sport scooter'),
'130':('fqix_orange',[.30,.04,.96,.80],'silver and black full-face helmet, black short-sleeve top, peach vest, grey cargo trousers, black gloves and grey shoes','orange and camouflage-black FQiX 300 scooter')}
session=rembg.new_session('birefnet-general-lite',providers=['CPUExecutionProvider'])
records=[]
for r in json.loads((src/'selected.json').read_text()):
 if r['id'] not in config:continue
 short,frac,rider,vehicle=config[r['id']];name='web26_'+short;p=W/'packets'/name;p.mkdir(exist_ok=True);s=W/'sources'/name;s.mkdir(exist_ok=True)
 im=Image.open(r['original_path']).convert('RGB');box=[round(frac[i]*im.size[i%2]) for i in range(4)];crop=im.crop(box);crop.save(p/'source_crop.png')
 rgba=rembg.remove(crop,session=session);rgba.save(p/'anchor_rgba.png');white=Image.new('RGB',rgba.size,'white');white.paste(rgba,mask=rgba.getchannel('A'));white.save(p/'anchor.jpg',quality=95)
 ob=dict(id='birefnet_native',image_path=str(p/'source_crop.png'),anchor_path=str(p/'anchor.jpg'),rgba_path=str(p/'anchor_rgba.png'),sha256=hashlib.sha256((p/'anchor_rgba.png').read_bytes()).hexdigest(),segmentation_method='BiRefNet general lite; original RGB crop without generative edits',source_observation=dict(source_url=r['url'],page=r['page'],original_path=r['original_path'],original_sha256=r['sha256'],source_crop_xyxy=box,shot_id=name+'_single_image',temporal_evidence='Single image; no temporal stability assertion',license=r['license']))
 (p/'segmentation.json').write_text(json.dumps(dict(observations=[ob]),indent=2));(p/'attributes.json').write_text(json.dumps(dict(source_key=name,rider_appearance=rider,vehicle_appearance=vehicle),indent=2));(s/'source.json').write_text(json.dumps(dict(page=r['page'],download_url=r['url'],license=r['license']),indent=2))
 records.append(dict(name=name,candidate_id=r['id'],status='segmentation_review_pending',native_crop_size=crop.size));print(name,crop.size,flush=True)
(src/'queue.json').write_text(json.dumps(records,indent=2))
sheet=Image.new('RGB',(1200,((len(records)+3)//4)*420),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(records):
 im=ImageOps.contain(Image.open(W/'packets'/r['name']/'anchor.jpg'),(296,390));x=i%4*300;y=i//4*420;sheet.paste(im,(x,y+25));d.text((x+3,y+5),r['candidate_id']+' '+r['name'][6:],fill='black')
sheet.save(src/'segmentation_sheet.jpg')
