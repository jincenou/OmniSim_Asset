"""Compare original TRELLIS FLUX baseline with frozen real-image additions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from asset_adapter import ROOT,inspect

p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed-check',action='store_true');a=p.parse_args()
variants=['B0','B0_real1','B0_real4','H']
labels=['B0: FLUX 2 / TRELLIS','FLUX 2 + real 1 / TRELLIS','FLUX 2 + real 4 / TRELLIS','H: FLUX 2 + real 4 / ReconViaGen']
specs=[(v,42) for v in variants]
stem='trellis_real_assist_comparison'
if a.seed_check:
    specs=[('B0_real1',42),('B0_real1',7),('B0_real4',42),('B0_real4',7)]
    labels=['FLUX 2 + real 1 / seed 42','FLUX 2 + real 1 / seed 7','FLUX 2 + real 4 / seed 42','FLUX 2 + real 4 / seed 7']
    stem='trellis_real_assist_seed_comparison'
out=ROOT/'results'/a.case;assets=[out/f'{v}_seed{seed}/asset' for v,seed in specs]
configs=[json.loads((x/'render_config.json').read_text()) for x in assets]
for key in ['length_m','distance_m','focal_px','size','elevation_deg','center_m','background','azimuths_deg','video_frames','video_fps']:
    assert len({str(c[key]) for c in configs})==1,key
for asset in assets:
    inspect(asset/'canonical.ply')
    assert json.loads((asset/'experiment.json').read_text())['heading_verified']
    for i in range(16):
        assert Image.open(asset/f'view_{i:02}.png').size==(512,512)
        assert Image.open(asset/f'alpha_{i:02}.png').getextrema()[1]>0
for v,seed in specs:
    if v not in ('B0_real1','B0_real4'):continue
    n=1 if v=='B0_real1' else 4
    d=out/f'{v}_seed{seed}';manifest=json.loads((d/'input_manifest.json').read_text())
    rows=manifest['images'];assert len(rows)==n+2
    assert sum(x['source_type']=='real' for x in rows)==n
    assert all(x['split']=='train' for x in rows)
    for row in rows:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    for i in range(2):
        # Identical processed FLUX pixels to the existing B0 control.
        assert np.array_equal(np.asarray(Image.open(d/f'processed_{i:02}.png')),
                              np.asarray(Image.open(out/f'B0_seed42/processed_{i:02}.png')))

size=512;pad=32
poster=Image.new('RGB',(size*4,(size+pad)*3),'#eee');draw=ImageDraw.Draw(poster)
for j,(asset,label) in enumerate(zip(assets,labels)):
    for k,i in enumerate([1,4,8]):
        x=j*size;y=k*(size+pad);poster.paste(Image.open(asset/f'view_{i:02}.png'),(x,y+pad))
        draw.text((x+8,y+8),label,fill='black')
poster.save(out/f'{stem}.jpg',quality=94)
readers=[imageio.get_reader(x/'turntable.mp4') for x in assets]
writer=imageio.get_writer(out/f'{stem}.mp4',fps=16,codec='libx264',quality=8)
try:
    for i in range(96):
        canvas=Image.new('RGB',(1024,1088),'#eee');d=ImageDraw.Draw(canvas)
        for j,(reader,label) in enumerate(zip(readers,labels)):
            x=j%2*512;y=j//2*544;canvas.paste(Image.fromarray(reader.get_data(i)),(x,y+32))
            d.text((x+8,y+8),label,fill='black')
        writer.append_data(np.asarray(canvas))
finally:
    writer.close()
    for reader in readers:reader.close()
(out/f'{stem}.json').write_text(json.dumps(dict(variants=specs,
    render_config=configs[0],same_processed_flux_as_B0=True,original_loader_verified=True,
    real_inputs='frozen train; one=FLUX reference real view, four=all four real train views',
    heldout_used=False,decoded_frames_per_input=96,
    scope='Standalone rendered Gaussian comparison; new mixed TRELLIS assets have not been inserted into SplatAD'),indent=2))
print(out/f'{stem}.mp4')
