"""Native decoded RGB with the unchanged half-resolution alpha master."""
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import av
import numpy as np
from PIL import Image

ROUND=Path(__file__).resolve().parents[1];R=ROUND.parent/'reconstruction'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def view(label,source,destination,index):
    original=R/'data'/label;folder=destination/'images'/f'cam{index:02d}';folder.mkdir(parents=True,exist_ok=False)
    frames=[];maximum=0
    video=source/'videos'/f'{index:02d}.mp4'
    with av.open(str(video)) as container:
        container.streams.video[0].thread_count=1
        for frame_index,frame in enumerate(container.decode(video=0)):
            if float(frame.pts*container.streams.video[0].time_base)!=frame_index/25:raise ValueError('Unexpected native timestamp')
            rgb=frame.to_image().convert('RGB')
            if rgb.size!=(704,1280):raise ValueError('Source video is not native model output resolution')
            old=original/'images'/f'cam{index:02d}'/f'{frame_index:06d}.png'
            with Image.open(old) as master:
                if master.mode!='RGBA' or master.size!=(352,640):raise ValueError('Unexpected alpha master')
                reduced=np.asarray(rgb.resize((352,640),Image.Resampling.LANCZOS)).astype(np.int16)
                difference=int(np.abs(reduced-np.asarray(master)[...,:3].astype(np.int16)).max())
                if difference!=0:raise ValueError(f'Frozen half RGB is not identical decoded source: {label}/{index}/{frame_index}, {difference}')
                alpha=master.getchannel('A').resize(rgb.size,Image.Resampling.BILINEAR)
                maximum=max(maximum,difference);rgb.putalpha(alpha)
            path=folder/f'{frame_index:06d}.png';rgb.save(path)
            frames.append({'frame':frame_index,'seconds':frame_index/25,'rgba_sha256':sha(path),'half_master_sha256':sha(old)})
    if len(frames)!=121:raise ValueError('Incomplete native video')
    record={'view':index,'source_video':str(video),'video_sha256':sha(video),'frames':frames,'half_rgb_reproduction_max_abs_byte':maximum}
    save(destination/f'view{index:02d}.json',record)
    print(json.dumps({'label':label,'view':index,'frames':len(frames)}),flush=True)


async def main():
    labels=['phone_full','pilot_043_twentyfour'];jobs=[];semaphore=asyncio.Semaphore(2);outputs=[]
    async def bounded(*args):
        async with semaphore:await asyncio.to_thread(view,*args)
    for label in labels:
        record=R/'research'/('full-export.json' if label=='phone_full' else f'{label}-export.json')
        source=Path(json.loads(record.read_text())['source']);original=R/'data'/label
        destination=ROUND/'quality_masters'/label/'native';destination.mkdir(parents=True,exist_ok=False)
        sparse=destination/'sparse/0';sparse.mkdir(parents=True)
        for name in ('images.txt','points3D.txt'):shutil.copy2(original/'sparse/0'/name,sparse/name)
        native=json.loads((source/'cameras.json').read_text())['cameras'];lines=[]
        for line in (original/'sparse/0/cameras.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):lines.append(line);continue
            parts=line.split();assert parts[1]=='PINHOLE' and parts[2:4]==['352','640']
            index=int(parts[0])-1;parameters=np.asarray(list(map(float,parts[4:])))*2
            k=np.asarray(native[index]['K']);expected=np.array([k[0,0],k[1,1],k[0,2],k[1,2]])
            if not np.allclose(parameters,expected,atol=1e-4,rtol=0):raise ValueError('Rescaled K differs from native generated camera')
            lines.append(' '.join([parts[0],'PINHOLE','704','1280',*map(lambda x:f'{x:.12g}',parameters)]))
        (sparse/'cameras.txt').write_text('\n'.join(lines)+'\n')
        (destination/'init.npz').symlink_to(original/'init.npz')
        outputs.append((label,destination,source));jobs.extend(bounded(label,source,destination,index) for index in range(24))
    results=await asyncio.gather(*jobs,return_exceptions=True)
    failures=[str(value) for value in results if isinstance(value,BaseException)]
    if failures:raise RuntimeError(json.dumps(failures))
    for label,destination,source in outputs:
        save(destination/'manifest.json',{'state':'native_rgb_with_fixed_alpha_master_ready','label':label,'source':str(source),
            'views':24,'frames':121,'dimensions':[704,1280],'RGB':'decoded original704x1280 video; never upscaled',
            'alpha':'original352x640 predicted alpha, bilinear-resampled to704x1280; no claim of native-resolution mask detail',
            'half_control':'all2904 native RGB frames downsample byte-identically to frozen Q0 RGB',
            'geometry':'original21-view initialization and COLMAP PCA points unchanged; K doubled in both image axes',
            'init_sha256':sha(destination/'init.npz'),'per_frame_hashes':'view00.json through view23.json','gpu_used':False})
    print('Both native RGB masters complete; no new mask model or GPU work',flush=True)


if __name__=='__main__':asyncio.run(main())
