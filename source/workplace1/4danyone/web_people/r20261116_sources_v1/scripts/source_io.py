"""Original RGB frames and native timestamps; no interpolation or enhancement."""
from pathlib import Path
import configparser
import hashlib
import json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()


def read_source(path):
    import av
    import numpy as np
    from PIL import Image
    path=Path(path)
    if path.is_dir():
        config=configparser.ConfigParser();config.read(path/'seqinfo.ini');cfg=config['Sequence']
        fps=int(cfg['frameRate']);length=int(cfg['seqLength']);width=int(cfg['imWidth']);height=int(cfg['imHeight'])
        for index in range(length):
            file=path/cfg['imDir']/f'{index+1:06d}{cfg["imExt"]}'
            rgb=np.asarray(Image.open(file).convert('RGB'))
            assert rgb.shape==(height,width,3),file
            yield index,index/fps,rgb
    else:
        with av.open(str(path)) as con:
            con.streams.video[0].thread_count=2
            for index,frame in enumerate(con.decode(video=0)):
                yield index,float(frame.pts*frame.time_base),frame.to_ndarray(format='rgb24')


def source_identity(path):
    path=Path(path)
    if path.is_file():return dict(path=str(path),sha256=sha(path),kind='original_video')
    config=configparser.ConfigParser();config.read(path/'seqinfo.ini');cfg=config['Sequence']
    rows=[dict(file=f'{i+1:06d}{cfg["imExt"]}',sha256=sha(path/cfg['imDir']/f'{i+1:06d}{cfg["imExt"]}'))
          for i in range(int(cfg['seqLength']))]
    return dict(path=str(path),kind='original_JPEG_sequence',seqinfo_sha256=sha(path/'seqinfo.ini'),
                image_list_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
