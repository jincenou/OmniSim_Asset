import os,sys,json,hashlib
from pathlib import Path
import rembg
from PIL import Image
import numpy as np
os.environ['U2NET_HOME']='/mnt/wmj/workspace4/pipeline'
p=Path(sys.argv[1]);im=Image.open(p).convert('RGB');session=rembg.new_session('birefnet-general-lite',providers=['CPUExecutionProvider']);rgba=rembg.remove(im,session=session);rgba.save(p.with_name('flux_rgba.png'))
white=Image.new('RGB',im.size,'white');white.paste(rgba,mask=rgba.getchannel('A'));white.save(p.with_name('flux_matte_preview.jpg'))
print('MATTE_DONE',np.count_nonzero(np.asarray(rgba.getchannel('A'))>204),flush=True)
