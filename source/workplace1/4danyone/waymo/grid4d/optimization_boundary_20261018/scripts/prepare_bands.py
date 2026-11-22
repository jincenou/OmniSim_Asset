"""CPU-only immutable training partition cache; no heldout images are opened."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from boundary_loss import DISK, regions

NEW = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True,choices=['N04','N11']);args=parser.parse_args()
    begin=time.monotonic();plan=json.loads((NEW/'PLAN.json').read_text());assert sha(NEW/'PLAN.json')==(NEW/'plan_hash.txt').read_text().strip()
    item=plan['identities'][args.label];directory=NEW/'data'/args.label/'bands';directory.mkdir(exist_ok=True)
    assert not (directory/'complete.json').exists(), 'Already prepared; verify and reuse'
    source=Path(item['data']['training']);assert sha(source)==item['data']['training_sha256'];rows=json.loads(source.read_text())
    views=sorted({r['view'] for r in rows});assert len(views)==21 and not set(views)&{0,8,16}
    cache={v:np.lib.format.open_memmap(directory/f'view{v:02d}.npy',mode='w+',dtype=np.uint8,shape=(item['frames'],640,352)) for v in views}
    counts=[]
    for row in rows:
        p=Path(row['path'])
        with Image.open(p) as image:
            assert image.mode=='RGBA' and image.size==(352,640);alpha=np.asarray(image)[...,3].copy()
        part=regions(alpha.astype(np.float32)/255);cache[row['view']][row['frame']]=part
        sizes=np.bincount(part.flatten(),minlength=4).tolist()
        assert np.array_equal((part==1)|(part==2),alpha>=128)
        counts.append(dict(**row,sha256=sha(p),areas=sizes,added_term_valid=bool(sizes[2] and sizes[3])))
    for array in cache.values():array.flush()
    files={str(directory/f'view{v:02d}.npy'):sha(directory/f'view{v:02d}.npy') for v in views}
    output=dict(state='passed',label=args.label,plan_sha256=sha(NEW/'PLAN.json'),source_list_sha256=sha(source),definition=plan['regions'],disk=DISK.tolist(),files=files,images=counts,
        empty_band_count=sum(not r['added_term_valid'] for r in counts),heldout_content_reads=0,seconds=time.monotonic()-begin,pid=os.getpid())
    (directory/'complete.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['images','files','disk','definition']},allow_nan=False),flush=True)


if __name__=='__main__':main()
