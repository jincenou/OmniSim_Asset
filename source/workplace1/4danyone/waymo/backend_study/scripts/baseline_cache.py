"""Re-render immutable HUST checkpoint to a temporary lossless cache for paired diagnostics."""
from pathlib import Path
from argparse import ArgumentParser
import ctypes,hashlib,json,os,sys,uuid
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'random8/r20261015_113332'

@torch.inference_mode()
def main():
    parser=ArgumentParser();parser.add_argument('--label',required=True);arg=parser.parse_args()
    assert arg.label in ['N01','N02','N03','N04','N06','N07','N08','N11']
    manifest_path=BASE/'four_dg/data'/arg.label/'fdanyone.json';data=json.loads(manifest_path.read_text())
    os.environ['FDANYONE_FRAMES']=str(data['frame_count'])
    sys.path.insert(0,str(BASE/'scripts'))
    from evaluate_branch import HUST
    expected=os.environ['CUDA_VISIBLE_DEVICES'];assert expected.startswith('GPU-')
    assert torch.cuda.device_count()==1
    torch.empty(1,device='cuda');driver=ctypes.CDLL('libcuda.so.1');ordinal=ctypes.c_int()
    assert driver.cuCtxGetDevice(ctypes.byref(ordinal))==0
    value=(ctypes.c_ubyte*16)();assert driver.cuDeviceGetUuid(ctypes.byref(value),ordinal.value)==0
    assert 'GPU-'+str(uuid.UUID(bytes=bytes(value)))==expected
    out=ROOT/'tmp'/f'{arg.label}_hust_raw';out.mkdir(exist_ok=False,parents=True)
    model=HUST(BASE/'samples'/arg.label,'4dgs_grid61',data);model.validate()
    for view in [0,8,16]:
        rgb=[];alpha=[]
        for frame in range(data['frame_count']):
            r,a=model.render(view,frame);assert np.isfinite(r).all() and np.isfinite(a).all()
            rgb.append(r);alpha.append(a)
        np.savez_compressed(out/f'view{view:02d}.npz',rgb=np.stack(rgb),alpha=np.stack(alpha))
        print(json.dumps({'label':arg.label,'view':view,'frames':len(rgb)}),flush=True)
    record={'state':'raw_rerendered','label':arg.label,'frame_count':data['frame_count'],
        'asset':model.asset(),'device_uuid':expected,'pid':os.getpid(),
        'purpose':'temporary_lossless_raw_for_fair_comparison_delete_after_final_media_and_numeric_validation'}
    (out/'complete.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
