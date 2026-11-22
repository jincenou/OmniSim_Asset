"""Exercise GPU flat-L2 queries at the actual failing size and larger, against exhaustive distances."""
import argparse,json,time,os
import numpy as np
import torch,faiss
from common import ROOT,save,device,stamp

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--name',required=True);args=cli.parse_args()
    gpu=device();rng=np.random.default_rng(42);rows=[];began=time.monotonic()
    for n,q in [(3105,42),(10000,256)]:
        data=rng.normal(size=(n,3)).astype(np.float32);query=rng.normal(size=(q,3)).astype(np.float32)
        resource=faiss.StandardGpuResources();index=faiss.GpuIndexFlatL2(resource,3);index.add(data)
        print(json.dumps({'state':'searching','database':n,'queries':q,'faiss':faiss.__version__,'module':faiss.__file__}),flush=True)
        distance,ids=index.search(query,1)
        exact=((query[:,None].astype(np.float64)-data[None].astype(np.float64))**2).sum(-1)
        expected=exact.argmin(1);error=float(abs(distance[:,0]-exact[np.arange(q),expected]).max())
        assert np.array_equal(ids[:,0],expected) and error<1e-4,(n,q,error)
        rows.append({'database':n,'queries':q,'dimension':3,'identical_neighbor_indices':True,'max_distance_error':error})
    result={'state':'passed','checked_at':stamp(),'device':gpu,'faiss_version':faiss.__version__,
        'module':faiss.__file__,'torch':torch.__version__,'rows':rows,'elapsed_seconds':time.monotonic()-began}
    save(ROOT/'research'/f'faiss_probe_{args.name}.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
