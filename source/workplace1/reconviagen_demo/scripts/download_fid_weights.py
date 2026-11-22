from pathlib import Path
import subprocess,concurrent.futures,hashlib,os
root=Path('/mnt/ojc/workplace1/reconviagen_demo/cache/fid');url='https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
env=os.environ.copy();env.update(HTTPS_PROXY='',HTTP_PROXY='',ALL_PROXY='')
loc=subprocess.check_output(['curl','-sS','-I','-L','--max-time','40','-o','/dev/null','-w','%{url_effective}',url],env=env,text=True)
total=95628359;n=24;chunk=(total+n-1)//n;(root/'parts').mkdir(exist_ok=True)
def get(i):
 lo=i*chunk;hi=min(total,(i+1)*chunk)-1;p=root/'parts'/f'{i:03}'
 if p.exists() and p.stat().st_size==hi-lo+1:return p
 subprocess.run(['curl','-sS','-L','--fail','--retry','2','--max-time','240','--range',f'{lo}-{hi}',loc+'&part='+str(i),'-o',str(p)],env=env,check=True)
 assert p.stat().st_size==hi-lo+1,(i,p.stat().st_size);print('PART',i,flush=True);return p
with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:parts=list(ex.map(get,range(n)))
data=b''.join(p.read_bytes() for p in parts);digest=hashlib.sha256(data).hexdigest();assert digest.startswith('6726825d'),digest
(root/'verified.pth').write_bytes(data);print('VERIFIED',digest,flush=True)
