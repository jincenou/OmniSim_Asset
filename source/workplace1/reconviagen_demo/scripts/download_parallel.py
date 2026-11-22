"""Resume pinned checkpoints through mirror in verified byte ranges."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,hashlib,time,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[1]
proxy=os.environ.get('RECON_DOWNLOAD_PROXY')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy,'http':proxy} if proxy else {}))

def download(repo,rev,item):
    path=ROOT/'cache'/repo.replace('/','--')/item['rfilename'];path.parent.mkdir(parents=True,exist_ok=True)
    size=item['size'];sha=item.get('lfs',{}).get('sha256')
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    if path.exists() and path.stat().st_size==size and (not sha or digest(path)==sha):return
    part=path.with_name(path.name+'.partial');state=path.with_name(path.name+'.ranges.json')
    host=os.environ.get('RECON_DOWNLOAD_HOST','hf-mirror.com')
    url=f'https://{host}/{repo}/resolve/{rev}/{item["rfilename"]}'
    print('FETCH',repo,item['rfilename'],size,flush=True)
    if size<1024**2:
        for attempt in range(5):
            try:
                with opener.open(url,timeout=60) as r:part.write_bytes(r.read())
                break
            except urllib.error.HTTPError as error:
                if error.code!=429 or attempt==4:raise
                time.sleep(min(60,max(5,int(error.headers.get('Retry-After','60')))))
    else:
        if state.exists():st=json.loads(state.read_text())
        else:
            prefix=part.stat().st_size if part.exists() else 0
            st=dict(prefix=prefix,done=[])
            state.write_text(json.dumps(st))
        chunks=[(start,min(size-1,start+32*1024**2-1)) for start in range(st['prefix'],size,32*1024**2)]
        # Resolve the signed object URL once, so each byte range does not hit
        # the model-host resolver API again. Refresh on a later resumed run.
        with opener.open(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
            object_url=response.geturl()
        fd=os.open(part,os.O_CREAT|os.O_RDWR,0o664)
        def fetch(chunk):
            start,end=chunk
            if start in st['done']:return start
            for attempt in range(5):
                try:
                    request=urllib.request.Request(object_url,headers={'Range':f'bytes={start}-{end}'})
                    with opener.open(request,timeout=90) as r:
                        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                            raise ValueError(f'Invalid range response: {r.status} {r.headers.get("Content-Range")}')
                        offset=start
                        while b:=r.read(1024**2):
                            if offset+len(b)>end+1:raise ValueError('Range overflow')
                            os.pwrite(fd,b,offset);offset+=len(b)
                        if offset!=end+1:raise ValueError('Incomplete range')
                    return start
                except Exception:
                    if attempt==4:raise
                    time.sleep(2**attempt)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for start in pool.map(fetch,chunks):
                    if start not in st['done']:st['done'].append(start)
                    state.write_text(json.dumps(st))
                    print(repo,item['rfilename'],'completed_chunks',len(st['done']), '/',len(chunks),flush=True)
        finally:os.close(fd)
    if part.stat().st_size!=size or (sha and digest(part)!=sha):raise ValueError(f'Integrity failure {path}')
    part.replace(path)
    print('VERIFIED',path,flush=True)

if __name__=='__main__':
    results=[]
    for repo in ['Stable-X/trellis-vggt-v0-2','Stable-X/vggt-object-v0-1','ZhengPeng7/BiRefNet']:
        meta=json.loads((ROOT/'docs'/(repo.replace('/','--')+'.json')).read_text())
        for item in meta['siblings']:
            if item['rfilename'].endswith(('.json','.safetensors','.py')) and item['rfilename']!='handler.py':
                download(repo,meta['sha'],item);results.append(dict(repo=repo,revision=meta['sha'],**item))
    (ROOT/'docs/weights_downloaded.json').write_text(json.dumps(results,indent=2))
