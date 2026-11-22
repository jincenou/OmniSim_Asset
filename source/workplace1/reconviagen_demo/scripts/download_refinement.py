"""Fetch named official refinement dependencies; keep provenance and hashes."""
from pathlib import Path
import urllib.request,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1]
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def fetch(name,url,size=None):
    dst=ROOT/'cache/refinement'/name;dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():return
    part=dst.with_suffix(dst.suffix+'.partial')
    for attempt in range(4):
        try:
            offset=part.stat().st_size if part.exists() else 0
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with op.open(req,timeout=60) as response:
                mode='ab' if offset and response.status==206 else 'wb'
                if mode=='wb':offset=0
                expected=offset+int(response.headers['Content-Length'])
                if size is not None and expected!=size:raise ValueError('Unexpected length')
                with part.open(mode) as f:
                    last=time.monotonic()
                    while b:=response.read(4*1024**2):
                        f.write(b)
                        if time.monotonic()-last>30:print(name,f.tell(),'/',expected,flush=True);last=time.monotonic()
            if part.stat().st_size!=expected:raise ValueError('Incomplete download')
            break
        except Exception:
            if attempt==3:raise
            time.sleep(5)
    h=hashlib.sha256()
    with part.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    part.replace(dst)
    dst.with_suffix(dst.suffix+'.provenance.json').write_text(json.dumps(dict(url=url,bytes=expected,
        sha256=h.hexdigest(),hash_kind='computed local content hash; upstream digest not provided'),indent=2))
    print('COMPLETE',name,expected,flush=True)

if __name__=='__main__':
    fetch('MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
          'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',2754910614)
    fetch('dreamsim_dino_vitb16_checkpoint.zip',
          'https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_dino_vitb16_checkpoint.zip')
