"""Copy completed first-run JIT cache into the project's cache, without changing the source."""
import fcntl, hashlib, json, os, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    destination=Path(os.environ['CUDA_CACHE_PATH'])
    assert destination==ROOT/'cache/cuda'
    destination.mkdir(parents=True,exist_ok=True)
    receipt=ROOT/'research/cuda_cache_reuse.json'
    with (ROOT/'cache/cuda-reuse.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            return
        assert json.loads((ROOT/'research/preflight.json').read_text())['state']=='passed'
        source=Path.home()/'.nv/ComputeCache'
        copied=[]
        for path in sorted(source.rglob('*')):
            if not path.is_file():
                continue
            before=path.stat();payload=path.read_bytes();after=path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(path)
            target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists(),str(target)
            temporary=target.with_name(target.name+'.copying');temporary.write_bytes(payload)
            shutil.copystat(path,temporary);temporary.replace(target)
            copied.append({'file':str(path.relative_to(source)),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        assert copied,'Expected the validated first-run CUDA cache to exist'
        receipt.write_text(json.dumps({'source':str(source),'destination':str(destination),
            'source_modified':False,'bytes':sum(x['bytes'] for x in copied),'files':copied},indent=2)+'\n')
        print(json.dumps({'cuda_cache_reused_files':len(copied),'bytes':sum(x['bytes'] for x in copied)}),flush=True)

if __name__=='__main__':main()
