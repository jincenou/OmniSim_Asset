"""Download the exact official README-linked RIFE 4.25 archive."""
import asyncio
import hashlib
import json
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
URL='https://drive.usercontent.google.com/download?id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download&authuser=0&confirm=t'


async def main():
    dest=ROOT/'rife_weights.zip'
    sha=hashlib.sha256();size=0
    async with httpx.AsyncClient(follow_redirects=True,timeout=httpx.Timeout(60,read=60)) as client:
        async with client.stream('GET',URL) as response:
            response.raise_for_status()
            if 'text/html' in response.headers.get('content-type',''):
                raise RuntimeError('Google Drive returned HTML, not model archive')
            with dest.with_suffix('.part').open('wb') as out:
                async for chunk in response.aiter_bytes(1024*1024):
                    out.write(chunk);sha.update(chunk);size+=len(chunk)
                    if size%(10*1024*1024)<len(chunk):print(f'downloaded {size} bytes',flush=True)
    dest.with_suffix('.part').rename(dest)
    record={'source_readme':'https://github.com/hzwer/Practical-RIFE','official_archive_id':'1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg','filename':'RIFEv4.25_0919.zip','bytes':size,'observed_sha256':sha.hexdigest(),'checksum_scope':'Locally recorded hash; upstream checksum not supplied.'}
    (ROOT/'research/rife-download.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':asyncio.run(main())
