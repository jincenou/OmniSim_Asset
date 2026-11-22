"""Official MOT16 archive; extract only three prescribed image sequences."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
STATE = dict(state='starting', pid=os.getpid(), started=time.time(), bytes=0)


def save():
    path = ROOT/'sources/mot_download_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(STATE, indent=2))
    tmp.replace(path)


async def main():
    url = 'https://motchallenge.net/data/MOT16.zip'
    target = ROOT/'raw/MOT16.zip'
    part = target.with_suffix('.zip.part')
    assert not target.exists()
    save()
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(90, connect=20),
                                    headers={'User-Agent':'WebHumanResearch/1.0'}) as client:
            page = await client.get('https://motchallenge.net/data/MOT16/')
            page.raise_for_status()
            (ROOT/'sources/MOT16_source_page.html').write_text(page.text)
            assert 'NonCommercial' in page.text or 'CC BY-NC-SA' in page.text
            async with client.stream('GET', url) as response:
                response.raise_for_status()
                STATE.update(state='downloading', url=str(response.url),
                             expected_bytes=int(response.headers.get('content-length',0)))
                digest = hashlib.sha256()
                last = 0
                with part.open('wb') as output:
                    async for chunk in response.aiter_bytes(1024**2):
                        output.write(chunk)
                        digest.update(chunk)
                        STATE['bytes'] += len(chunk)
                        if time.monotonic()-last > 10:
                            STATE['last_progress'] = time.time()
                            save()
                            last = time.monotonic()
            if STATE['expected_bytes']:
                assert STATE['bytes'] == STATE['expected_bytes']
        STATE['state'] = 'validating_zip'
        save()
        with zipfile.ZipFile(part) as archive:
            bad = await asyncio.to_thread(archive.testzip)
            assert bad is None, bad
            allowed = ('MOT16-09', 'MOT16-02', 'MOT16-10')
            members = [n for n in archive.namelist() if any(f'/{s}/' in '/'+n for s in allowed)
                       and ('/img1/' in n or n.endswith('/seqinfo.ini')) and not n.endswith('/')]
            assert members, 'No prescribed source images found'
            for name in members:
                dest = (ROOT/'raw/mot16'/name).resolve()
                assert dest.is_relative_to((ROOT/'raw/mot16').resolve())
            await asyncio.to_thread(archive.extractall, ROOT/'raw/mot16', members)
        part.replace(target)
        receipt = dict(source_page='https://motchallenge.net/data/MOT16/', download_url=url,
                       archive=str(target), sha256=digest.hexdigest(), size=target.stat().st_size,
                       license='CC BY-NC-SA 3.0; per-sequence third-party conditions may additionally apply',
                       license_url='https://creativecommons.org/licenses/by-nc-sa/3.0/',
                       extracted_sequences=list(allowed), extracted_files=len(members),
                       annotations_used=False, decoded=False, downloaded_at=time.time())
        (ROOT/'sources/MOT16.json').write_text(json.dumps(receipt,indent=2))
        STATE.update(state='download_and_zip_crc_verified_decode_pending', finished=time.time())
    except Exception as exc:
        STATE.update(state='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    asyncio.run(main())
