"""Finite Commons acquisition, two concurrent transfers, original files only."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TITLES = [
    'File:Piedira pasejo en strato Respubliko (Tjumeno).webm',
    'File:Piedirantoj ĉe kruciĝo de stratoj Respubliko kaj Orĝonikidze (Tjumeno).webm',
    'File:Piedirejo en strato Herzen (Tjumeno).webm',
]
STATE = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'sources': {}}


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


async def main():
    sem = asyncio.Semaphore(1)
    headers = {'User-Agent': 'WebHumanResearch/1.0 (local academic qualitative video study)'}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True,
                                timeout=httpx.Timeout(60, connect=20)) as client:
        async def one(index, title):
            key = f'C{index:02d}'
            async with sem:
                STATE['sources'][key] = {'title': title, 'state': 'metadata'}
                save(ROOT/'sources/supplement_download_status.json', STATE)
                try:
                    response = await client.get('https://commons.wikimedia.org/w/api.php', params={
                        'action': 'query', 'format': 'json', 'formatversion': 2,
                        'prop': 'imageinfo', 'redirects': 1,
                        'iiprop': 'url|size|mime|sha1|extmetadata', 'titles': title})
                    response.raise_for_status()
                    metadata = response.json()
                    save(ROOT/f'sources/{key}_imageinfo.json', metadata)
                    info = metadata['query']['pages'][0]['imageinfo'][0]
                    ext = info['extmetadata']
                    assert ext.get('LicenseShortName', {}).get('value'), 'Missing license'
                    assert info['mime'].startswith('video/'), info['mime']
                    assert info['size'] < 1024**3, 'Unexpected size; inspect before downloading'
                    target = ROOT/f'raw/{key}.webm'
                    assert not target.exists(), 'Do not overwrite source'
                    part = target.with_suffix('.webm.part')
                    STATE['sources'][key].update(state='downloading', expected_bytes=info['size'],
                                                  downloaded_bytes=0, url=info['url'])
                    # A failed attempt restarts from zero; never blindly append a Range response.
                    async def transfer(attempt=0):
                        try:
                            digest = hashlib.sha1()
                            total = 0
                            last = 0
                            async with client.stream('GET', info['url']) as stream:
                                stream.raise_for_status()
                                with part.open('wb') as output:
                                    async for chunk in stream.aiter_bytes(1024**2):
                                        output.write(chunk)
                                        digest.update(chunk)
                                        total += len(chunk)
                                        if time.monotonic() - last > 5:
                                            STATE['sources'][key].update(downloaded_bytes=total,
                                                                        last_progress=time.time())
                                            save(ROOT/'sources/supplement_download_status.json', STATE)
                                            last = time.monotonic()
                            assert total == info['size'], (total, info['size'])
                            assert digest.hexdigest() == info['sha1'], 'MediaWiki SHA1 differs'
                            part.replace(target)
                        except (httpx.TimeoutException, httpx.TransportError):
                            if attempt >= 2:
                                raise
                            STATE['sources'][key]['network_retries'] = attempt + 1
                            await asyncio.sleep(2**(attempt+1))
                            await transfer(attempt+1)
                    await transfer()
                    record = dict(id=key, title=title, source_page=info['descriptionurl'],
                                  download_url=info['url'], path=str(target), downloaded_at=time.time(),
                                  size=target.stat().st_size, sha1=info['sha1'],
                                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                  author=ext.get('Artist', {}).get('value'),
                                  license=ext['LicenseShortName']['value'],
                                  license_url=ext.get('LicenseUrl', {}).get('value'),
                                  metadata=info, decoded=False, selected=False)
                    save(ROOT/f'sources/{key}.json', record)
                    STATE['sources'][key].update(state='download_verified_decode_pending',
                                                  downloaded_bytes=target.stat().st_size,
                                                  finished=time.time())
                    print(key, 'download verified', target.stat().st_size, flush=True)
                except Exception as exc:
                    STATE['sources'][key].update(state='failed', error=repr(exc), finished=time.time())
                    print(key, repr(exc), flush=True)
                save(ROOT/'sources/supplement_download_status.json', STATE)
        await asyncio.gather(*(one(i+5, title) for i, title in enumerate(TITLES)))
    records = [json.loads(p.read_text()) for p in sorted((ROOT/'sources').glob('C[0-9][0-9].json'))]
    (ROOT/'sources/manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (ROOT/'sources/ATTRIBUTION.md').write_text('# Source attribution\n\n' + '\n\n'.join(
        f"{r['id']}: [{r['title']}]({r['source_page']}) — {r['author']} — {r['license']} ({r['license_url']})." for r in records))
    STATE.update(state='finished_requires_decode_and_selection', finished=time.time())
    save(ROOT/'sources/supplement_download_status.json', STATE)


if __name__ == '__main__':
    asyncio.run(main())
