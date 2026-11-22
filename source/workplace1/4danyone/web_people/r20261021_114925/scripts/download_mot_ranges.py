"""Fetch only prescribed ZIP members through verified official HTTP byte ranges."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import zlib
import httpx

ROOT=Path(__file__).resolve().parents[1]
URL='https://motchallenge.net/data/MOT16.zip'
SIZE=1954509127
ETAG='"747f7147-59ed65f64a040"'
STATE=dict(state='starting',pid=os.getpid(),started=time.time(),chunks={},sequences={})


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def publish():
    STATE['updated']=time.time()
    save(ROOT/'sources/mot_ranges_status.json',STATE)


async def main():
    entries=json.loads((ROOT/'sources/mot_selected_zip_entries.json').read_text())
    chunks=[]
    # Each range includes the maximum ZIP local extra field after the last member.
    for entry in sorted(entries,key=lambda row:row['offset']):
        start=entry['offset'];end=start+30+len(entry['path'].encode())+65535+entry['compressed_size']-1
        assert end<SIZE
        if chunks and start<=chunks[-1]['end']+65536 and end-chunks[-1]['start']<16*1024**2:
            chunks[-1]['end']=max(chunks[-1]['end'],end);chunks[-1]['entries'].append(entry)
        else:chunks.append(dict(start=start,end=end,entries=[entry]))
    cache=ROOT/'raw/mot_ranges';cache.mkdir(exist_ok=True)
    STATE['expected_range_bytes']=sum(c['end']-c['start']+1 for c in chunks)
    STATE['selected_uncompressed_bytes']=sum(e['size'] for e in entries)
    STATE['state']='downloading_verified_ranges';publish()
    sem=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=httpx.Timeout(90,connect=20),
            headers={'User-Agent':'WebHumanResearch/1.0','Accept-Encoding':'identity'}) as client:
        async def one(index,chunk):
            async with sem:
                a,b=chunk['start'],chunk['end'];key=f'{index:03d}'
                target=cache/f'{key}.bin';part=target.with_suffix('.part')
                assert not target.exists(),target
                STATE['chunks'][key]=dict(state='downloading',bytes=0,expected=b-a+1);publish()
                async def transfer(attempt=0):
                    try:
                        async with client.stream('GET',URL,headers={'Range':f'bytes={a}-{b}','If-Range':ETAG}) as response:
                            assert response.status_code==206,response.status_code
                            assert response.headers['content-range']==f'bytes {a}-{b}/{SIZE}'
                            assert response.headers.get('etag')==ETAG
                            total=0;last=0
                            with part.open('wb') as f:
                                async for blob in response.aiter_bytes(1024**2):
                                    f.write(blob);total+=len(blob)
                                    if time.monotonic()-last>10:
                                        STATE['chunks'][key].update(bytes=total,last_progress=time.time());publish();last=time.monotonic()
                            assert total==b-a+1
                            part.replace(target)
                    except (httpx.TimeoutException,httpx.TransportError):
                        if attempt>=2:raise
                        STATE['chunks'][key]['network_retries']=attempt+1;publish()
                        await asyncio.sleep(2**(attempt+1));await transfer(attempt+1)
                await transfer()
                buffer=target.read_bytes();checks=[]
                for entry in chunk['entries']:
                    offset=entry['offset']-a
                    fields=struct.unpack('<4s5H3L2H',buffer[offset:offset+30])
                    assert fields[0]==b'PK\x03\x04' and not fields[2]&1
                    assert fields[3]==entry['compression']
                    name=buffer[offset+30:offset+30+fields[-2]].decode('utf-8' if fields[2]&2048 else 'cp437')
                    assert name==entry['path']
                    start=offset+30+fields[-2]+fields[-1]
                    compressed=buffer[start:start+entry['compressed_size']]
                    assert len(compressed)==entry['compressed_size']
                    assert entry['compression'] in (0,8)
                    data=compressed if entry['compression']==0 else zlib.decompress(compressed,-15)
                    assert len(data)==entry['size'] and zlib.crc32(data)&0xffffffff==entry['crc32']
                    path=(ROOT/'raw/mot16'/name).resolve()
                    assert path.is_relative_to((ROOT/'raw/mot16').resolve())
                    path.parent.mkdir(parents=True,exist_ok=True)
                    assert not path.exists(),path
                    temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
                    checks.append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest(),
                                       official_zip_crc32=entry['crc32'],sequence=entry['sequence']))
                save(ROOT/'sources'/f'mot_range_{key}_receipt.json',dict(start=a,end=b,
                    range_sha256=hashlib.sha256(buffer).hexdigest(),files=checks))
                target.unlink()
                STATE['chunks'][key].update(state='extracted_crc_verified',bytes=b-a+1,files=len(checks));publish()
                print(key,len(checks),'files verified',flush=True)
        results=await asyncio.gather(*(one(i,c) for i,c in enumerate(chunks)),return_exceptions=True)
    failures={str(i):repr(x) for i,x in enumerate(results) if isinstance(x,BaseException)}
    all_records=[]
    for path in sorted((ROOT/'sources').glob('mot_range_*_receipt.json')):
        all_records.extend(json.loads(path.read_text())['files'])
    for seq in ['MOT16-09','MOT16-02','MOT16-10']:
        rows=[r for r in all_records if r['sequence']==seq]
        expected=sum(e['sequence']==seq for e in entries)
        STATE['sequences'][seq]=dict(expected_files=expected,verified_files=len(rows),complete=len(rows)==expected)
    save(ROOT/'sources/MOT16.json',dict(source_page='https://motchallenge.net/data/MOT16/',
        download_url=URL,license='CC BY-NC-SA 3.0; additional third-party conditions may apply',
        license_url='https://creativecommons.org/licenses/by-nc-sa/3.0/',archive_etag=ETAG,
        archive_size=SIZE,archive_sha256=None,transport='verified HTTP206 selected ZIP member ranges',
        official_central_directory_sha256=hashlib.sha256((ROOT/'sources/mot_zip_central_directory.bin').read_bytes()).hexdigest(),
        files=all_records,annotations_used=False,decoded=False,downloaded_at=time.time()))
    STATE.update(state='failed' if failures else 'selected_sequences_crc_verified_decode_pending',failures=failures,finished=time.time());publish()
    if failures:raise RuntimeError(failures)


if __name__=='__main__':asyncio.run(main())
