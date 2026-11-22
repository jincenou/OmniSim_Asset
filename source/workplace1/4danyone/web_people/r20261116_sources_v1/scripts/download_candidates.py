"""Bounded downloads from reviewed official links; no discovery or bulk crawling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import datetime

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ('P13518524', 'https://videos.pexels.com/video-files/13518524/13518524-hd_1920_1080_30fps.mp4', 'https://www.pexels.com/video/a-man-walking-on-the-street-13518524/', 'Pexels License', None),
    ('P8480544', 'https://videos.pexels.com/video-files/8480544/8480544-hd_1080_1920_25fps.mp4', 'https://www.pexels.com/video/a-man-raising-his-arms-and-putting-his-hands-together-8480544/', 'Pexels License', 'Yan Krukau'),
    ('C_ARAKAWA', 'https://upload.wikimedia.org/wikipedia/commons/1/1b/Jogging_-_near_arakawa_river_-_tokyo_japan_-_2022_may_3.webm', 'https://commons.wikimedia.org/wiki/File:Jogging_-_near_arakawa_river_-_tokyo_japan_-_2022_may_3.webm', 'CC BY 4.0', 'Nesnad'),
    ('UBC01', 'https://m.media-amazon.com/images/I/91D23ZVV6NS.mp4', 'https://vision.cs.ubc.ca/datasets/fashion/', 'Per-video rights not stated in official list; research screening only, publication pending', None),
    ('UBC02', 'https://m.media-amazon.com/images/I/91hECpvrRSS.mp4', 'https://vision.cs.ubc.ca/datasets/fashion/', 'Per-video rights not stated in official list; research screening only, publication pending', None),
]

def download(row):
    ident, url, page, license, author = row
    dst = ROOT / 'raw' / (ident + ('.webm' if '.webm' in url else '.mp4'))
    record = dict(source_id=ident, download_url=url, source_page=page, license=license,
                  author=author, upstream_example=False, path=str(dst),
                  checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  accepted=False, selected_interval=None)
    try:
        if not dst.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'WebHumanResearch/1.0'})
            with urllib.request.urlopen(req, timeout=45) as response, dst.with_suffix(dst.suffix+'.part').open('wb') as out:
                record['resolved_url'] = response.url
                record['content_type'] = response.headers.get('content-type')
                count = 0
                while chunk := response.read(1024*1024):
                    count += len(chunk)
                    if count > 350*1024**2 or shutil.disk_usage(ROOT).free < 11*1024**3:
                        raise RuntimeError('Download size or persistent disk reserve exceeded')
                    out.write(chunk)
            dst.with_suffix(dst.suffix+'.part').replace(dst)
        record.update(status='downloaded_unreviewed', bytes=dst.stat().st_size,
                      sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    except Exception as exc:
        record.update(status='download_failed', error=str(exc))
    (ROOT/'sources'/f'{ident}_download.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(ident, record['status'], record.get('bytes', record.get('error')), flush=True)
    return record

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, CANDIDATES))
    (ROOT/'source_provenance.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
