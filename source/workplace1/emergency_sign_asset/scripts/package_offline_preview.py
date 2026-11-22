"""Build downloadable previews without relying on server-side absolute paths."""
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path('/mnt/ojc/workplace1')
PAGE = ROOT / 'asset_library/previews/emergency_traffic_signs/index.html'
OUT = PAGE.parent
REPORT = ROOT / 'emergency_sign_asset/reports/implementation_report.html'
SINGLE = OUT / 'emergency_traffic_signs_offline.html'
ARCHIVE = OUT / 'emergency_traffic_signs_offline.zip'
MEDIA = {'.png', '.jpg', '.jpeg', '.webp', '.mp4'}
ATTR = re.compile(r'\b(href|src)=("|\')([^"\']*)\2')
ANCHOR = re.compile(r'<a\b([^>]*?)href=("|\')([^"\']*)\2([^>]*)>(.*?)</a>', re.S)


def local(value, page):
    value = html.unescape(value)
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = (page.parent / unquote(parsed.path)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Unexpected dependency outside workspace: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()


def standalone(content, page):
    def anchor(m):
        p = local(m[3], page)
        if p is None:
            return m[0]
        if p == REPORT:
            return '<a href="#offline-report">' + m[5] + '</a>'
        if page == REPORT and p.suffix == '.mp4':
            return '<span>' + m[5] + '（主要演示见上方；完整实验视频见 ZIP 包）</span>'
        if p.suffix in MEDIA | {'.json', '.txt', '.patch'}:
            return '<a href="' + data_url(p) + '" download="' + html.escape(p.name) + '">' + m[5] + '</a>'
        return '<span title="此文件包含在完整离线 ZIP 包中">' + m[5] + '（见完整离线包）</span>'
    content = ANCHOR.sub(anchor, content)
    def attr(m):
        if m[1] != 'src':
            return m[0]
        p = local(m[3], page)
        return 'src="' + data_url(p) + '"' if p else m[0]
    return ATTR.sub(attr, content)


original = PAGE.read_text()
original = re.sub(r'<div id="offline-downloads".*?</div>', '', original, flags=re.S)
single = standalone(original, PAGE)
report = REPORT.read_text().split('</style>', 1)[1].rsplit('</html>', 1)[0]
report = standalone(report, REPORT)
single = single.replace('<main>', '<main><p class="note">离线单文件版：图片和视频已内嵌，无需服务器。点击视频播放；PLY 等完整资产请使用配套 ZIP 包。页面中的厂家来源链接需要联网。</p>', 1)
single = single.replace('</main>', '<section id="offline-report">' + report + '</section></main>', 1)
SINGLE.write_text(single)


def archive_name(path):
    return 'index.html' if path == PAGE else 'resources/' + path.relative_to(ROOT).as_posix()


files = {PAGE}
queue = [PAGE]
html_contents = {}
while queue:
    page = queue.pop()
    content = original if page == PAGE else page.read_text()
    def relocate(m):
        p = local(m[3], page)
        if p is None:
            return m[0]
        if p not in files:
            files.add(p)
            if p.suffix == '.html':
                queue.append(p)
        target = os.path.relpath(archive_name(p), str(Path(archive_name(page)).parent))
        fragment = urlsplit(html.unescape(m[3])).fragment
        if fragment:
            target += '#' + fragment
        return m[1] + '="' + html.escape(target, quote=True) + '"'
    html_contents[page] = ATTR.sub(relocate, content)

# Preserve complete asset package directories so asset.json relative paths work.
catalogue = json.loads((ROOT / 'asset_library/manifests/asset_library_rigid.json').read_text())
for row in catalogue['assets']:
    if row.get('dataset') == 'emergency_traffic_signs':
        package = ROOT / 'asset_library' / row['package_path']
        files.update(p for p in package.rglob('*') if p.is_file())
files.add(OUT / 'catalogue.json')
files.update(p for p in (ROOT / 'asset_library/rigid/emergency_traffic_control').glob('*.json'))

manifest = []
with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr('打开说明.txt', '请先完整解压本 ZIP，再用 Chrome、Edge 或 Firefox 打开顶层 index.html。\n不要只提取 index.html，也不要在压缩软件预览窗口中打开。\n图片、视频及五件待审核 Gaussian 资产均已包含，不需要本地服务器。\n外部厂家来源链接仍需联网。资产元数据中的原始服务器绝对路径仅用于溯源。\n')
    for p in sorted(files):
        data = html_contents[p].encode() if p in html_contents else p.read_bytes()
        name = archive_name(p)
        z.writestr(name, data)
        manifest.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    z.writestr('offline_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []
    def handle_starttag(self, tag, attrs):
        self.values.extend(v for k, v in attrs if k in {'src', 'href'} and v)


parser = References()
parser.feed(single)
assert not [v for v in parser.values if not urlsplit(v).scheme and urlsplit(v).path], 'Single-file preview still has external file dependencies'
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    names = set(z.namelist())
    for name in names:
        if name.endswith('.html'):
            parser = References()
            parser.feed(z.read(name).decode())
            for value in parser.values:
                u = urlsplit(value)
                if not u.scheme and not u.netloc and u.path:
                    ref = os.path.normpath(str(Path(name).parent / unquote(u.path)))
                    assert ref in names, (name, ref)

banner = '<div id="offline-downloads" class="note"><strong>下载到自己电脑查看：</strong> <a href="emergency_traffic_signs_offline.html" download>单文件离线预览（含图片和视频）</a> · <a href="emergency_traffic_signs_offline.zip" download>完整离线包（含 Gaussian 资产）</a>。原始页面单独下载不包含资源。</div>'
PAGE.write_text(original.replace('<main>', '<main>' + banner, 1))
result = dict(single_file=str(SINGLE), single_file_bytes=SINGLE.stat().st_size, archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size, archived_files=len(manifest), standalone_has_no_local_dependencies=True, archive_links_and_crc_valid=True)
(OUT / 'offline_validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
