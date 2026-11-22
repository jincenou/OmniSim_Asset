"""One-shot, explicit legacy archive/index cleanup; never prune model assets."""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

PROJECT = Path('/mnt/ojc/workplace1/4danyone')
OUT = Path(__file__).resolve().parents[1] / 'research' / 'legacy_cleanup_20261021'


def sha(stream):
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return sha(stream)


def save(name, data):
    target = OUT / name
    temporary = target.with_suffix('.writing')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(target)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    specs = [
        ('waymo/rife_weights.zip', 'waymo/rife_model',
         ['train_log/flownet.pkl', 'train_log/IFNet_HDv3.py', 'train_log/refine.py', 'train_log/RIFE_HDv3.py']),
        ('waymo/frontend_quality_20261020/models/rohm_checkpoints.zip',
         'waymo/frontend_quality_20261020/models/rohm', None),
        ('waymo/frontend_general_v1/text_motion/condmdi_random_joints.zip',
         'waymo/frontend_general_v1/text_motion/weights', ['args.json', 'model000750000.pt']),
    ]
    candidates = []
    preserved = []
    for archive, destination, chosen in specs:
        path = PROJECT / archive
        before = path.stat()
        verified = []
        with zipfile.ZipFile(path) as z:
            members = chosen or [i.filename for i in z.infolist() if not i.is_dir()]
            for member in members:
                target = PROJECT / destination / member
                info = z.getinfo(member)
                assert target.is_file() and target.stat().st_size == info.file_size, target
                with z.open(member) as stream:
                    digest = sha(stream)
                assert file_sha(target) == digest, target
                row = dict(path=str(target), bytes=info.file_size, sha256=digest)
                preserved.append(row)
                verified.append(row)
            unused = [i.filename for i in z.infolist() if not i.is_dir() and i.filename not in members]
        candidates.append(dict(path=str(path), bytes=before.st_size, inode=before.st_ino,
                               mtime_ns=before.st_mtime_ns, blocks=before.st_blocks,
                               sha256=file_sha(path), reason='Runtime payload matches extracted files byte-for-byte',
                               preserved=verified, unused_archive_members=unused))
    # These exact two directories contain only stale package-index metadata,
    # not extracted packages, CUDA/JIT builds or model weights.
    for relative in ['cache/conda/cache', 'eff4d/cache/conda/cache']:
        for path in sorted((PROJECT / relative).iterdir()):
            assert path.is_file() and not path.is_symlink()
            assert path.suffix in {'.json', '.tmp'}
            s = path.stat()
            assert time.time() - s.st_mtime > 86400
            candidates.append(dict(path=str(path), bytes=s.st_size, inode=s.st_ino,
                                   mtime_ns=s.st_mtime_ns, blocks=s.st_blocks,
                                   sha256=file_sha(path), reason='Stale conda repository index or interrupted index transfer'))
    targets = {Path(r['path']) for r in candidates}
    protected = {Path(s) for s in json.loads((PROJECT / 'cleanup_protected_symlink_targets.json').read_text())}
    assert not targets.intersection(protected)
    # Resolve current project links, including links into a parent directory.
    for parent, dirs, files in os.walk(PROJECT, followlinks=False):
        for name in dirs + files:
            link = Path(parent) / name
            if link.is_symlink():
                resolved = link.resolve()
                assert not any(t == resolved or resolved in t.parents for t in targets), link
    save('plan.json', dict(created_at=time.time(), candidates=candidates,
                          preserved=preserved, free_bytes_before=shutil.disk_usage(PROJECT).free,
                          policy='Explicit archives and package index only. No checkpoints, diagnostics, inputs, source, media or runtime files.'))
    # Host /proc inspection immediately before mutation. Refuse if any current
    # process holds one of the exact candidate inodes or names an archive.
    busy = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(str(t) in command for t in targets):
                busy.append(dict(pid=proc.name, kind='command'))
            for fd in (proc / 'fd').iterdir():
                try:
                    if fd.resolve() in targets:
                        busy.append(dict(pid=proc.name, kind='open_fd'))
                except (OSError, RuntimeError):
                    pass
        except (OSError, PermissionError):
            continue
    assert not busy, busy
    receipt = dict(started_at=time.time(), deleted=[], retained_payloads=preserved)
    save('receipt.json', receipt)
    for row in candidates:
        path = Path(row['path'])
        s = path.stat()
        assert (s.st_ino, s.st_size, s.st_mtime_ns) == (row['inode'], row['bytes'], row['mtime_ns'])
        assert not path.is_symlink() and path.is_relative_to(PROJECT)
        path.unlink()
        receipt['deleted'].append(row)
        save('receipt.json', receipt)
    for row in preserved:
        assert file_sha(Path(row['path'])) == row['sha256']
    receipt.update(finished_at=time.time(), verified=True,
                   deleted_bytes=sum(x['bytes'] for x in receipt['deleted']),
                   released_blocks_bytes=sum(x['blocks'] * 512 for x in receipt['deleted']),
                   free_bytes_after=shutil.disk_usage(PROJECT).free)
    save('receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['verified', 'deleted_bytes', 'released_blocks_bytes', 'free_bytes_after']}))


if __name__ == '__main__':
    main()
