"""Portable DriveLab rigid publication core (also installed in the producer).

No Hasim or reconstruction-model imports. Administrative receipts stay outside
the portable release; publication never promotes an unresolved asset to keep.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import tempfile

import numpy as np
from plyfile import PlyData

DIRECTIONAL = {'car', 'truck', 'bus', 'van', 'vehicle', 'bicycle', 'motorcycle',
               'traffic_sign', 'traffic_light'}
VEHICLES = {'car', 'truck', 'bus', 'van', 'vehicle', 'bicycle', 'motorcycle'}


class PendingPublication(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            os.fchmod(f.fileno(), (path.stat().st_mode & 0o777) if path.exists() else 0o644)
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        except BaseException:
            temp.unlink(); raise
    os.replace(temp, path)


def _positive(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape:
        raise PendingPublication('invalid numeric shape or type')
    a = a.astype(float)
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise PendingPublication('nonpositive/nonfinite dimension or scale')
    return a


def entry_from_metadata(meta, *, binding=None, placement=None, override=None):
    """Return the nine HF fields and scale evidence. Never guess an axis swap."""
    override = override or {}
    identity = meta.get('global_uid') or meta.get('uid')
    category = meta.get('category'); dataset = meta.get('dataset', 'unknown')
    if not isinstance(identity, str) or not identity:
        raise PendingPublication('missing identity')
    if any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', x)
           for x in (category, dataset)):
        raise ValueError('invalid category/dataset path component')
    if category == 'human' or meta.get('status') != 'keep':
        raise PendingPublication('only reviewed keep rigid assets may enter release')
    c = meta.get('canonical', {})
    if c.get('up') != '+Z' or c.get('metric_scale_baked') is not False:
        raise PendingPublication('canonical Z-up/unbaked metric scale required')
    size = _positive(c.get('size_xyz'), (3,))
    rotation = np.asarray(c.get('rotation'), float)
    translation = np.asarray(c.get('translation'), float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(rotation), 1., atol=2e-4)):
        raise PendingPublication('invalid baked transform provenance')
    yaw = float(override.get('yaw_degrees', 0))
    if yaw not in (0., 90., -90., 180.):
        raise PendingPublication('only reviewed right-angle correction supported')
    if override and not override.get('heading_evidence'):
        raise PendingPublication('orientation override requires review evidence')
    if category in DIRECTIONAL and not (c.get('heading_verified') is True or override.get('heading') == '+X'):
        raise PendingPublication('direction requires review')
    rad = np.deg2rad(yaw)
    turn = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    oriented_size = np.abs(turn).round(12) @ size
    if category in VEHICLES and oriented_size[1] > oriented_size[0]:
        raise PendingPublication('vehicle long axis not X')
    axis = int(np.argmax(oriented_size))
    if binding is not None:
        if binding.get('coordinate', {}).get('metric_scale_baked'):
            raise PendingPublication('legacy binding scale already baked')
        scale = float(_positive(binding.get('scale'), ()))
        if not np.allclose(size * scale, _positive(binding.get('size_lwh_m'), (3,)), rtol=1e-6, atol=1e-6):
            raise PendingPublication('legacy binding dimensions mismatch')
        basis = 'preserved legacy fixed binding scale'
    elif placement is not None and 'metric_scale' in placement:
        scale = float(_positive(placement['metric_scale'], ()))
        if not np.allclose(size * scale, _positive(placement.get('dimensions_lwh_m'), (3,)), rtol=.001, atol=.0002):
            raise PendingPublication('fixed placement dimensions mismatch')
        basis = 'preserved fixed placement scale'
    else:
        estimates = meta.get('scale') or {}
        if estimates.get('scale_confidence') not in {'medium', 'high'}:
            raise PendingPublication('metric estimate confidence requires review')
        if axis == 1:
            raise PendingPublication('dominant Y metric-width meaning requires review')
        scale = float(_positive(estimates.get(['est_length_m', 'est_width_m', 'est_height_m'][axis]), ())) / oriented_size[axis]
        basis = 'dominant canonical axis matched to metric estimate'
    canonical = dict(rotation=(turn @ rotation).tolist(), translation=(turn @ translation).tolist())
    if c.get('heading_verified') is True or override.get('heading') == '+X':
        canonical['heading'] = '+X'
    name = dataset + '__' + category + '_' + hashlib.sha256(identity.encode()).hexdigest()[:16] + '.ply'
    entry = dict(global_uid=identity, dataset=dataset, **{'class': meta.get('class', category)},
                 category=category, asset_path=f'assets/rigid/{category}/{name}',
                 size_xyz=oriented_size.tolist(), scale_m=float(oriented_size[axis] * scale),
                 scale_axis='xyz'[axis], canonical=canonical)
    return entry, dict(scale=scale, scale_basis=basis, yaw_degrees=yaw,
                       dimensions_m=(oriented_size * scale).tolist(), heading_evidence=override.get('heading_evidence'))


def validate_ply(path, size):
    data = PlyData.read(str(path))['vertex'].data
    fields = data.dtype.names
    required = ['x', 'y', 'z', 'opacity', *[f'f_dc_{i}' for i in range(3)],
                *[f'scale_{i}' for i in range(3)], *[f'rot_{i}' for i in range(4)]]
    if not len(data) or not set(required) <= set(fields):
        raise PendingPublication('missing Gaussian fields or empty PLY')
    for key in fields:
        if data[key].dtype.kind in 'fiu' and not np.isfinite(data[key]).all():
            raise PendingPublication('nonfinite Gaussian attributes')
    q = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(float)
    if np.any(np.linalg.norm(q, axis=1) < 1e-8):
        raise PendingPublication('degenerate quaternion')
    scales = np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(float)
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(scales)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise PendingPublication('invalid activated Gaussian scale')
    xyz = np.column_stack([data[k] for k in ['x', 'y', 'z']]).astype(float)
    low, high = np.quantile(xyz, [.005, .995], axis=0)
    if not np.allclose(high - low, size, rtol=.02, atol=.0002):
        raise PendingPublication('PLY bounds and canonical size disagree')
    return dict(gaussian_count=len(data), bounds=[low.tolist(), high.tolist()])


def copy_verified(source, target, *, hardlink=False, expected_sha=None):
    """Immutable destination. Hard links only for frozen migration inputs."""
    source, target = Path(source).resolve(), Path(target)
    before = sha256(source)
    if expected_sha and before != expected_sha:
        raise ValueError('Source payload hash mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != before:
            raise ValueError('Refusing to replace a different published payload')
        return before
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=target.parent); os.close(fd)
    temp = Path(name)
    try:
        if hardlink:
            temp.unlink(); os.link(source, temp)
        else:
            shutil.copyfile(source, temp)
        if sha256(temp) != before:
            raise ValueError('Source changed during publication')
        # The caller holds the publication lock; no overwrite of another payload.
        if target.exists():
            raise FileExistsError(target)
        temp.rename(target)
    finally:
        temp.unlink(missing_ok=True)
    return before


def manifest(reference, entries):
    rows = sorted(entries, key=lambda x: x['global_uid'])
    if len({x['global_uid'] for x in rows}) != len(rows) or len({x['asset_path'] for x in rows}) != len(rows):
        raise ValueError('duplicate release identity/path')
    result = {k: deepcopy(reference[k]) for k in ('schema_version', 'manifest_type', 'path_policy', 'scale_policy', 'heading_policy')}
    result.update(total=len(rows), source_manifest='reviewed_rigid_publication',
                  counts={k: dict(Counter(x[f] for x in rows)) for k, f in
                          [('by_dataset', 'dataset'), ('by_category', 'category'), ('by_class', 'class')]}, assets=rows)
    return result
