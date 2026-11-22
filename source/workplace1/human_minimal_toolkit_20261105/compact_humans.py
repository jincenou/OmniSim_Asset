#!/usr/bin/env python3
"""Batch calibrated human assets into one Hasim index and two files per asset.
Requires Python >=3.10 and NumPy. Only explicit inplace --apply prunes source data.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile
import numpy as np

FORMAT = 'hasim_asset_runtime_v1'
CONVENTION = 'right_handed_x_forward_y_left_z_up'
MAP = {'frame_xyz':'frame_xyz', 'frame_quaternions':'frame_rotation',
       'frame_scales':'frame_scaling', 'opacities':'opacity', 'sh':'shs',
       'timestamps':'frame_timestamps'}
CALIBRATION = ('root_xy','root_yaw','asset_to_standard','ground_offset')
# Exact-array-bound qualified profiles; populated from the verified Hasim catalog.
BUILTIN_CYCLES = {'waymo_grid4d_H0007': {'asset_id': 'waymo_grid4d_H0007', 'start_s': 0.78, 'end_s': 2.02, 'blend_half_window_s': 0.16, 'source_duration_s': 2.4, 'source_sha256': '10d8a0ff29a73c56aa779270468238ca9692841f080c7457a08aa22a05600438', 'mode': 'locomotion', 'native_speed_mps': 0.9260069727897644, 'min_playback_rate': 0.75, 'max_playback_rate': 1.35, 'qualified': True, 'diagnostics': {'body_window_rms_m': 0.018394071603183806, 'foot_window_rms_m': 0.027152267025256305, 'blended_wrap_velocity_disagreement_mps': 0.0030793684357126637, 'original_gaussian_count': 27974, 'foot_proxy_speed_median_mps': 1.0208520889282227, 'qualification_scope': 'Temporal reuse of this exact baked asset; not a naturalness or foot-lock certificate', 'visual_review': 'Two fixed cameras, three cycles at 25 fps; all frames decoded; complete first-cycle and seam-window images reviewed', 'contact_verified': False, 'anti_skating_verified': False, 'velocity_seam_epsilon_s': 0.001}}, 'waymo_grid4d_H1132': {'asset_id': 'waymo_grid4d_H1132', 'start_s': 1.14, 'end_s': 2.26, 'blend_half_window_s': 0.16, 'source_duration_s': 4.8, 'source_sha256': '023b8e37d8b448cdf2cefba65a0bf4ae253bb1f7c63410c3ac6f4b68fca962cf', 'mode': 'locomotion', 'native_speed_mps': 0.9973883032798767, 'min_playback_rate': 0.75, 'max_playback_rate': 1.35, 'qualified': True, 'diagnostics': {'body_window_rms_m': 0.023713472992683757, 'foot_window_rms_m': 0.02437896460143072, 'blended_wrap_velocity_disagreement_mps': 0.002979373122898555, 'original_gaussian_count': 27491, 'foot_proxy_speed_median_mps': 1.0021306276321411, 'qualification_scope': 'Temporal reuse of this exact baked asset; not a naturalness or foot-lock certificate', 'visual_review': 'Two fixed cameras, three cycles at 25 fps; all frames decoded; complete first-cycle and seam-window images reviewed', 'contact_verified': False, 'anti_skating_verified': False, 'velocity_seam_epsilon_s': 0.001}}, 'waymo_grid4d_H0752': {'asset_id': 'waymo_grid4d_H0752', 'start_s': 0.54, 'end_s': 1.82, 'blend_half_window_s': 0.16, 'source_duration_s': 2.4, 'source_sha256': '345173dee5b1737ba6712d1f82b0404ce005acc89b3848b82e97759450319af3', 'mode': 'stationary', 'native_speed_mps': 0.04532882002854253, 'min_playback_rate': 0.75, 'max_playback_rate': 1.35, 'qualified': True, 'diagnostics': {'body_window_rms_m': 0.017178837980144024, 'foot_window_rms_m': 0.014406055947748485, 'blended_wrap_velocity_disagreement_mps': 0.0012110007373271237, 'original_gaussian_count': 33930, 'foot_proxy_speed_median_mps': None, 'qualification_scope': 'Temporal reuse of this exact baked asset; not a naturalness or foot-lock certificate', 'visual_review': 'Two fixed cameras, three cycles at 25 fps; all frames decoded; complete first-cycle and seam-window images reviewed', 'contact_verified': False, 'anti_skating_verified': False, 'velocity_seam_epsilon_s': 0.001, 'known_visual_limit': 'Source asset has sparse floating Gaussians behind shoulders; standing use only, not a clean appearance example'}}}


def read_json(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def fingerprint(arrays):
    # Same representation as Hasim human_cycle.runtime_fingerprint.
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(json.dumps([key, a.dtype.str, a.shape], separators=(',', ':')).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def safe_path(root, value):
    path = Path(value)
    if path.is_absolute(): raise ValueError(f'Expected package-relative path: {value}')
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()): raise ValueError(f'Path escapes source: {value}')
    return result


def source_index(source):
    root = Path(source).expanduser().resolve()
    if root.is_file(): return root
    for p in (root/'human/drivelab_asset.json', root/'drivelab_asset.json'):
        if p.is_file(): return p
    raise ValueError('Expected human/drivelab_asset.json or human/drivelab_asset.json parent')


def records(source):
    index = source_index(source)
    entries = read_json(index).get('assets')
    if not isinstance(entries, list) or not entries: raise ValueError('Empty/missing human assets list')
    result = []; seen = set(); packages = set()
    for row in entries:
        aid = row.get('global_uid', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', aid): raise ValueError(f'Unsafe/missing asset ID: {aid!r}')
        if aid in seen: raise ValueError(f'Duplicate ID: {aid}')
        seen.add(aid)
        package = safe_path(index.parent, row['package_path'])
        if package == index.parent.resolve() or package in packages:
            raise ValueError(f'Package must be a unique child folder, not the whole library: {package}')
        packages.add(package)
        insertion = read_json(package/'insertion.json')
        manifest = read_json(package/'manifest.json')
        practical = insertion.get('practical_playback', {})
        if not practical.get('approximate_placement_available'):
            raise ValueError(f'{aid}: missing practical playback calibration; cannot infer it safely')
        if practical.get('coordinate_convention') != CONVENTION:
            raise ValueError(f'{aid}: unsupported coordinate convention')
        reason = None
        if row.get('skip_insertion') or row.get('quality_state') == 'severe_failure':
            reason = str(row.get('insertion_exclusion_reason') or row.get('quality_state') or 'skip_insertion')
        if insertion.get('skip_insertion'):
            reason = str(insertion.get('insertion_exclusion_reason') or reason or 'skip_insertion')
        if manifest.get('skip_insertion'):
            reason = str(manifest.get('insertion_exclusion_reason') or reason or 'skip_insertion')
        payload = safe_path(package, manifest.get('asset', {}).get('path', 'asset_4dgs.npz'))
        playback = safe_path(package, practical.get('arrays', 'placement_playback.npz'))
        for path in (payload, playback):
            if not path.is_file(): raise FileNotFoundError(path)
        result.append(dict(id=aid, row=row, insertion=insertion, manifest=manifest,
                           payload=str(payload), playback=str(playback), package=str(package), excluded_reason=reason))
    if any(any(parent in packages for parent in package.parents) for package in packages):
        raise ValueError('Nested/overlapping human packages are unsafe for batch pruning')
    # Do not silently miss additional delivered human packages outside the source manifest.
    unindexed = [str(p.parent) for p in index.parent.glob('*/asset_4dgs.npz') if p.parent.resolve() not in packages]
    if unindexed: raise ValueError(f'Unindexed human packages: {unindexed}')
    return index, sorted(result, key=lambda x:x['id'])


def metadata(record):
    e, m = record['row'], record['manifest']; p = record['insertion']['practical_playback']
    fps = float(m['fps']); count = int(m['frame_count'])
    if not np.isfinite(fps) or fps <= 0 or count < 2: raise ValueError('Invalid source clock')
    dimensions = np.asarray(e['size_xyz'], dtype=float) * float(e.get('asset_unit_to_model_meter', 1))
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Invalid asset dimensions')
    capabilities = e.get('playback_capabilities') or {
        'heading_normalized':p.get('body_heading_from_gaussians_available') is True,
        'contact_verified':p.get('contact_verified') is True,
        'anti_skating_verified':p.get('anti_skating_verified') is True,
        'action_verified':p.get('action',{}).get('human_verified') is True,
        'speed_basis':p.get('speed_basis','unspecified'), 'ground_basis':p.get('ground_basis','unspecified')}
    meta = dict(format=FORMAT,id=record['id'],category='human',kind='human',payload='runtime.npz',
        duration_s=(count-1)/fps, native_speed_mps=p.get('native_speed_model_mps'),
        motion_type=p.get('action',{}).get('label','unknown'), dimensions=dimensions.tolist(),
        quality_approved=bool(e.get('quality_approved',False)),quality_state=e.get('quality_state','unknown'),
        producer_status=e.get('producer_status',e.get('status')),quality_flags=e.get('quality_flags',[]),
        coordinate_convention=CONVENTION,playback='once',
        metric_scale_verified=bool(e.get('metric_scale_verified',False)),playback_capabilities=copy.deepcopy(capabilities),
        animation_sampling='baked_frames_linear_positions_and_spd_covariance')
    if record['excluded_reason']:
        meta.update(skip_insertion=True,insertion_exclusion_reason=record['excluded_reason'])
    return meta


def read_arrays(record):
    with np.load(record['payload'], allow_pickle=False) as z:
        arrays = {k:z[v] for k,v in MAP.items()}
        arrays['opacities'] = arrays['opacities'].reshape(-1)
        if bool(np.asarray(z['use_rgb']).reshape(-1)[0]):
            arrays['sh'] = ((arrays['sh'].reshape(-1,3)-.5)/0.28209479177387814)[:,None]
        else:
            degree = int(np.asarray(z['active_sh_degree']).reshape(-1)[0])
            if arrays['sh'].ndim != 3 or arrays['sh'].shape[1] != (degree+1)**2: raise ValueError('SH degree mismatch')
    with np.load(record['playback'],allow_pickle=False) as z:
        if not np.allclose(z['frame_timestamps'],arrays['timestamps'],atol=1e-8,rtol=0): raise ValueError('Placement clock mismatch')
        for k in CALIBRATION: arrays[k] = z[k]
    arrays['root_yaw'] = np.unwrap(arrays['root_yaw'])
    return arrays


def validate(arrays, meta):
    if set(arrays) != set(MAP) | set(CALIBRATION): raise ValueError('Unexpected runtime fields')
    xyz = arrays['frame_xyz']; times = arrays['timestamps']
    if xyz.ndim != 3 or xyz.shape[2] != 3 or xyz.shape[0]<2 or xyz.shape[1]<1: raise ValueError('Invalid frame_xyz')
    f,n,_ = xyz.shape
    expected={'timestamps':(f,),'frame_quaternions':(f,n,4),'frame_scales':(f,n,3),'opacities':(n,),
              'root_xy':(f,2),'root_yaw':(f,),'asset_to_standard':(4,4),'ground_offset':()}
    for k, shape in expected.items():
        if arrays[k].shape != shape: raise ValueError(f'{k}: shape {arrays[k].shape} != {shape}')
    for k,a in arrays.items():
        if not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all(): raise ValueError(f'{k}: nonfinite/non-numeric')
    if np.any(np.diff(times)<=0) or not np.isclose(times[-1]-times[0],meta['duration_s'],atol=1e-6): raise ValueError('Invalid duration')
    if np.any(arrays['frame_scales']<=0): raise ValueError('Nonpositive scales')
    if np.any((arrays['opacities']<0)|(arrays['opacities']>1)): raise ValueError('Opacity must be activated')
    if not np.allclose(np.linalg.norm(arrays['frame_quaternions'],axis=-1),1,atol=2e-4): raise ValueError('Quaternions not normalized')
    sh=arrays['sh']
    if sh.ndim!=3 or sh.shape[0]!=n or sh.shape[1] not in (1,4,9,16) or sh.shape[2]!=3: raise ValueError('Incomplete SH')
    mat=arrays['asset_to_standard']; linear=mat[:3,:3]; scale=np.linalg.norm(linear,axis=0)
    if not np.allclose(mat[3],[0,0,0,1]) or np.any(scale<=0) or not np.allclose(scale,scale[0],rtol=1e-5,atol=1e-7): raise ValueError('Invalid similarity transform')
    rotation=linear/scale[0]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5): raise ValueError('Invalid transform rotation')


def attach_cycle(record,meta,arrays,profiles):
    explicit=record['row'].get('cycle_profile') or record['insertion'].get('cycle_profile')
    profile=explicit or profiles.get(record['id'])
    if not profile: return
    profile=copy.deepcopy(profile)
    if not profile.get('qualified'): return
    if profile.get('asset_id') != record['id'] or profile.get('source_sha256') != fingerprint(arrays):
        raise ValueError(f'{record["id"]}: qualified cycle fingerprint mismatch; refusing silent loss/reuse')
    if not np.isclose(profile['source_duration_s'],meta['duration_s']): raise ValueError('Cycle source duration mismatch')
    start,end,width=profile['start_s'],profile['end_s'],profile['blend_half_window_s']
    if width<=0 or start<width or end+width>meta['duration_s']+1e-12 or end-start<=2*width: raise ValueError('Invalid cycle window')
    meta['cycle_profile']=profile;meta['playback_capabilities']['qualified_cycle_available']=True


def hasim_api(root):
    if 'hasim' not in sys.modules:
        source=Path(root).expanduser().resolve()/'src'
        spec=importlib.util.spec_from_file_location('hasim',source/'__init__.py',submodule_search_locations=[str(source)])
        if spec is None or spec.loader is None: raise ValueError('Invalid Hasim checkout')
        module=importlib.util.module_from_spec(spec);sys.modules['hasim']=module;spec.loader.exec_module(module)
    from hasim.shared.assets.library_v2 import AssetLibrary,LibraryAsset,load_asset
    return AssetLibrary,LibraryAsset,load_asset


def verify_hasim(root,record,meta,arrays,payload):
    _,Asset,load=hasim_api(root)
    # Independently exercise Hasim's original loader, including calibration and selection metadata.
    entry=record['row'];manifest=record['manifest'];practical=record['insertion']['practical_playback']
    srcmeta=dict(entry,format='human_npz',duration_s=meta['duration_s'],dimensions=meta['dimensions'],
                 native_speed_mps=meta['native_speed_mps'],motion_type=meta['motion_type'],
                 playback_path=record['playback'],practical_playback=practical)
    original=load(Asset(record['id'],'human','human',Path(record['payload']),srcmeta,()))
    new=load(Asset(record['id'],'human','human',Path(payload),meta,()))
    for key in arrays: np.testing.assert_array_equal(original.arrays[key],new.arrays[key])
    for t in np.linspace(0,meta['duration_s'],5):
        a,b=original.sample(float(t)),new.sample(float(t))
        for field in ('means','quaternions','scales','opacities','sh','local_from_appearance'):
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    if meta.get('cycle_profile'):
        from hasim.shared.assets.human_cycle import CyclicRuntimeAsset,CycleProfile
        profile=CycleProfile.from_dict(meta['cycle_profile'])
        a,b=CyclicRuntimeAsset(original,profile),CyclicRuntimeAsset(new,profile)
        for t in (0.,profile.period_s-1e-4,profile.period_s+1e-4,20.5):
            first,second=a.sample(t),b.sample(t)
            for field in ('means','quaternions','scales','opacities','sh','local_from_appearance'):
                np.testing.assert_array_equal(getattr(first,field),getattr(second,field))
    return dict(array_equality=True,sample_times=5,cycle_checked=bool(meta.get('cycle_profile')))


def process_one(job):
    record, output, report_dir, profiles, hasim, trial, resume = job
    aid=record['id'];arrays=read_arrays(record);meta=metadata(record);validate(arrays,meta);attach_cycle(record,meta,arrays,profiles)
    relative=Path('excluded' if record['excluded_reason'] else 'assets')/aid
    target=Path(output)/relative;report=Path(report_dir)/'assets'/f'{aid}.json'
    original_fingerprint=fingerprint(arrays)
    if not trial and (target.is_symlink() or target.parent.is_symlink()): raise ValueError('Output asset paths must not be symlinks')
    if target.exists() and not trial:
        if not resume: raise FileExistsError(f'{target}; use --resume for verified existing output')
        if any(p.is_symlink() for p in target.iterdir()): raise ValueError('Output files must not be symlinks')
        if sorted(p.name for p in target.iterdir()) != ['asset.json','runtime.npz']: raise ValueError(f'Unexpected files in {target}')
        if read_json(target/'asset.json') != meta: raise ValueError(f'{aid}: existing metadata differs')
        staging=None;folder=target
    else:
        base=Path(report_dir)/'trial' if trial else target.parent
        base.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=f'.{aid}-',dir=base));folder=staging
    try:
        if staging is not None:
            with (folder/'runtime.npz').open('wb') as f:
                with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for key, array in arrays.items():
                        with archive.open(key+'.npy', 'w', force_zip64=True) as member:
                            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
                f.flush();os.fsync(f.fileno())
            write_json(folder/'asset.json',meta)
        with np.load(folder/'runtime.npz',allow_pickle=False) as z:
            if set(z.files)!=set(arrays): raise ValueError('Serialized field mismatch')
            for key in arrays: np.testing.assert_array_equal(arrays[key],z[key])
        verification=verify_hasim(hasim,record,meta,arrays,folder/'runtime.npz') if hasim else None
        # Detect source array or metadata changes during the export before accepting it.
        disk_row_meta=read_json(Path(record['package'])/'insertion.json')
        if disk_row_meta != record['insertion'] or read_json(Path(record['package'])/'manifest.json') != record['manifest']:
            raise ValueError('Source metadata changed during conversion')
        result=dict(id=aid,metadata=str(relative/'asset.json'),excluded_reason=record['excluded_reason'],
                    source_array_fingerprint=original_fingerprint,frame_count=len(arrays['timestamps']),
                    gaussian_count=arrays['frame_xyz'].shape[1],duration_s=meta['duration_s'],motion_type=meta['motion_type'],
                    qualified_cycle=bool(meta.get('cycle_profile')),roundtrip_exact=True,hasim=verification,
                    runtime_bytes=(folder/'runtime.npz').stat().st_size,
                    runtime_sha256=digest_file(folder/'runtime.npz'),metadata_sha256=digest_file(folder/'asset.json'),
                    source_payload=record['payload'],trial=trial)
        if staging is not None and not trial: os.replace(staging,target);staging=None
        write_json(report,result)
        return result
    finally:
        if staging is not None: shutil.rmtree(staging)


def plan(entries):
    compressed=0;raw=0
    wanted={v+'.npy' for v in MAP.values()}
    for r in entries:
        with zipfile.ZipFile(r['payload']) as z:
            info=[i for i in z.infolist() if i.filename in wanted]
            if len(info)!=len(wanted): raise ValueError(f'{r["id"]}: missing source arrays')
            compressed+=sum(i.compress_size for i in info);raw+=sum(i.file_size for i in info)
        metadata(r)
    return dict(total=len(entries),eligible=sum(not x['excluded_reason'] for x in entries),
                excluded=[{'id':x['id'],'reason':x['excluded_reason']} for x in entries if x['excluded_reason']],
                estimated_payload_bytes=compressed,uncompressed_core_array_bytes=raw,
                note='ZIP member estimate, not exact output size; allow 20% headroom plus 2 GiB.')


def patch_hasim(root):
    """Install only the narrowly scoped mixed human-library adapter, idempotently."""
    path=Path(root).expanduser().resolve()/'src/shared/assets/library_v2.py'
    text=path.read_text()
    marker='# human_compaction_shared_index_v1'
    minimal_loop='        for row in data.get("assets", []):'
    exclusion_marker='# human_compaction_exclusions_v1'
    if exclusion_marker not in text:
        if text.count(minimal_loop)!=1: raise ValueError('Unknown Hasim minimal loader; patch refused')
        text=text.replace(minimal_loop, '        '+exclusion_marker+'\n        for excluded in data.get("excluded_assets", []):\n            self.excluded[str(excluded["id"])] = str(excluded.get("reason", "excluded"))\n'+minimal_loop)
    if marker in text:
        compile(text,str(path),'exec')
        temporary=path.with_name(path.name+'.compact.tmp');temporary.write_text(text);os.replace(temporary,path)
        return
    if text.count('            if human.is_file():\n                rows.extend(self._human(human))')!=2:
        raise ValueError('Unknown Hasim loader revision; adapter patch refused. Use an updated Hasim checkout.')
    text=text.replace('            if human.is_file():\n                rows.extend(self._human(human))',
        '            '+marker+'\n            human_runtime = self.root / "human/library.json"\n            if human_runtime.is_file():\n                rows.extend(self._minimal(human_runtime))\n            elif human.is_file():\n                rows.extend(self._human(human))')
    needle='            package = (index.parent / entry["package_path"]).resolve()'
    if text.count(needle)!=1: raise ValueError('Unknown Hasim human loader; patch refused')
    text=text.replace(needle,'''            if entry.get("runtime_metadata"):
                metadata_path = (index.parent / entry["runtime_metadata"]).resolve()
                meta = _json(metadata_path)
                if meta.get("format") != _MINIMAL_FORMAT:
                    raise CatalogError(f"Unsupported migrated human format: {metadata_path}")
                if meta.get("skip_insertion") or meta.get("quality_state") == "severe_failure":
                    self.excluded[identity] = str(meta.get("insertion_exclusion_reason") or meta.get("quality_state"))
                    continue
                yield self._record(identity, "human", "human", metadata_path.parent / meta["payload"],
                                   meta, (index, metadata_path))
                continue
'''+needle)
    compile(text,str(path),'exec')
    temporary=path.with_name(path.name+'.compact.tmp');temporary.write_text(text);os.replace(temporary,path)


def verify_compacted(record, package, result, hasim):
    meta=read_json(package/'asset.json')
    if digest_file(package/'asset.json') != result['metadata_sha256'] or digest_file(package/'runtime.npz') != result['runtime_sha256']:
        raise ValueError(f'{record["id"]}: committed files changed')
    with np.load(package/'runtime.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    validate(arrays,meta)
    if fingerprint(arrays)!=result['source_array_fingerprint']: raise ValueError('Committed runtime fingerprint differs')
    if hasim:
        _,Asset,load=hasim_api(hasim)
        runtime=load(Asset(record['id'],'human','human',package/'runtime.npz',meta,()))
        for t in (0.,meta['duration_s']/2,meta['duration_s']): runtime.sample(t)
    return meta


def finish_package(record, root, result, registry, index, report_dir, hasim):
    """After exact verification: switch index, retain video by move, then prune."""
    package=Path(record['package'])
    meta=verify_compacted(record,package,result,hasim)
    for row in registry['assets']:
        if row['global_uid']==record['id']:
            row['runtime_metadata']=str((package/'asset.json').relative_to(root))
            row['asset_path']=str((package/'runtime.npz').relative_to(root))
            row['presentation_path']='previews/'+package.name+'.mp4'
            break
    # New reader can use the verified runtime before old files disappear.
    write_json(index,registry)
    video=package/'demo.mp4'; preview=root/'previews'/f'{package.name}.mp4'
    if video.is_file():
        preview.parent.mkdir(exist_ok=True)
        if preview.exists():
            if digest_file(preview)!=digest_file(video): raise ValueError('Preview target collision')
            video.unlink()
        else: os.replace(video,preview)
    kept={'asset.json','runtime.npz'}
    pruned=0
    for p in list(package.iterdir()):
        if p.name in kept: continue
        if p.is_symlink():p.unlink()
        elif p.is_dir():shutil.rmtree(p)
        else: pruned+=p.stat().st_size;p.unlink()
    # Remove the original alias, rather than silently changing its NPZ schema.
    alias=root/'assets/nonrigid/human'/f'{package.name}.npz'
    if alias.is_symlink():
        expected=Path(record['payload'])
        if alias.resolve()!=expected.resolve():raise ValueError(f'Unexpected legacy alias: {alias}')
        alias.unlink()
    result=dict(result,metadata=str((package/'asset.json').relative_to(root)),committed=True,
                preview=str(preview.relative_to(root)) if preview.is_file() else None,original_package_pruned=True)
    write_json(report_dir/'committed'/f'{record["id"]}.json',result)
    return result


def inplace_main(args):
    if not args.apply: raise ValueError('inplace permanently removes old producer files after verification; pass --apply only when authorized')
    if args.output or args.in_place_demo: raise ValueError('inplace operates on the original source; omit --output / --in-place-demo')
    report_dir=args.report_dir.expanduser().resolve();report_dir.mkdir(parents=True,exist_ok=True)
    journal_path=report_dir/'journal.json'
    source=Path(args.source).expanduser().resolve()
    with (report_dir/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if journal_path.exists():
            if not args.resume: raise ValueError('Existing in-place journal; use --resume')
            journal=read_json(journal_path);index=Path(journal['source_index']);root=index.parent
            if source not in (root,index,root.parent): raise ValueError('Resume source differs from journal')
            entries=journal['records'];registry=copy.deepcopy(journal['registry'])
            if index.is_file():registry=read_json(index)
        else:
            index,entries=records(args.source);root=index.parent
            if report_dir.is_relative_to(root):raise ValueError('Reports/journal must be outside source')
            registry=read_json(index)
            journal=dict(source_index=str(index),records=entries,registry=registry,
                         authorization='--apply: no asset backup; preserve demo videos, verify before deleting producer files',
                         initial_free_bytes=shutil.disk_usage(root).free)
            write_json(journal_path,journal)
        with (root/'.catalogue.lock').open('a+') as catalog_lock:
            fcntl.flock(catalog_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.hasim_root:
                if args.install_hasim_adapter:patch_hasim(args.hasim_root)
                loader=Path(args.hasim_root)/'src/shared/assets/library_v2.py'
                if '# human_compaction_shared_index_v1' not in loader.read_text():
                    raise ValueError('Mixed old/new library needs adapter; use --install-hasim-adapter with a supported Hasim checkout')
            selected=entries
            if args.asset_id:
                ids=set(args.asset_id);unknown=ids-{r['id'] for r in entries}
                if unknown:raise ValueError(f'Unknown IDs: {unknown}')
                selected=[r for r in entries if r['id'] in ids]
            profiles={} if args.no_builtin_cycles else copy.deepcopy(BUILTIN_CYCLES)
            if args.cycle_catalog:
                for p in read_json(args.cycle_catalog)['profiles']:profiles[p['asset_id']]=p
            staging=report_dir/'staging';results=[];failures=[];pending=[]
            for record in selected:
                receipt=report_dir/'committed'/f'{record["id"]}.json'
                verified=report_dir/'assets'/f'{record["id"]}.json'
                package=Path(record['package'])
                if not receipt.is_file() and verified.is_file() and (package/'runtime.npz').is_file() and not (package/'asset.json').exists():
                    staged_result=read_json(verified);staged_meta=staging/staged_result['metadata']
                    if staged_meta.is_file() and digest_file(package/'runtime.npz')==staged_result['runtime_sha256']:
                        os.replace(staged_meta,package/'asset.json')
                if receipt.is_file():
                    result=read_json(receipt);verify_compacted(record,package,result,args.hasim_root);results.append(result)
                elif (package/'asset.json').is_file() and (package/'runtime.npz').is_file() and verified.is_file():
                    result=finish_package(record,root,read_json(verified),registry,index,report_dir,args.hasim_root);results.append(result)
                else:pending.append(record)
            # Only a few temporary runtime payloads exist at once; each finished source is immediately pruned.
            if pending and shutil.disk_usage(root).free < 2*1024**3:
                raise OSError('Need at least 2 GiB free to start an in-place batch; lower --jobs if necessary')
            jobs=[(r,str(staging),str(report_dir),profiles,args.hasim_root,False,True) for r in pending]
            by_id={r['id']:r for r in entries}
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
                future_map={pool.submit(process_one,job):job[0]['id'] for job in jobs}
                for future in concurrent.futures.as_completed(future_map):
                    aid=future_map[future];record=by_id[aid]
                    try:
                        result=future.result();temp=staging/result['metadata'];package=Path(record['package'])
                        # If interrupted between these renames, verified staging receipt enables safe resume.
                        os.replace(temp.parent/'runtime.npz',package/'runtime.npz')
                        os.replace(temp,package/'asset.json')
                        result=finish_package(record,root,result,registry,index,report_dir,args.hasim_root)
                        results.append(result)
                        print(f'COMMITTED {len(results)}/{len(selected)} {aid}',flush=True)
                    except Exception as exc:
                        failures.append({'id':aid,'error':f'{type(exc).__name__}: {exc}'})
                        print(f'FAILED {aid}: {exc}',flush=True)
            committed=[]
            for record in entries:
                receipt=report_dir/'committed'/f'{record["id"]}.json'
                if receipt.exists():committed.append(read_json(receipt))
            complete=len(committed)==len(entries) and not failures
            if complete:
                shared=dict(format=FORMAT,
                    assets=[{'metadata':r['metadata']} for r in committed if not r['excluded_reason']],
                    excluded_assets=[{'id':r['id'],'metadata':r['metadata'],'reason':r['excluded_reason']} for r in committed if r['excluded_reason']])
                write_json(root/'library.json',shared)
                if args.hasim_root:
                    Library,_,_=hasim_api(args.hasim_root)
                    if {a.id for a in Library(root).assets}!={r['id'] for r in committed if not r['excluded_reason']}:raise ValueError('Final candidate mismatch')
                # Audit metadata stays in the journal; obsolete runtime index and empty alias tree go away.
                if index.exists():index.unlink()
                legacy=root/'assets'
                if legacy.exists() and not any(p.is_file() or p.is_symlink() for p in legacy.rglob('*')):shutil.rmtree(legacy)
                html=['<!doctype html><meta charset="utf-8"><title>人体资产原预览</title><h1>人体资产原预览</h1><p>视频已保留，运行包为各人体目录内 asset.json 与 runtime.npz；质量状态未升级。</p>']
                import html as html_module
                for r in committed:
                    if r.get('preview'):html.append(f'<p><a href="{html_module.escape(Path(r["preview"]).name)}">{html_module.escape(r["id"])}</a></p>')
                (root/'previews').mkdir(exist_ok=True);(root/'previews/index.html').write_text('\n'.join(html),encoding='utf-8')
            if complete and staging.exists() and not any(p.is_file() or p.is_symlink() for p in staging.rglob('*')): shutil.rmtree(staging)
            summary=dict(command='inplace',source_manifest_total=len(entries),selected=len(selected),committed_total=len(committed),
                         complete=complete,complete_selected=not failures and len(results)==len(selected),failures=failures,
                         runtime_bytes=sum(r['runtime_bytes'] for r in committed),
                         free_bytes_before=journal['initial_free_bytes'],free_bytes_after=shutil.disk_usage(root).free,
                         preview_policy='Moved original demo.mp4 to human/previews; no video backup copy',
                         source_assets_backed_up=False,hasim_verified=bool(args.hasim_root),results=committed)
            write_json(report_dir/'summary.json',summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False,indent=2),flush=True)
            if failures:raise RuntimeError('In-place batch partially completed; use same report directory and --resume')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','convert','trial','inplace'])
    parser.add_argument('--source',required=True,help='Human library folder or total drivelab_asset.json')
    parser.add_argument('--output',type=Path,help='New release folder, outside source; required for convert')
    parser.add_argument('--report-dir',type=Path,required=True,help='Separate audit folder, outside source and output')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--apply',action='store_true',help='Authorize inplace removal of old producer files after exact verification')
    parser.add_argument('--install-hasim-adapter',action='store_true',help='Install known minimal mixed-library adapter in --hasim-root')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--asset-id',action='append',help='Explicit small-batch IDs; repeat. Default: complete manifest')
    parser.add_argument('--in-place-demo',action='store_true',help='Write a reversible small runtime area under the original human library; never replace/delete producer files')
    parser.add_argument('--hasim-root',help='Optional Hasim checkout: compare its original/new loaders and playback')
    parser.add_argument('--no-builtin-cycles',action='store_true',help='For another library reusing IDs with different data; do not apply bundled profiles')
    parser.add_argument('--cycle-catalog',type=Path,help='Additional Hasim human_cycles.json; overrides built-in profiles by ID')
    args=parser.parse_args()
    if not 1<=args.jobs<=16: parser.error('--jobs must be 1..16')
    if args.command=='inplace':
        inplace_main(args); return
    index,entries=records(args.source);source_root=index.parent.resolve()
    total_source_entries=len(entries)
    if args.asset_id:
        wanted=set(args.asset_id); unknown=wanted-{r['id'] for r in entries}
        if unknown: parser.error(f'Unknown asset IDs: {sorted(unknown)}')
        entries=[r for r in entries if r['id'] in wanted]
    report_dir=args.report_dir.expanduser().resolve()
    output=args.output.expanduser().resolve() if args.output else None
    if report_dir.is_relative_to(source_root): parser.error('Report must be outside source')
    if args.in_place_demo:
        if args.command!='convert' or not args.asset_id: parser.error('--in-place-demo requires convert and explicit --asset-id values')
        if output is not None: parser.error('--in-place-demo selects its output automatically; omit --output')
        output=source_root/'_hasim_runtime_demo'
    if args.command=='convert' and output is None: parser.error('--output is required')
    if output and not args.in_place_demo and (output.is_relative_to(source_root) or source_root.is_relative_to(output)):
        parser.error('Output and source must not contain one another')
    if output and (report_dir.is_relative_to(output) or output.is_relative_to(report_dir)):
        parser.error('Reports and runtime output must be separate')
    report_dir.mkdir(parents=True,exist_ok=True)
    with (report_dir/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output_lock = None
        if output and args.command == 'convert':
            output.parent.mkdir(parents=True,exist_ok=True)
            output_lock=(output.parent/('.'+output.name+'.compact.lock')).open('a+')
            fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        summary=plan(entries);summary['source_manifest_total']=total_source_entries;summary['selection']='explicit_subset' if args.asset_id else 'complete_manifest';summary['source_index']=str(index);summary['source_index_sha256']=digest_file(index)
        write_json(report_dir/'plan.json',summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        if args.command=='plan': return
        if args.command=='convert':
            if output.exists() and any(output.iterdir()) and not args.resume: raise ValueError('Output not empty; use a fresh directory or --resume')
            output.mkdir(parents=True,exist_ok=True)
            allowed_top={'assets','excluded','library.json'}
            unexpected=[str(p) for p in output.iterdir() if p.name not in allowed_top]
            expected_paths={output/('excluded' if r['excluded_reason'] else 'assets')/r['id'] for r in entries}
            for category in ('assets','excluded'):
                if (output/category).exists():
                    unexpected.extend(str(p) for p in (output/category).iterdir() if p not in expected_paths)
            if unexpected: raise ValueError(f'Unrecognized existing output paths; use a new release directory: {unexpected[:10]}')
            existing=sum((output/('excluded' if r['excluded_reason'] else 'assets')/r['id']/'runtime.npz').stat().st_size
                         for r in entries if (output/('excluded' if r['excluded_reason'] else 'assets')/r['id']/'runtime.npz').is_file())
            required=max(0,int(summary['estimated_payload_bytes']*1.2)-existing)+2*1024**3
            free=shutil.disk_usage(output).free
            if free<required: raise OSError(f'Insufficient space: free {free/1024**3:.2f} GiB, require {required/1024**3:.2f} GiB; originals unchanged')
        profiles={} if args.no_builtin_cycles else copy.deepcopy(BUILTIN_CYCLES)
        if args.cycle_catalog:
            for p in read_json(args.cycle_catalog)['profiles']: profiles[p['asset_id']]=p
        jobs=[(r,str(output) if output else '',str(report_dir),profiles,args.hasim_root,args.command=='trial',args.resume) for r in entries]
        results=[];failures=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures={pool.submit(process_one,job):job[0]['id'] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                aid=futures[future]
                try:
                    result=future.result();results.append(result)
                    print(f'PASS {len(results)}/{len(entries)} {aid} {result["runtime_bytes"]/1e6:.2f} MB',flush=True)
                except Exception as exc:
                    failures.append({'id':aid,'error':f'{type(exc).__name__}: {exc}'});print(f'FAIL {aid}: {exc}',flush=True)
        source_unchanged=digest_file(index)==summary['source_index_sha256']
        if not source_unchanged: failures.append({'id':'INDEX','error':'Source index changed during run'})
        results.sort(key=lambda x:x['id'])
        final=dict(command=args.command,total=len(entries),source_manifest_total=total_source_entries,selection=summary['selection'],succeeded=len(results),failures=failures,
                   source_index_unchanged=source_unchanged,output=str(output) if output else None,
                   total_runtime_bytes=sum(r['runtime_bytes'] for r in results),hasim_verified=bool(args.hasim_root),
                   complete=not failures and len(results)==len(entries),results=results)
        write_json(report_dir/'summary.json',dict(final,complete=False,stage='payload_verification'))
        if not final['complete']: raise RuntimeError('Batch incomplete; inspect summary.json. No new total index published.')
        if args.command=='convert':
            index_data=dict(format=FORMAT,assets=[{'metadata':r['metadata']} for r in results if not r['excluded_reason']],
                            excluded_assets=[{'id':r['id'],'metadata':r['metadata'],'reason':r['excluded_reason']} for r in results if r['excluded_reason']])
            if args.hasim_root:
                Library,_,_=hasim_api(args.hasim_root)
                # Check the same metadata through a temporary shared index before publishing.
                with tempfile.TemporaryDirectory(prefix='index-check-',dir=report_dir) as td:
                    checked=copy.deepcopy(index_data)
                    for row in checked['assets']:
                        row['metadata']=os.path.relpath(output/row['metadata'],td)
                    write_json(Path(td)/'library.json',checked)
                    ids={a.id for a in Library(td).candidates('human')}
                    if ids!={r['id'] for r in results if not r['excluded_reason']}:raise ValueError('Final Hasim selection mismatch')
            # Publish a complete library index only after every asset has passed.
            write_json(output/'library.json',index_data)
        write_json(report_dir/'summary.json',final)
        print(f'COMPLETE {args.command}: {len(results)} assets; {final["total_runtime_bytes"]/1024**3:.2f} GiB',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
