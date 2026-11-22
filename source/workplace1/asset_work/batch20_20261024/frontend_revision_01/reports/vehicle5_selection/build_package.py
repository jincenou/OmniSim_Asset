"""Deliver seven existing standard vehicle assets without changing their payloads."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

LIB = Path('/mnt/ojc/workplace1/asset_library')
PILOT = Path('/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01')
LEGACY = Path('/mnt/wmj/workspace4/pipeline')
sys.path[:0] = [str(LEGACY / 'traj_pipeline'), str(LEGACY), '/mnt/ojc/pre_flux_pipeline']
from corner_case.assets import AssetLibrary, bind_rigid
from corner_case.export import _binding_dict
from curate.core import load_rigid
from asset_extraction.export import inspect_ply

NAME = 'rigid_vehicle7_H_20261024'
DEST = LIB / 'deliveries' / NAME
SELECTED = [
    ('V01_white_sedan', '白色轿车', 'waymo_val_000_t000021'),
    ('V02_dark_blue_sedan', '深蓝色轿车', 'waymo_val_000_t000127'),
    ('V03_silver_sedan', '银色轿车', 'waymo_val_000_t000128'),
    ('V04_white_suv', '白色 SUV', 'waymo_val_000_t000129'),
    ('V05_white_pickup', '白色皮卡', 'waymo_val_000_t000029'),
    ('V06_tanker_truck', '白蓝色罐车', 'scene-0001_t000065'),
    ('V07_bucket_truck', '白色高空作业卡车', 'waymo_val_000_t000047'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rebase(value, mapping):
    if isinstance(value, dict):
        return {k: rebase(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


assert not DEST.exists(), 'Use a new delivery name; never overwrite a previous package'
DEST.mkdir(parents=True)
rows = json.loads((LIB / 'manifests/asset_library_rigid.json').read_text())['assets']
matched = {r['obj_id']: r for p in (PILOT / 'evaluation/matches').glob('*.json')
           for r in json.loads(p.read_text())['thresholds']['0.5']['query']['matches']}
entries, index, flux_manifest, validation = [], [], [], []

for alias, label, oid in SELECTED:
    original = next(r for r in rows if r['provenance']['source_obj_id'] == oid)
    assert original['status'] == 'keep' and original['route'] == 'H'
    assert original['canonical']['heading_verified']
    assert original['quality']['rendering']['all_16_nonempty']
    folder = DEST / 'assets' / alias
    folder.mkdir(parents=True)
    mapping = {}

    def copy(source, filename, expected=None):
        source = Path(source)
        target = folder / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if expected is not None:
            assert sha(source) == expected, source
        shutil.copy2(source, target)
        assert sha(source) == sha(target), target
        relative = str(target.relative_to(DEST))
        mapping[str(source)] = relative
        return relative

    canonical = original['canonical']
    ply_rel = copy(LIB / canonical['canonical_path'], 'canonical.ply', canonical['sha256'])
    mapping[canonical['canonical_path']] = ply_rel
    copy(original['quality']['audit_image'], 'preview/source_and_16_views.jpg',
         original['quality']['audit_image_sha256'])
    for key, filename in [('evidence', 'preview/orbit_16_views.jpg'),
                          ('heading_evidence', 'preview/front_and_rear.jpg')]:
        copy(original['quality']['rendering'][key], filename)
    provenance = original['provenance']
    source_crop = copy(provenance['source_image'], 'source/foreground.jpg', provenance['source_image_sha256'])
    source = provenance['source']
    source_frame = copy(source['source_image'], 'source/frame.jpg')
    for i, real in enumerate(provenance['real_images']):
        copy(real['rgba_path'], f'source/real_{i:02d}.png', real['sha256'])
    chunk = PILOT / 'chunks' / provenance['batch_chunk']
    generation = json.loads((chunk / 'generation/generation.json').read_text())
    job = next(r for r in generation['assets'] if r['obj_id'] == oid)
    for i, flux in enumerate(job['stages']['flux']['images']):
        copy(flux['path'], f'source/flux_{i:02d}.png', flux['sha256'])
    entry = rebase(deepcopy(original), mapping)
    entry['source_path'] = ply_rel
    entry['canonical']['canonical_path'] = ply_rel
    write(folder / 'metadata.json', entry)
    entries.append(entry)
    camera = {k: deepcopy(source[k]) for k in ['scene_id', 'camera_name', 'frame_index',
        'timestamp_us', 'bbox_xyxy', 'camera_intrinsic', 'camera_to_world'] if k in source}
    with Image.open(folder / 'source/frame.jpg') as image:
        camera['image_wh'] = list(image.size)
    camera.update(image_path=source_frame, path_root='package_root',
        purpose='Source observation calibration; not a target insertion camera or object pose.',
        native_object_pose_included=False)
    write(folder / 'source_camera.json', camera)
    flux_manifest.append(dict(scene_id=job['scene_id'], obj_id=oid, category=job['category'],
                              image_path=source_crop, prompts=job['prompts']))
    native = matched[oid]
    index.append(dict(asset_id=alias, name=label, uid=entry['uid'], obj_id=oid,
        category=entry['category'], route=entry['route'], real_images=len(provenance['real_images']),
        gaussian_count=canonical['validation']['gaussian_count'],
        quality_score=entry['quality_score'], canonical_ply=ply_rel,
        metadata=str((folder / 'metadata.json').relative_to(DEST)),
        binding=str((folder / 'binding.json').relative_to(DEST)),
        native_camera_track_id=native['gt_id'], native_match_iou=native['iou'],
        dataset=entry['dataset'], native_identity_scope=native['native_annotation'].get('identity_scope',
            'native_instance_token' if entry['dataset']=='nuscenes' else 'camera_track')))
    info = inspect_ply(folder / 'canonical.ply')
    assert info['gaussian_count'] == canonical['validation']['gaussian_count']
    gs = load_rigid(folder / 'canonical.ply')
    assert gs.xyz.shape == (1, info['gaussian_count'], 3)
    assert all(np.isfinite(v).all() for v in [gs.xyz, gs.rot, gs.scale, gs.opacity, gs.color])
    assert np.allclose(info['robust_bounds'], canonical['validation']['robust_bounds'])
    validation.append(dict(asset_id=alias, sha256=sha(folder / 'canonical.ply'),
        payload_unchanged=True, legacy_gaussian_decoding='PASS',
        all_16_nonempty_evidence=original['quality']['rendering']['foreground_pixels'], **info))
    print('Prepared', alias, oid, flush=True)

assert len({r['native_camera_track_id'] for r in index}) == len(SELECTED)
manifest = DEST / 'manifests/asset_library_rigid.json'
write(manifest, dict(schema_version=2, total=len(SELECTED), counts=dict(car=4, truck=3), assets=entries))
write(DEST / 'manifests/flux_input.json', flux_manifest)
library = AssetLibrary(manifest, pipeline_root=DEST)
assert len(library.entries) == len(SELECTED)
bindings = []
for row, entry, check in zip(index, library.entries, validation):
    assert row['uid'] == entry.uid
    bound = bind_rigid(entry, np.random.default_rng(42))
    binding = _binding_dict(bound)
    binding['asset_path'] = str(bound.asset_path.relative_to(DEST))
    assert np.allclose(bound.size_lwh_m, np.asarray(entry.raw['canonical']['size_xyz']) * bound.scale)
    assert np.isclose(bound.size_lwh_m[0], entry.raw['scale']['est_length_m'] *
                      np.random.default_rng(42).uniform(.94, 1.06))
    write(DEST / row['binding'], binding)
    bindings.append(binding)
    row.update(metric_scale=bound.scale, bound_size_lwh_m=list(bound.size_lwh_m))
    check.update(legacy_asset_library='PASS', legacy_bind_rigid='PASS',
                 scale=bound.scale, size_lwh_m=list(bound.size_lwh_m))
write(DEST / 'manifests/rigid_bindings.json', bindings)
write(DEST / 'INDEX.json', index)
with (DEST / 'INDEX.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(index[0]))
    writer.writeheader()
    writer.writerows(index)
from PIL import ImageOps, ImageDraw
overview = Image.new('RGB', (1500, len(entries)*300), '#f4f4f4')
draw = ImageDraw.Draw(overview)
for i, (row, entry) in enumerate(zip(index, entries)):
    y = i*300
    draw.text((8,y+5), f'{row["asset_id"]} | {row["obj_id"]} | H | source / 45 / 135 / 225 / 315 deg', fill='black')
    with Image.open(DEST/entry['provenance']['source_image']) as image:
        thumb = ImageOps.contain(image.convert('RGB'), (290,270))
        overview.paste(thumb, ((300-thumb.width)//2,y+25+(270-thumb.height)//2))
    with Image.open(DEST/entry['quality']['rendering']['evidence']) as image:
        size = entry['quality']['rendering']['resolution']
        for j, view in enumerate([2,6,10,14],1):
            x, top = view%4*size, view//4*(size+24)+24
            overview.paste(image.crop((x,top,x+size,top+size)).resize((270,270)), (j*300+15,y+25))
overview.save(DEST/'OVERVIEW.jpg', quality=93)

readme = '''七个车辆资产交接包（当前 H 路线标准输出，无公交车）

打开 OVERVIEW.jpg 看原图与四个方向的渲染；INDEX.csv / INDEX.json 是清单。
V01 白色轿车；V02 深蓝色轿车；V03 银色轿车；V04 白色 SUV；V05 白色皮卡。
新增 V06 白蓝色罐车（nuScenes）；V07 白色高空作业卡车（Waymo）。
七个对象来自当前试验场景，均为 H、自动筛选保留，交接时查看了多方向效果。
原生标注匹配的 ID 均不同；Waymo 是相机轨迹 ID，nuScenes 是 instance token，
这些 ID 的作用域不同，不把 Waymo 相机轨迹称为跨相机三维跟踪 ID。
V01–V04 使用 1 张合格真实参考图，V05 使用 0 张；均为 H，使用已有输出。
V06 使用 2 张真实参考图，V07 使用 1 张。
车辆仍有局部生成缺陷：罐车后端出现类似格栅/窗的非真实结构，
作业卡车的吊臂、车斗和标识有局部噪声或形状偏差。插入前应重点查看对应后视图；
本包保留原自动 keep 结论与全部缺陷记录，不将它们标成无缺陷的最终验收资产。

目录与配置：
  manifests/asset_library_rigid.json  原 pipeline 的标准资产库格式，7 条完整记录。
  manifests/rigid_bindings.json       原 bind_rigid + _binding_dict 导出的插入绑定。
  manifests/flux_input.json           原 5 字段 FLUX 输入，prompts 原样保留。
  assets/V*/canonical.ply            标准 Gaussian PLY，原输出字节不变。
  assets/V*/metadata.json            分类、质量、尺度估计、变换、朝向、来源。
  assets/V*/binding.json             kind/uid/asset_path/scale/size_lwh_m。
  assets/V*/source_camera.json       原始观测的相机内参和 camera_to_world 等。
  assets/V*/source/                  数据集原帧、分割前景、实际真实参考和 FLUX 图片。
  assets/V*/preview/                 16 方向预览、原图对比、正反面检查。
  VALIDATION.json / SHA256SUMS.txt    加载验证及全部文件校验。

所有可加载路径均相对“解压后的本包根目录”，不是相对 metadata.json。
provenance 中未随包提供的历史帧、工作目录和 raw_source_path 仅为溯源记录；
使用 canonical.ply 和插入配置无需访问原服务器或重新下载模型。

接入旧 pipeline：
  from pathlib import Path
  import json
  from corner_case.assets import AssetLibrary
  root = Path('/你的解压目录/rigid_vehicle7_H_20261024').resolve()
  library = AssetLibrary(root / 'manifests/asset_library_rigid.json', pipeline_root=root)
  bindings = json.loads((root / 'manifests/rigid_bindings.json').read_text())
  for binding in bindings:
      binding['asset_path'] = str(root / binding['asset_path'])
  # 将所选 binding 作为原 scenario.json 对应 object 的 binding。

上述模块来自旧工程 traj_pipeline/corner_case/assets.py 和 export.py。
本包已实际调用旧 AssetLibrary、bind_rigid、curate.core.load_rigid 验证。
binding 固定使用原 bind_rigid 的 seed=42，包括其 ±6% 随机尺度扰动；
如需完全一致地插入，直接复用已导出的 binding，不要再随机绑定。

坐标及调整规则：
1. canonical.ply 已转换到 Z-up、+X 车头、水平居中、地面 z=0。
   metadata.canonical.rotation/translation 已经施加，不能再施加一次。
   旧 load_rigid 的 up 字段是历史默认值；本包以 metadata.canonical.up=+Z 为准，
   canonical 插入路径不得再套原始 TRELLIS 的 UP_FIX。
2. PLY 为归一化单位，尚未烘焙米制尺度；scale 是归一化单位到米的倍数。
   尺度来源是视觉估计，不是传感器测量；实际等比放缩后的长宽高看 binding.size_lwh_m，
   不要将 scale 中三个估计尺寸当成各轴单独缩放比例。
3. 后续位置/朝向由目标场景轨迹和接地决定：
   p_world = Rz(yaw) @ (scale * p_canonical) + [x, y, ground_z]。
   Gaussian 尺度同步乘 scale，Gaussian 旋转同步左乘世界旋转；颜色与 opacity 不变。
4. 需要调整大小时同时改 binding.scale 和 size_lwh_m；碰撞轮廓同步等比更新。
   归一化碰撞轮廓在 metadata.canonical.collision_footprint_xy。

本包是资产及配置交接，并非已在目标重建场景完成插入的结果。
目标背景 checkpoint、目标相机、轨迹、接地高度和场景颜色校准由后续插入任务提供；
source_camera.json 是源观测相机，不能当成目标插入相机，也不能据此推断车辆世界位姿。
已有审核缺陷原样保留在 metadata 中；局部纹理、车标、轮胎或车型细节仍可有偏差。
本次只整理现有标准输出，未重新生成或改写资产几何。
'''
(DEST / 'README.txt').write_text(readme)
write(DEST / 'VALIDATION.json', dict(status='PASS',
    checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), count=len(SELECTED),
    paths_relative_to='package_root', metrics_or_model_inference_rerun=False,
    source_manifest_sha256=sha(LIB / 'manifests/asset_library_rigid.json'),
    legacy_code_sha256={str(p): sha(p) for p in [LEGACY/'curate/core.py',
        LEGACY/'traj_pipeline/corner_case/assets.py', LEGACY/'traj_pipeline/corner_case/export.py']},
    native_annotation_ids_distinct=True, assets=validation))

files = sorted(p for p in DEST.rglob('*') if p.is_file())
hashes = {str(p.relative_to(DEST)): sha(p) for p in files}
(DEST / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name, h in hashes.items()))
archive = DEST.with_suffix('.zip')
temporary = DEST.with_suffix('.zip.partial')
assert not archive.exists() and not temporary.exists()
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file():
            z.write(p, str(p.relative_to(DEST.parent)))
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    assert sum(p.endswith('/canonical.ply') for p in z.namelist()) == len(SELECTED)
    for name, expected in hashes.items():
        h = hashlib.sha256()
        with z.open(f'{NAME}/{name}') as f:
            for block in iter(lambda: f.read(4 << 20), b''):
                h.update(block)
        assert h.hexdigest() == expected, name
temporary.replace(archive)
archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
print(json.dumps(dict(archive=str(archive), size_mib=archive.stat().st_size/1024**2,
    assets=len(SELECTED), files=len(hashes)+1, zip_crc='PASS', all_file_hashes='PASS'), indent=2), flush=True)
