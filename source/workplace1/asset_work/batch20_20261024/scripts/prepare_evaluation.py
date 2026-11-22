"""Evaluation-only native annotation registry and deterministic source matching."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/mnt/ojc/pre_flux_pipeline')
from asset_extraction.io_utils import read_json, write_json
from asset_extraction.generation import sha256
from asset_extraction.real_views import VEHICLES


def rotation(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project_box(object_to_world, size, frame, width, height):
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    corners = signs * np.asarray(size) / 2
    transform = np.linalg.inv(frame['camera_to_world']) @ object_to_world
    points = corners @ transform[:3, :3].T + transform[:3, 3]
    # Clip the cuboid's 12 edges to the camera near plane before projection.
    visible = [point for point in points if point[2] >= .1]
    for i, j in itertools.combinations(range(8), 2):
        if np.count_nonzero(signs[i] != signs[j]) != 1:
            continue
        a, b = points[i], points[j]
        if (a[2] < .1) != (b[2] < .1):
            visible.append(a + (.1-a[2])/(b[2]-a[2])*(b-a))
    if not visible:
        return None
    projected = np.asarray(visible) @ np.asarray(frame['camera_intrinsic']).T
    uv = projected[:, :2] / projected[:, 2:3]
    low, high = np.maximum(uv.min(0), 0), np.minimum(uv.max(0), [width, height])
    if (high-low < [8, 12]).any():
        return None
    return [*low.tolist(), *high.tolist()]


def nusc_category(name):
    if name.startswith('vehicle.'):
        kind = name.split('.')[1]
        return kind if kind in {'car', 'bus', 'truck', 'trailer', 'bicycle', 'motorcycle'} else 'vehicle'
    return {'movable_object.barrier': 'barrier', 'movable_object.trafficcone': 'cone',
            'movable_object.debris': 'road_debris'}.get(name)


def make_registry():
    import ijson
    registry = read_json(B / 'registry.json')
    root = Path('/mnt/public_data/nuScenes/v1.0-trainval')
    categories = {r['token']: r['name'] for r in read_json(root / 'category.json')}
    instances = {r['token']: categories[r['category_token']] for r in read_json(root / 'instance.json')}
    indexes = {scene: read_json(B / f'runs/nuscenes10/debug/dataset_scenes/{scene}.json')
               for scene in registry['nuscenes']}
    wanted = {f['sample_token'] for index in indexes.values() for f in index['frames']}
    annotations = defaultdict(list)
    with (root / 'sample_annotation.json').open('rb') as handle:
        for row in ijson.items(handle, 'item', use_float=True):
            if row['sample_token'] in wanted and nusc_category(instances[row['instance_token']]):
                annotations[row['sample_token']].append(row)
    for scene, index in indexes.items():
        records = []
        for frame in index['frames']:
            with Image.open(frame['path']) as image:
                width, height = image.size
            for row in annotations[frame['sample_token']]:
                transform = np.eye(4)
                transform[:3, :3] = rotation(row['rotation'])
                transform[:3, 3] = row['translation']
                size = [row['size'][1], row['size'][0], row['size'][2]]
                bbox = project_box(transform, size, frame, width, height)
                if bbox is None:
                    continue
                records.append(dict(id=f'nuscenes:{scene}:{row["instance_token"]}',
                    category=nusc_category(instances[row['instance_token']]),
                    native_category=instances[row['instance_token']], annotation_token=row['token'],
                    source_image=frame['path'], camera_name=frame['camera_name'], frame_index=frame['frame_id'],
                    bbox_xyxy=bbox, visibility_token=row['visibility_token'],
                    object_to_world=transform.tolist(), size_lwh=size,
                    camera_intrinsic=frame['camera_intrinsic'], camera_to_world=frame['camera_to_world'],
                    pose_link='native_sample_annotation', source_geometry='projected_3D_cuboid'))
        write_json(B / f'evaluation/ground_truth/{scene}.json', dict(dataset='nuscenes', scene_id=scene,
            scope='Annotated in-frustum objects. Cuboid boxes are not per-camera visible masks; occlusion is not removed.',
            identity_scope='native instance token', records=records))
        print(scene, len(records), 'boxes', len({r['id'] for r in records}), 'identities', flush=True)
    for scene_info in registry['waymo']:
        scene = scene_info['scene_id']
        index = read_json(B / f'inputs/waymo_indexes/{scene}.json')
        frames = {f['path']: f for f in index['frames']}
        records = []
        with (B / f'evaluation/waymo_native/{scene}.jsonl').open() as handle:
            for line in handle:
                native = json.loads(line)
                lidar = {r['id']: r for r in native['laser_labels']}
                camera_rows = [dict(row, source_image=camera['path'], camera_name=camera['camera_name'])
                               for camera in native.get('camera_annotations', []) for row in camera['annotations']]
                for row in camera_rows:
                    # Cyclist labels include a person, so are not a bicycle/motorcycle asset GT.
                    category = {1: 'vehicle', 3: 'traffic_sign'}.get(row['type'])
                    if category is None:
                        continue
                    box = row['bbox_xyxy']
                    if box[2]-box[0] < 8 or box[3]-box[1] < 12:
                        continue
                    frame = frames[row['source_image']]
                    association = row.get('laser_object_id')
                    identity = f'laser_association:{association}' if association else f'camera:{row["camera_name"]}:{row["camera_track_id"]}'
                    record = dict(id=f'waymo:{scene}:{identity}', category=category,
                        native_category=str(row['type']), source_image=row['source_image'],
                        camera_name=row['camera_name'], frame_index=frame['frame_id'], bbox_xyxy=box,
                        camera_intrinsic=frame['camera_intrinsic'], camera_to_world=frame['camera_to_world'],
                        identity_scope='laser_association' if association else 'camera_track',
                        pose_link=None, source_geometry='rectified_native_camera_box')
                    if association in lidar:
                        cx, cy, cz, length, width, height, yaw = lidar[association]['box']
                        local = np.eye(4)
                        local[:3, :3] = rotation([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                        local[:3, 3] = [cx, cy, cz]
                        transform = np.asarray(native['ego_to_world']).reshape(4, 4) @ local
                        record.update(object_to_world=transform.tolist(), size_lwh=[length, width, height],
                            pose_link='native_laser_association', timing='Frame pose vs camera exposure timestamp approximation')
                    records.append(record)
        write_json(B / f'evaluation/ground_truth/{scene}.json', dict(dataset='waymo', scene_id=scene,
            scope='Native 2D vehicle/sign labels, rectified. Cyclist includes rider and is excluded from rigid coverage.',
            identity_scope='Native laser association when available; otherwise camera-scoped track ID.', records=records))
        print(scene, len(records), 'boxes', len({r['id'] for r in records}), 'identities', flush=True)


def family(category):
    return 'vehicle' if category in VEHICLES else category


def match_boxes(rows, truth, threshold):
    if not rows or not truth:
        return []
    a, b = np.asarray([r['bbox_xyxy'] for r in rows]), np.asarray([r['bbox_xyxy'] for r in truth])
    size = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])-np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = size.prod(-1)
    area_a = np.maximum(0, a[:, 2:]-a[:, :2]).prod(-1)
    area_b = np.maximum(0, b[:, 2:]-b[:, :2]).prod(-1)
    iou = intersection / np.maximum(area_a[:, None]+area_b[None, :]-intersection, 1e-12)
    compatible = np.array([family(r['category']) for r in rows])[:, None] == np.array([family(r['category']) for r in truth])[None, :]
    eligible = compatible & (iou >= threshold)
    # Maximize match count first, then IoU; stable input order resolves exact ties.
    score = np.where(eligible, min(len(rows), len(truth))+1+iou, 0)
    rr, tt = linear_sum_assignment(-score)
    return [(int(r), int(t), float(iou[r, t])) for r, t in zip(rr, tt) if eligible[r, t]]


def scene_matches(scene, run_dir=None, output_dir=None):
    gt = read_json(B / f'evaluation/ground_truth/{scene}.json')
    dataset = gt['dataset']
    run = Path(run_dir) if run_dir is not None else B / f'runs/{dataset}10'
    routes_path = run / 'handoff/asset_routes.json'
    if not routes_path.exists():
        raise FileNotFoundError('Scene matching requires finalized query routes')
    truth_by_image = defaultdict(list)
    for row in gt['records']:
        truth_by_image[row['source_image']].append(row)
    detection_path = run / f'scenes/{scene}/logs/detections.json'
    detections = read_json(detection_path)
    query_rows = [r for r in read_json(routes_path)['assets'] if r['scene_id'] == scene]
    queries = [dict(r['source'], category=r['category'], obj_id=r['obj_id']) for r in query_rows]
    all_source = {r['asset_id']: r for r in read_json(run / f'scenes/{scene}/logs/export_items_full.json')}
    results = {}
    for threshold in [.5, .25]:
        outcome = {}
        for stage, items in [('detector', detections), ('query', queries)]:
            by_image = defaultdict(list)
            for row in items:
                by_image[row.get('source_image', row.get('image_path'))].append(row)
            linked = []
            for image, rows in sorted(by_image.items()):
                truth = sorted(truth_by_image[image], key=lambda r: r['id'])
                rows = sorted(rows, key=lambda r: (r.get('obj_id', ''), r['category'], r['bbox_xyxy']))
                for ri, ti, iou in match_boxes(rows, truth, threshold):
                    row, target = rows[ri], truth[ti]
                    link = dict(obj_id=row.get('obj_id'), gt_id=target['id'], category=target['category'],
                                iou=iou, source_image=image)
                    if stage == 'query':
                        source = all_source[row['obj_id']]
                        anchor = next((r for r in source['real_images'] if r['source_image'] == image and
                                       np.allclose(r['bbox_xyxy'], row['bbox_xyxy'], atol=.01)), None)
                        if anchor is None:
                            raise ValueError('Source SAM mask is unavailable: '+row['obj_id'])
                        link.update(source_rgba=anchor['rgba_path'], source_rgba_sha256=sha256(anchor['rgba_path']),
                                    native_annotation=target)
                    linked.append(link)
            identities = {r['gt_id'] for r in linked}
            outcome[stage] = dict(matched_identities=len(identities), matches=linked)
        results[str(threshold)] = outcome
    ids = {r['id'] for r in gt['records']}
    report = dict(dataset=dataset, scene_id=scene, denominator_identities=len(ids),
                  denominator_by_category=dict(Counter(next(r['category'] for r in gt['records'] if r['id']==i) for i in ids)),
                  native_scope=gt['scope'], identity_scope=gt['identity_scope'],
                  thresholds=results, queries=len(queries), query_routes_sha256=sha256(routes_path))
    destination=Path(output_dir) if output_dir is not None else B/'evaluation/matches'
    write_json(destination / f'{scene}.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['registry', 'match'])
    parser.add_argument('--scene')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.action == 'registry':
        make_registry()
    elif args.scene:
        print(json.dumps({k:v for k,v in scene_matches(args.scene,args.run_dir,args.output_dir).items() if k!='thresholds'}))
    else:
        raise ValueError('Specify a scene for matching')
