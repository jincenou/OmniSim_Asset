"""Known-camera adapter and complete dynamic asset loading for Grid4D."""
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from gpu_device import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'random8/r20261015_113332'
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from gaussian_renderer import render
from utils.graphics_utils import BasicPointCloud, focal2fov

PARAMETERS = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def manifest(label):
    assert label in ['N01', 'N02', 'N03', 'N04', 'N06', 'N07', 'N08', 'N11']
    path = BASE / 'four_dg/data' / label / 'fdanyone.json'
    data = json.loads(path.read_text())
    assert data['heldout_views'] == [0, 8, 16]
    assert data['train_views'] == [v for v in range(24) if v not in [0, 8, 16]]
    assert data['time_denominator'] == data['frame_count'] - 1
    assert (data['width'], data['height']) == (352, 640)
    return data, path


def camera(data, view, frame, rgb=None):
    record = next(c for c in data['cameras'] if c['view'] == view)
    K = np.asarray(record['K'], dtype=np.float64)
    w = np.linalg.inv(np.asarray(record['camtoworld'], dtype=np.float64))
    if rgb is None:
        rgb = torch.zeros(3, data['height'], data['width'], device='cuda')
    result = Camera(colmap_id=view, R=w[:3, :3].T, T=w[:3, 3],
        FoVx=focal2fov(K[0, 0], data['width']), FoVy=focal2fov(K[1, 1], data['height']),
        image=rgb, gt_alpha_mask=None, image_name=f'cam{view:02d}/time{frame:.6f}',
        uid=view * data['frame_count'] + frame, data_device='cuda', fid=frame / data['time_denominator'])
    result.projection_matrix[2, 0] = (2 * K[0, 2] + 1) / data['width'] - 1
    result.projection_matrix[2, 1] = (2 * K[1, 2] + 1) / data['height'] - 1
    result.full_proj_transform = result.world_view_transform @ result.projection_matrix
    result.source_view, result.source_frame = view, frame
    return result


class KnownScene:
    def __init__(self, dataset, gaussians):
        self.model_path = dataset.model_path
        self.gaussians = gaussians
        self.data, path = manifest(dataset.label)
        assert dataset.eval is True and dataset.background_value == .5
        with np.load(path.parent / self.data['canonical_npz']) as c:
            cloud = BasicPointCloud(points=c['points'].copy(), colors=c['colors'].copy(), normals=np.zeros_like(c['points']))
        centers = np.asarray([c['camtoworld'] for c in self.data['cameras'] if c['view'] in self.data['train_views']])[:, :3, 3]
        self.cameras_extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        self.gaussians.create_from_pcd(cloud, self.cameras_extent)
        self.train_cameras = []
        self.loaded_paths = []
        for view in self.data['train_views']:
            for frame in range(self.data['frame_count']):
                image_path = Path(self.data['source_data']) / 'images' / f'cam{view:02d}' / f'{frame:06d}.png'
                with Image.open(image_path) as image:
                    assert image.mode == 'RGBA' and image.size == (352, 640)
                    rgba = torch.from_numpy(np.asarray(image).copy()).to('cuda', dtype=torch.float32).permute(2, 0, 1) / 255
                rgb = rgba[:3] * rgba[3:] + dataset.background_value * (1 - rgba[3:])
                cam = camera(self.data, view, frame, rgb)
                cam.foreground_pixels = int((rgba[3] > .5).sum())
                self.train_cameras.append(cam)
                self.loaded_paths.append(str(image_path))
        assert len(self.train_cameras) == self.data['frame_count'] * 21
        assert not any(f'/cam{v:02d}/' in p for p in self.loaded_paths for v in [0, 8, 16])
        save(Path(self.model_path) / 'training_image_paths.json', self.loaded_paths)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return []

    def save(self, iteration, is_best=False):
        assert not is_best
        target = Path(self.model_path) / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        assert not target.exists()
        self.gaussians.save_ply(str(target))


def new_deform(config, regularize=False):
    d, opt = config['dataset'], config['optimization']
    return DeformModel(grid_args=d['grid_args'], net_args=d['network_args'], scale_xyz=d['scale_xyz'],
        spatial_downsample_ratio=opt['spatial_downsample_ratio'], spatial_perturb_range=opt['spatial_perturb_range'],
        temporal_downsample_ratio=opt['temporal_downsample_ratio'], temporal_perturb_range=opt['temporal_perturb_range'],
        reg_spatial_able=regularize and opt['lambda_spatial_tv'] > 0,
        reg_temporal_able=regularize and opt['lambda_temporal_tv'] > 0)


def raw_render(model, deform, pipe, cam, background=0.):
    time = cam.fid.unsqueeze(0).expand(len(model.get_xyz), -1)
    warp = deform.step(model.get_xyz.detach(), time)
    image = render(cam, model, pipe, torch.full((3,), background, device='cuda'),
                   warp['d_xyz'], warp['d_rotation'], warp['d_scaling'])['render']
    white = render(cam, model, pipe, torch.zeros(3, device='cuda'),
                   warp['d_xyz'], warp['d_rotation'], warp['d_scaling'], override_color=torch.ones_like(model.get_xyz))['render']
    assert float((white.max(0).values - white.min(0).values).max()) < 1e-5
    return image, white[:1]


def load_asset(run, iteration):
    config = json.loads((run / 'config.json').read_text())
    model = GaussianModel(config['dataset']['sh_degree'])
    model.load_ply(str(run / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'))
    snapshot = json.loads((run / 'snapshots' / f'{iteration}.json').read_text())
    model.active_sh_degree = snapshot['active_sh_degree']
    deform = new_deform(config)
    # Native loader restores BOTH hash encoders and deformation network, strict by default.
    deform.load_weights(str(run), iteration)
    deform.grid4d.eval()
    deform.deform.eval()
    return model, deform, SimpleNamespace(**config['pipeline']), config


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
