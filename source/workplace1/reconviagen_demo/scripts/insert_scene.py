"""Insert a 3DGS asset into a trained SplatAD scene and render 6-camera video.

The asset (canonical PLY: z-up, +x forward, base at z=0) follows a planned
trajectory in raw nuScenes *global* coordinates (traj_pipeline NPZ). Per
camera image the asset is posed in global, grounded on a lidar+gaussian
height field (same GroundModel as the OmniRe insert), transformed into
SplatAD's model world (dataparser w2m), and its gaussians are appended right
before gsplat rasterization, so occlusion is handled by depth sorting.

Color adaptation: SplatAD colors are 16-dim learned features decoded by a
per-scene CNN (albedo = first 3 channels, spec residual from the rest). We
put the asset RGB into the albedo slot and zero the 13 residual channels —
the decoder then renders `albedo * (1 + spec3) + spec3:` where spec is
driven by zeros + ray dirs, i.e. a mild learned view-dependent tint. Good
enough for a smoke test; proper feature fitting is a later step.

Usage (env `splatad`, from /mnt/ojc/splatad_repro):
    python /mnt/wmj/workspace4/Recon/splatad_insert/insert_asset_splatad.py \
        --config outputs/full_scene0104/unnamed/splatad/2026-07-12_151931/config.yml \
        --traj_npz .../scene-0104_V5_merge_in_traj.npz \
        --asset_ply .../car/t000098.ply \
        --out_dir /mnt/wmj/workspace4/Recon/splatad_insert/out/scene-0104_V5_t000098
"""
import argparse
import json
import os
import sys
from pathlib import Path

# For ojc-trained ckpts prefer the training-time sources (copied from his
# splatad_repro); our own ckpts are trained with the env's nerfstudio, so the
# copied sources are opt-in via --use_ojc_src (parsed early, before imports).
_HERE = os.path.dirname(os.path.abspath(__file__))
if "--use_ojc_src" in sys.argv:
    sys.argv.remove("--use_ojc_src")
    sys.path.insert(0, os.path.join(_HERE, "src", "neurad-studio"))
    sys.path.insert(0, os.path.join(_HERE, "src", "splatad"))

import numpy as np
import torch
import imageio.v2 as imageio

C0 = 0.28209479177387814
TARGET_LENGTH_M = 4.6


# --------------------------------------------------------------------------
# asset (same as OmniRe insert)
# --------------------------------------------------------------------------

def load_canonical_ply(path, target_len=TARGET_LENGTH_M):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    op = np.asarray(v["opacity"], dtype=np.float64)
    sc = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)

    ext = np.percentile(means[:, 0], 99.5) - np.percentile(means[:, 0], 0.5)
    scale = target_len / max(ext, 1e-6)
    means *= scale
    sc = np.exp(sc) * scale
    means[:, 2] -= np.percentile(means[:, 2], 0.5)
    rgb = np.clip(dc * C0 + 0.5, 0.0, 1.0)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    return {"means": means, "quats": rot, "scales": sc,
            "opacities": 1.0 / (1.0 + np.exp(-op)), "rgbs": rgb,
            "metric_scale": scale}


def rotmat_to_quat(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_left_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def interp_traj(ts_grid, traj, t):
    valid = ~np.isnan(traj[:, 0])
    tg, tr = ts_grid[valid], traj[valid]
    if t < tg[0] or t > tg[-1]:
        return None
    x = np.interp(t, tg, tr[:, 0])
    y = np.interp(t, tg, tr[:, 1])
    yaw = np.interp(t, tg, np.unwrap(tr[:, 2]))
    v = np.interp(t, tg, tr[:, 3]) if tr.shape[1] > 3 else 0.0
    return float(x), float(y), float(yaw), float(v)


# --------------------------------------------------------------------------
# 4DGS human asset (walking clip, de-baked root motion; mirrors
# traj_pipeline corner_case/assets.py::bind_human)
# --------------------------------------------------------------------------

UP_FIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
ASSET_FPS = 30.0
NATIVE_WALK_SPEED = 1.4
IDLE_PLAYBACK_RATE = 0.15
HUMAN_HEIGHT_M = 1.75
YAW_SLEW_RAD_S = np.radians(240.0)


class HumanAsset:
    """Per-frame 4DGS playback posed on the planned trajectory.

    Gait phase advances with the planned speed; the clip's own root yaw and
    horizontal drift (from the SMPL-X fit) are subtracted so path and heading
    come solely from the plan.
    """

    def __init__(self, npz_path, smplx_dir, target_h=HUMAN_HEIGHT_M):
        z = np.load(npz_path)
        self.fx = z["frame_xyz"]        # (F, N, 3) float16, camera frame
        self.fr = z["frame_rotation"]   # (F, N, 4)
        self.fs = z["frame_scaling"]    # (F, N, 3) activated
        self.opac = z["opacity"].astype(np.float32).squeeze(-1)
        assert int(z["use_rgb"]) == 1
        self.rgb = z["shs"].astype(np.float32).reshape(len(self.opac), 3)

        y = z["canonical_xyz"][:, 1].astype(np.float64)
        lo, hi = np.percentile(y, [1.0, 99.0])
        self.scale = target_h / float(hi - lo)

        self.yaw_baked, self.trans_baked = smplx_root_motion(smplx_dir)
        self.F = min(len(self.fx), len(self.yaw_baked))

        # The clip lives in the source video's camera frame; UP_FIX assumes
        # that camera was level, but the SMPL-X track carries a baked,
        # frame-varying root pitch/roll. Measure the body "up" axis
        # (feet->head) per frame, smooth it, and build per-frame corrective
        # rotations onto +z so the person stays upright.
        def smooth(a, k=9):
            k = min(k, self.F if self.F % 2 else self.F - 1)
            pad = k // 2
            p = np.pad(a, ((pad, pad),) + ((0, 0),) * (a.ndim - 1),
                       mode="edge")
            return np.array([np.median(p[i:i + k], axis=0)
                             for i in range(self.F)])

        ups = []
        for f in range(self.F):
            p = self.fx[f].astype(np.float64) @ UP_FIX.T
            zq = p[:, 2]
            lo_m = p[zq <= np.percentile(zq, 5.0)].mean(axis=0)
            hi_m = p[zq >= np.percentile(zq, 95.0)].mean(axis=0)
            ups.append(hi_m - lo_m)
        ups = smooth(np.array(ups))
        ups /= np.linalg.norm(ups, axis=1, keepdims=True)
        self.R_lean = np.zeros((self.F, 3, 3))
        for f in range(self.F):
            up = ups[f]
            v = np.cross(up, [0.0, 0.0, 1.0])
            s, c = np.linalg.norm(v), up[2]
            if s < 1e-8:
                self.R_lean[f] = np.eye(3)
            else:
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]]) / s
                self.R_lean[f] = np.eye(3) + s * K + (1 - c) * (K @ K)
        tilt = np.degrees(np.arccos(np.clip(ups[:, 2], -1, 1)))
        print(f"lean fix: baked tilt mean/max {tilt.mean():.1f}/"
              f"{tilt.max():.1f} deg")

        # per-frame foot height (same transform chain as gaussians(): de-bake
        # translation FIRST, then lean fix), smoothed: removes baked vertical
        # drift without fighting the gait bounce.
        foot = np.zeros(self.F)
        for f in range(self.F):
            p = self.fx[f].astype(np.float64) @ UP_FIX.T
            p[:, :2] -= self.trans_baked[f]
            foot[f] = np.percentile(p @ self.R_lean[f].T[:, 2], 1.0)
        self.foot_z = smooth(foot[:, None]).squeeze(-1) * self.scale

        self.phase = 0.0
        self.t_prev = None
        self.yaw_prev = None

    def advance(self, t, v, yaw_target):
        """Update gait phase and slew-limited heading at epoch time t."""
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        rate = max(abs(v) / NATIVE_WALK_SPEED, IDLE_PLAYBACK_RATE)
        self.phase = (self.phase + rate * ASSET_FPS * dt) % self.F
        if self.yaw_prev is None:
            self.yaw_prev = yaw_target
        else:
            err = (yaw_target - self.yaw_prev + np.pi) % (2 * np.pi) - np.pi
            self.yaw_prev += np.clip(err, -YAW_SLEW_RAD_S * dt,
                                     YAW_SLEW_RAD_S * dt)
        return int(self.phase)

    def gaussians(self, f, x, y, z_ground):
        """means/quats/scales (np) for clip frame f placed at (x, y)."""
        pts = self.fx[f, :, :].astype(np.float64) @ UP_FIX.T
        pts[:, :2] -= self.trans_baked[f]
        pts = pts @ self.R_lean[f].T   # cancel baked root pitch/roll
        pts *= self.scale
        delta = self.yaw_prev - self.yaw_baked[f]
        c, s = np.cos(delta), np.sin(delta)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        means = pts @ Rz.T
        means[:, 0] += x
        means[:, 1] += y
        means[:, 2] += z_ground - self.foot_z[f]
        R = Rz @ self.R_lean[f] @ UP_FIX
        quats = quat_left_mul(rotmat_to_quat(R),
                              self.fr[f].astype(np.float64))
        scales = self.fs[f].astype(np.float32) * self.scale
        return means, quats, scales


def smplx_root_motion(smplx_dir):
    """Baked (world yaw, horizontal xy translation) per clip frame."""
    from scipy.spatial.transform import Rotation
    files = sorted(Path(smplx_dir).glob("*.json"))
    frames = [json.loads(f.read_text()) for f in files]
    root = np.array([f["root_pose"] for f in frames], dtype=np.float64)
    trans = np.array([f["trans"] for f in frames], dtype=np.float64)
    forward = Rotation.from_rotvec(root).apply(
        np.tile([0.0, 0.0, 1.0], (len(root), 1)))
    yaw_world = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
    return yaw_world, trans[:, [0, 2]]


# --------------------------------------------------------------------------
# ground model (same as OmniRe insert, slope-aware v1)
# --------------------------------------------------------------------------

class GroundModel:
    CELL = 0.4
    MAX_TILT_DEG = 6.0

    def __init__(self, lidar_pts, gs_pts=None, band_m=0.35):
        from scipy import ndimage

        pts = lidar_pts
        self.x0 = pts[:, 0].min() - 2.0
        self.y0 = pts[:, 1].min() - 2.0
        nx = int(np.ceil((pts[:, 0].max() + 2.0 - self.x0) / self.CELL)) + 1
        ny = int(np.ceil((pts[:, 1].max() + 2.0 - self.y0) / self.CELL)) + 1

        def rasterize(p):
            ix = ((p[:, 0] - self.x0) / self.CELL).astype(int)
            iy = ((p[:, 1] - self.y0) / self.CELL).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            ix, iy, z = ix[ok], iy[ok], p[ok, 2]
            order = np.lexsort((z, iy * nx + ix))
            cell = (iy * nx + ix)[order]
            zs = z[order]
            first = np.searchsorted(cell, np.unique(cell))
            counts = np.diff(np.append(first, len(cell)))
            k = first + np.maximum((counts * 0.15).astype(int) - 1, 0)
            grid = np.full((ny, nx), np.nan)
            uc = cell[first]
            grid[uc // nx, uc % nx] = zs[k]
            return grid

        grid = rasterize(pts)
        if gs_pts is not None and len(gs_pts):
            gz = self._lookup(grid, gs_pts[:, 0], gs_pts[:, 1])
            keep = np.abs(gs_pts[:, 2] - gz) < band_m
            keep &= ~np.isnan(gz)
            if keep.sum() > 100:
                g2 = rasterize(np.concatenate([pts, gs_pts[keep]], axis=0))
                grid = np.where(np.isnan(g2), grid, g2)
            print(f"ground: {keep.sum()}/{len(gs_pts)} road gaussians kept")

        mask = np.isnan(grid)
        if mask.any():
            idx = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True)
            grid = grid[tuple(idx)]
        self.grid = ndimage.gaussian_filter(grid, sigma=1.0)

    def _lookup(self, grid, x, y):
        ny, nx = grid.shape
        ix = np.clip(((x - self.x0) / self.CELL).astype(int), 0, nx - 1)
        iy = np.clip(((y - self.y0) / self.CELL).astype(int), 0, ny - 1)
        return grid[iy, ix]

    def z(self, x, y):
        gx = np.clip((x - self.x0) / self.CELL, 0, self.grid.shape[1] - 1.001)
        gy = np.clip((y - self.y0) / self.CELL, 0, self.grid.shape[0] - 1.001)
        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0
        g = self.grid
        return float(g[y0, x0] * (1 - fx) * (1 - fy)
                     + g[y0, x0 + 1] * fx * (1 - fy)
                     + g[y0 + 1, x0] * (1 - fx) * fy
                     + g[y0 + 1, x0 + 1] * fx * fy)

    def contact_pose(self, x, y, yaw, length=4.2, width=2.0):
        c, s = np.cos(yaw), np.sin(yaw)
        fwd2, left2 = np.array([c, s]), np.array([-s, c])
        u = np.linspace(-length / 2, length / 2, 7)
        v = np.linspace(-width / 2, width / 2, 4)
        uu, vv = np.meshgrid(u, v)
        xy = (np.array([x, y]) + uu.ravel()[:, None] * fwd2
              + vv.ravel()[:, None] * left2)
        P = np.array([[px, py, self.z(px, py)] for px, py in xy])

        def fit(P):
            A = np.column_stack([P[:, 0] - x, P[:, 1] - y, np.ones(len(P))])
            coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
            return coef

        a, b, d = fit(P)
        resid = np.abs(P[:, 2] - (a * (P[:, 0] - x) + b * (P[:, 1] - y) + d))
        keep = resid < max(2.0 * np.median(resid), 0.03)
        if keep.sum() >= 6:
            a, b, d = fit(P[keep])
        max_g = np.tan(np.radians(self.MAX_TILT_DEG))
        g = np.hypot(a, b)
        if g > max_g:
            a, b = a * max_g / g, b * max_g / g
        n = np.array([-a, -b, 1.0])
        n /= np.linalg.norm(n)
        fwd = np.array([c, s, 0.0])
        fwd = fwd - fwd.dot(n) * n
        fwd /= np.linalg.norm(fwd)
        left = np.cross(n, fwd)
        return float(d), np.column_stack([fwd, left, n])


# --------------------------------------------------------------------------
# nuScenes lidar (global frame) via devkit
# --------------------------------------------------------------------------

def load_scene_lidar_global(dataroot, version, scene_name):
    from nuscenes import NuScenes
    from pyquaternion import Quaternion
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    # walk the full LIDAR_TOP sample_data chain (keyframes + sweeps)
    sd = nusc.get("sample_data",
                  nusc.get("sample", scene["first_sample_token"])["data"]["LIDAR_TOP"])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    pts_all, ts_all = [], []
    while True:
        fn = os.path.join(dataroot, sd["filename"])
        if os.path.exists(fn):
            pc = np.fromfile(fn, dtype=np.float32).reshape(-1, 5)[:, :3]
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            R_l2e = Quaternion(cs["rotation"]).rotation_matrix
            R_e2g = Quaternion(ep["rotation"]).rotation_matrix
            p = pc @ R_l2e.T + np.array(cs["translation"])
            p = p @ R_e2g.T + np.array(ep["translation"])
            pts_all.append(p.astype(np.float64))
            ts_all.append(sd["timestamp"] / 1e6)
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])

    # min sensor epoch across all cameras + lidar (matches ad_dataparser's
    # min_time shift, which uses every sample_data of every stream)
    min_t = min(ts_all)
    for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
        sd = nusc.get("sample_data",
                      nusc.get("sample", scene["first_sample_token"])["data"][cam])
        while sd["prev"]:
            sd = nusc.get("sample_data", sd["prev"])
        while True:
            min_t = min(min_t, sd["timestamp"] / 1e6)
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
    return np.concatenate(pts_all, axis=0), min_t


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traj_npz", required=True)
    ap.add_argument("--asset_ply", default=None,
                    help="rigid canonical PLY (cars, props)")
    ap.add_argument("--human_npz", default=None,
                    help="4DGS human asset_4dgs.npz (instead of --asset_ply)")
    ap.add_argument("--smplx_dir", default=None,
                    help="smplx_params dir of the human clip (root de-bake)")
    ap.add_argument("--human_height", type=float, default=HUMAN_HEIGHT_M)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=float, default=TARGET_LENGTH_M)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--downscale", type=int, default=2,
                    help="tile downscale for the output video")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    assert (args.asset_ply is None) != (args.human_npz is None), \
        "pass exactly one of --asset_ply / --human_npz"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "frames"), exist_ok=True)

    from nerfstudio.utils.eval_utils import eval_setup
    import nerfstudio.models.splatad as splatad_mod
    import nerfstudio.data.datasets.base_dataset as base_dataset_mod

    # GT images are unreadable (root-owned); rendering only needs cameras.
    _orig_get_numpy = base_dataset_mod.InputDataset.get_numpy_image

    def _safe_get_numpy(self, image_idx):
        try:
            return _orig_get_numpy(self, image_idx)
        except PermissionError:
            h = int(self.cameras.height[image_idx].item())
            w = int(self.cameras.width[image_idx].item())
            return np.zeros((h, w, 3), dtype=np.uint8)

    base_dataset_mod.InputDataset.get_numpy_image = _safe_get_numpy

    # strict_load=False: the ckpt's lidar_decoder is a tcnn MLP while our env
    # falls back to a torch MLP (no tinycudann). Camera rendering never calls
    # the lidar decoder, so mismatched/randomly-init lidar weights are fine.
    load_audit = []
    original_load = torch.nn.Module.load_state_dict
    def audited_load(self, state_dict, *a, **kw):
        result = original_load(self, state_dict, *a, **kw)
        if type(self).__name__ == "SplatADModel":
            record = dict(missing=list(result.missing_keys), unexpected=list(result.unexpected_keys))
            load_audit.append(record)
            invalid = [k for k in record['missing'] + record['unexpected'] if not k.startswith('lidar_decoder.')]
            if invalid:
                raise RuntimeError(f"Camera/background checkpoint mismatch: {invalid}")
        return result
    torch.nn.Module.load_state_dict = audited_load
    config, pipeline, ckpt_path, step = eval_setup(
        Path(args.config), test_mode="test", strict_load=False)
    torch.nn.Module.load_state_dict = original_load
    if not load_audit:
        raise RuntimeError("Checkpoint compatibility audit did not execute")
    with open(os.path.join(args.out_dir, "checkpoint_load_audit.json"), "w") as f:
        json.dump(load_audit, f, indent=2)
    model = pipeline.model
    device = model.device
    print(f"loaded {ckpt_path} (step {step})")

    dpo = pipeline.datamanager.train_dataparser_outputs
    if not np.isclose(float(dpo.dataparser_scale), 1.0):
        raise ValueError("Non-unit dataparser scale requires explicit similarity transform")
    w2m = dpo.dataparser_transform.cpu().numpy().astype(np.float64)  # (3,4)
    R_w2m, t_w2m = w2m[:, :3], w2m[:, 3]
    m2w_R = R_w2m.T
    m2w_t = -m2w_R @ t_w2m

    seq = config.pipeline.datamanager.dataparser.sequence
    dataroot = str(config.pipeline.datamanager.dataparser.data)
    version = config.pipeline.datamanager.dataparser.version
    print(f"scene {seq}, dataroot {dataroot}")

    # ---- trajectory (nuScenes global) ----
    d = np.load(args.traj_npz, allow_pickle=True)
    traj, ts_grid = d["traj"], d["ts_grid"]
    print("traj meta:", str(d.get("meta", "")))

    # ---- ground model in global frame: lidar + road gaussians ----
    lidar_pts, min_time = load_scene_lidar_global(dataroot, version, seq)
    print(f"lidar: {len(lidar_pts)} pts, min sensor epoch {min_time:.3f}")

    with torch.no_grad():
        ids = model.gauss_params["id"].cpu().numpy().squeeze(-1)
        num_actors = model.dynamic_actors.actor_sizes.shape[0]
        static = ids >= num_actors
        means_w = model.gauss_params["means"].detach().cpu().numpy()[static]
        scales_w = np.exp(model.gauss_params["scales"].detach().cpu().numpy()[static])
        quats_w = model.gauss_params["quats"].detach().cpu().numpy()[static]
    q = quats_w / np.linalg.norm(quats_w, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a_min = np.argmin(scales_w, axis=1)
    cols = np.stack([
        np.stack([1 - 2 * (y**2 + z**2), 2 * (x*y + w*z), 2 * (x*z - w*y)], 1),
        np.stack([2 * (x*y - w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z + w*x)], 1),
        np.stack([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x**2 + y**2)], 1),
    ], axis=1)
    normal_w = cols[np.arange(len(q)), a_min]
    normal_g = normal_w @ m2w_R.T
    s_sorted = np.sort(scales_w, axis=1)
    pancake = s_sorted[:, 0] < 0.5 * s_sorted[:, 1]
    flat = np.abs(normal_g[:, 2]) > 0.85
    gs_pts = (means_w[pancake & flat] @ m2w_R.T) + m2w_t
    print(f"ground: {len(gs_pts)}/{len(means_w)} flat-lying gaussian candidates")
    ground = GroundModel(lidar_pts, gs_pts)

    # ---- asset ----
    feature_dim = model.config.feature_dim
    human = None
    if args.human_npz:
        smplx_dir = args.smplx_dir
        if smplx_dir is None:  # infer from asset pipeline layout
            uid = Path(args.human_npz).parent.name
            smplx_dir = (f"/mnt/wmj/workspace4/pipeline/output/"
                         f"human_3k_smplx_clean/{uid}/smplx_params")
        human = HumanAsset(args.human_npz, smplx_dir,
                           target_h=args.human_height)
        print(f"human asset: {len(human.opac)} gaussians, {human.F} clip "
              f"frames, scale={human.scale:.4f}, "
              f"foot_z={human.foot_z.mean():.3f}")
        rgbs = human.rgb
        a_opac = torch.tensor(human.opac, dtype=torch.float32, device=device)
    else:
        asset = load_canonical_ply(args.asset_ply, target_len=args.target_len)
        print(f"asset: {len(asset['means'])} gaussians, "
              f"metric_scale={asset['metric_scale']:.3f}")
        rgbs = asset["rgbs"]
        a_scales = torch.tensor(asset["scales"], dtype=torch.float32,
                                device=device)
        a_opac = torch.tensor(asset["opacities"], dtype=torch.float32,
                              device=device)
    # ---- color calibration ----
    # With zero residual features the trained decoder maps albedo through a
    # roughly affine transform (out ~= k1 * albedo + k2, k1 << 1), washing
    # colors to gray. Probe the decoder and pre-distort the asset RGB so the
    # decoded color lands on the target.
    with torch.no_grad():
        ad = model.config.appearance_dim
        dirs = torch.randn(256, 3, device=device)
        dirs = (dirs / dirs.norm(dim=1, keepdim=True)).view(1, 16, 16, 3)
        ae = model.appearance_embedding.weight.mean(0)
        outs = []
        for alb in (0.0, 1.0):
            feats = torch.zeros(1, 16, 16, 3 + feature_dim + ad,
                                device=device)
            feats[..., :3] = alb
            feats[..., 3 + feature_dim:] = ae
            outs.append(model.rgb_decoder(feats, dirs).mean(dim=(0, 1, 2)))
        k2 = outs[0]
        k1 = outs[1] - outs[0]
        print(f"color calib: k1={k1.cpu().numpy().round(3)}, "
              f"k2={k2.cpu().numpy().round(3)}")
    rgb_t = torch.tensor(rgbs, dtype=torch.float32, device=device)
    # Preserve a negative decoder slope. Clamping it to a positive value
    # inverted vehicle colors in the audited scene-0012 checkpoint.
    denominator = torch.where(k1 >= 0, k1.clamp(min=0.01), k1.clamp(max=-0.01))
    albedo_fit = ((rgb_t - k2) / denominator).clamp(-20.0, 20.0)
    with open(os.path.join(args.out_dir, "color_calibration.json"), "w") as f:
        json.dump(dict(k1=k1.cpu().tolist(), k2=k2.cpu().tolist(),
            method="signed affine inverse; abs denominator floor .01; albedo clipped +/-20",
            limitation="mean-direction original decoder calibration, not learned asset features"), f, indent=2)
    a_colors = torch.cat([
        albedo_fit,
        torch.zeros(len(rgbs), feature_dim, device=device),
    ], dim=1)

    # ---- collect cameras (train + eval), group by CAM_FRONT time ----
    cams_list = []
    for ds in (pipeline.datamanager.train_dataset,
               getattr(pipeline.datamanager, "eval_dataset", None)):
        if ds is None or len(ds.cameras) == 0:
            continue
        c = ds.cameras
        for i in range(len(c)):
            cams_list.append(c[i:i + 1])
    sensor_names = dpo.metadata["sensor_idx_to_name"]
    print(f"{len(cams_list)} cameras, sensors: {sensor_names}")

    def cam_time(c):
        return float(c.times.item())

    def cam_sensor(c):
        return int(c.metadata["sensor_idxs"].item())

    by_sensor = {}
    for c in cams_list:
        by_sensor.setdefault(cam_sensor(c), []).append(c)
    for k in by_sensor:
        by_sensor[k].sort(key=cam_time)

    # sensor names may or may not carry the CAM_ prefix depending on parser
    name_to_idx = {v.replace("CAM_", ""): k for k, v in sensor_names.items()}
    front_idx = name_to_idx["FRONT"]
    anchors = by_sensor[front_idx]
    anchors = anchors[args.start_frame:args.start_frame + args.max_frames]
    if not anchors:
        raise ValueError("No cameras in selected short trajectory")

    layout = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT",
              "BACK_LEFT", "BACK", "BACK_RIGHT"]
    layout_idx = [name_to_idx[n] for n in layout]

    # ---- patch rasterization to append asset gaussians ----
    state = {"means": None, "quats": None, "scales": None}
    orig_rast = splatad_mod.rasterization

    def patched(*a, **kw):
        if state["means"] is not None:
            sc = state["scales"] if state["scales"] is not None else a_scales
            kw["means"] = torch.cat([kw["means"], state["means"]], dim=0)
            kw["quats"] = torch.cat([kw["quats"], state["quats"]], dim=0)
            kw["scales"] = torch.cat([kw["scales"], sc], dim=0)
            kw["opacities"] = torch.cat([kw["opacities"], a_opac], dim=0)
            kw["colors"] = torch.cat([kw["colors"], a_colors], dim=0)
            if kw.get("velocities") is not None:
                kw["velocities"] = torch.cat(
                    [kw["velocities"],
                     torch.zeros(len(a_opac), 3, device=device)], dim=0)
        return orig_rast(*a, **kw)

    splatad_mod.rasterization = patched

    # ---- render loop ----
    if human is None:
        a_means_np = asset["means"]
        a_quats_np = asset["quats"]
    writer = None
    out_mp4 = os.path.join(args.out_dir, "insert_splatad_6cam.mp4")
    z_prev, n_prev = None, None
    tilt_log = []
    with torch.no_grad():
        for fi, anchor in enumerate(anchors):
            t0 = cam_time(anchor)
            tiles = []
            for si in layout_idx:
                cams = by_sensor[si]
                cam = min(cams, key=lambda c: abs(cam_time(c) - t0))
                t_epoch = cam_time(cam) + min_time
                pose = interp_traj(ts_grid, traj, t_epoch)
                if pose is None:
                    state["means"] = None
                elif human is not None:
                    px, py, yaw, v = pose
                    gz = ground.z(px, py)
                    z_prev = gz if z_prev is None else 0.5 * z_prev + 0.5 * gz
                    f = human.advance(t_epoch, v, yaw)
                    m, qv, sc = human.gaussians(f, px, py, z_prev)
                    m = m @ R_w2m.T + t_w2m
                    qv = quat_left_mul(rotmat_to_quat(R_w2m), qv)
                    state["means"] = torch.tensor(m, dtype=torch.float32,
                                                  device=device)
                    state["quats"] = torch.tensor(qv, dtype=torch.float32,
                                                  device=device)
                    state["scales"] = torch.tensor(sc, dtype=torch.float32,
                                                   device=device)
                else:
                    px, py, yaw, _v = pose
                    gz, R_body = ground.contact_pose(px, py, yaw, length=args.target_len,
                        width=float(np.quantile(asset['means'][:,1],.995)-np.quantile(asset['means'][:,1],.005)))
                    z_prev = gz if z_prev is None else 0.5 * z_prev + 0.5 * gz
                    n = R_body[:, 2]
                    n_prev = n if n_prev is None else 0.7 * n_prev + 0.3 * n
                    n_prev = n_prev / np.linalg.norm(n_prev)
                    fwd = R_body[:, 0] - R_body[:, 0].dot(n_prev) * n_prev
                    fwd /= np.linalg.norm(fwd)
                    R_g = np.column_stack([fwd, np.cross(n_prev, fwd), n_prev])
                    if si == front_idx:
                        tilt_log.append((
                            fi,
                            float(np.degrees(np.arcsin(-R_g[2, 0]))),
                            float(np.degrees(np.arcsin(R_g[2, 1])))))
                    R = R_w2m @ R_g
                    t = R_w2m @ np.array([px, py, z_prev]) + t_w2m
                    m = a_means_np @ R.T + t
                    qv = quat_left_mul(rotmat_to_quat(R), a_quats_np)
                    state["means"] = torch.tensor(m, dtype=torch.float32,
                                                  device=device)
                    state["quats"] = torch.tensor(qv, dtype=torch.float32,
                                                  device=device)
                out = model.get_camera_outputs(cam.to(device))
                rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                tiles.append(rgb)
            state["means"] = None

            # cameras can differ slightly in crop; pad to the max tile size
            h = max(t.shape[0] for t in tiles)
            w_ = max(t.shape[1] for t in tiles)
            grid = np.zeros((2 * h, 3 * w_, 3), dtype=np.uint8)
            for k, tile in enumerate(tiles):
                r, c = divmod(k, 3)
                grid[r * h:r * h + tile.shape[0],
                     c * w_:c * w_ + tile.shape[1]] = tile
            if args.downscale > 1:
                grid = grid[::args.downscale, ::args.downscale]
            if writer is None:
                writer = imageio.get_writer(out_mp4, fps=args.fps,
                                            codec="libx264", quality=8,
                                            pixelformat="yuv420p",
                                            macro_block_size=2)
            writer.append_data(grid)
            imageio.imwrite(os.path.join(args.out_dir, "frames",
                                         f"f{fi:03d}.png"), grid)
            if fi % 10 == 0:
                print(f"frame {fi}/{len(anchors)}")
    writer.close()
    splatad_mod.rasterization = orig_rast
    if tilt_log:
        arr = np.array([(p, r) for _, p, r in tilt_log])
        print(f"attitude: pitch mean/max {arr[:, 0].mean():.2f}/"
              f"{np.abs(arr[:, 0]).max():.2f} deg, roll mean/max "
              f"{arr[:, 1].mean():.2f}/{np.abs(arr[:, 1]).max():.2f} deg")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"config": args.config, "traj_npz": args.traj_npz,
                   "asset_ply": args.asset_ply, "human_npz": args.human_npz,
                   "step": int(step),
                   "n_frames": len(anchors), "start_frame": args.start_frame,
                   "target_length_m": args.target_len,
                   "trajectory_and_color_seed": 0,
                   "checkpoint_load_audit": load_audit,
                   "metric_scale": float(asset['metric_scale']) if human is None else None,
                   "gaussian_count": len(rgbs),
                   "method": "original SplatAD rasterization append; LiDAR road grounding; original decoder color calibration"}, f, indent=2)
    print("->", out_mp4)


if __name__ == "__main__":
    main()
