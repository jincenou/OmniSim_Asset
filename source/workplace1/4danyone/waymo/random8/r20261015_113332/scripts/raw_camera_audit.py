"""Read raw Waymo camera metadata before building a calibrated motion branch.

No GPU or source mutation. Keeps frame, camera-pose and exposure-center times
distinct. The exposure-center approximation does not correct rolling shutter.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
from waymo_open_dataset import dataset_pb2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-id', default='R01')
    parser.add_argument('--manifest', type=Path, default=R/'sample_manifest.json')
    args = parser.parse_args()
    row = next(x for x in json.loads(args.manifest.read_text())['candidates']
               if x['review_id'] == args.review_id)
    from reserve_admission import verify
    verify(row,R)
    output = R/'camera_audit'/args.review_id
    output.mkdir(parents=True, exist_ok=False)
    scene = Path('/mnt/public_data/waymo')/row['split']/row['scene']
    observations = {x['frame_index']: x for x in row['observations']}
    cam = row['camera_index']
    stored_K = np.loadtxt(scene/'intrinsics'/f'{cam}.txt').reshape(-1)
    stored_ext = np.loadtxt(scene/'cam_to_ego'/f'{cam}.txt')
    # OpenCV right/down/forward -> Waymo forward/left/up.
    cv_to_waymo = np.eye(4)
    cv_to_waymo[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    records, frame_poses, camera_poses, center_poses, rotations = [], [], [], [], []
    contacts = []
    raw_path = Path(row['raw_tfrecord'])
    raw_stat = raw_path.stat()
    cache = json.loads(Path(row['cache']).read_text())
    if cache['raw_stat'] != {'size': raw_stat.st_size, 'mtime_ns': raw_stat.st_mtime_ns}:
        raise ValueError('Raw source changed since track audit')
    with raw_path.open('rb') as stream:
        index = 0
        while index <= max(observations):
            header = stream.read(12)
            if len(header) != 12:
                raise EOFError('Incomplete TFRecord header')
            size = struct.unpack('<Q', header[:8])[0]
            payload = stream.read(size)
            if len(payload) != size or len(stream.read(4)) != 4:
                raise EOFError('Incomplete TFRecord payload')
            if index not in observations:
                index += 1
                continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(payload)
            if frame.context.name != row['segment'].removeprefix('segment-').removesuffix('_with_camera_labels'):
                raise ValueError('Wrong raw segment')
            observation = observations[index]
            if frame.timestamp_micros != observation['timestamp_micros']:
                raise ValueError('Source time or frame mapping mismatch')
            calibration = next(x for x in frame.context.camera_calibrations if x.name == cam+1)
            image = next(x for x in frame.images if x.name == cam+1)
            intrinsic = np.asarray(calibration.intrinsic)
            ext = np.asarray(calibration.extrinsic.transform).reshape(4, 4)
            ego = np.asarray(frame.pose.transform).reshape(4, 4)
            cam_ego = np.asarray(image.pose.transform).reshape(4, 4)
            errors = {
                'intrinsic': float(np.max(np.abs(intrinsic-stored_K))),
                'raw_camera_to_ego': float(np.max(np.abs(ext-stored_ext))),
                'frame_ego_pose': float(np.max(np.abs(ego-np.loadtxt(scene/'ego_to_world'/f'{index:03d}.txt')))),
                'raw_axes_camera_to_world': float(np.max(np.abs(ego@ext-np.loadtxt(scene/'cam_to_world'/f'{index:03d}_{cam}.txt')))),
            }
            if max(errors.values()) > 1e-5:
                raise ValueError(f'Processed/raw calibration mismatch: {errors}')
            K = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]])
            source = scene/'images'/f'{index:03d}_{cam}.jpg'
            processed_bytes = source.read_bytes()
            raw_pixels = np.asarray(Image.open(io.BytesIO(image.image)).convert('RGB'))
            processed_pixels = np.asarray(Image.open(io.BytesIO(processed_bytes)).convert('RGB'))
            if raw_pixels.shape != (calibration.height, calibration.width, 3):
                raise ValueError('Raw dimensions mismatch')
            pixel_error = int(np.abs(raw_pixels.astype(np.int16)-processed_pixels.astype(np.int16)).max())
            if pixel_error != 0:
                raise ValueError('Processed image differs from raw decoded JPEG; undistortion status unresolved')
            pose_time = image.pose_timestamp
            center_time = 0.5*(image.camera_trigger_time+image.camera_readout_done_time)
            if not image.HasField('pose_timestamp') or not image.HasField('velocity'):
                raise ValueError('Exposure metadata unavailable')
            dt = center_time-pose_time
            velocity = image.velocity
            center_ego = cam_ego.copy()
            center_ego[:3, :3] = Rotation.from_rotvec(np.array([velocity.w_x, velocity.w_y, velocity.w_z])*dt).as_matrix()@cam_ego[:3, :3]
            center_ego[:3, 3] += np.array([velocity.v_x, velocity.v_y, velocity.v_z])*dt
            center_cv = center_ego@ext@cv_to_waymo
            rot = center_cv[:3, :3].T
            if np.max(np.abs(rot@rot.T-np.eye(3))) > 1e-6 or abs(np.linalg.det(rot)-1) > 1e-6:
                raise ValueError('Non-SO3 camera rotation')
            bbox = np.asarray(observation['bbox_xyxy'])
            x, y = np.meshgrid(np.linspace(bbox[0], bbox[2], 9), np.linspace(bbox[1], bbox[3], 9))
            pixels = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
            undistorted = cv2.undistortPointsIter(pixels, K, intrinsic[4:], None, K,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10))
            displacement = np.linalg.norm(undistorted-pixels, axis=-1)
            readout = image.camera_readout_done_time-image.camera_trigger_time-image.shutter
            if readout < 0 or image.shutter <= 0:
                raise ValueError('Invalid exposure metadata')
            records.append({'frame_index': index, 'frame_timestamp_micros': frame.timestamp_micros,
                'camera_pose_timestamp_s': pose_time, 'image_center_timestamp_s': center_time,
                'camera_pose_minus_frame_timestamp_s': pose_time-frame.timestamp_micros/1e6,
                'center_minus_camera_pose_s': dt, 'shutter_s': image.shutter, 'readout_s': readout,
                'rolling_direction': int(calibration.rolling_shutter_direction),
                'source_path': str(source), 'raw_jpeg_sha256': sha(image.image),
                'processed_jpeg_sha256': sha(processed_bytes), 'raw_record_sha256': sha(payload),
                'decoded_pixel_max_abs_difference': pixel_error,
                'calibration_errors': errors, 'bbox_xyxy_distorted': bbox.tolist(),
                'bbox_grid_undistortion_displacement_px_max': float(displacement.max()),
                'frame_pose_vs_camera_pose_translation_m': float(np.linalg.norm(ego[:3,3]-cam_ego[:3,3]))})
            frame_poses.append(ego);camera_poses.append(cam_ego);center_poses.append(center_cv);rotations.append(rot)
            if index in (min(observations), sorted(observations)[len(observations)//2], max(observations)):
                corrected = cv2.undistort(raw_pixels, K, intrinsic[4:], None, K)
                crop = tuple(np.rint(bbox+[-30,-30,30,30]).astype(int))
                pair = Image.new('RGB', (520, 540), '#222222');draw=ImageDraw.Draw(pair)
                for col, (im, title) in enumerate(((raw_pixels, 'Original distorted'), (corrected, 'Undistorted, same K'))):
                    panel=ImageOps.contain(Image.fromarray(im).crop(crop), (256,510))
                    pair.paste(panel,(col*260+(260-panel.width)//2,25))
                    draw.text((col*260+3,5), f'f{index} {title}', fill='white')
                contacts.append(pair)
            index += 1
    if len(records) != row['native_frame_count']:
        raise ValueError('Incomplete observation span')
    rotations = np.asarray(rotations)
    relative = rotations[1:]@rotations[:-1].transpose(0,2,1)
    # Matches GVHMR's PyTorch3D matrix_to_rotation_6d: first two rows.
    cam_angvel = relative[:, :2, :].reshape(-1,6)
    cam_angvel = np.concatenate([cam_angvel, cam_angvel[-1:]], axis=0)
    time = np.array([x['image_center_timestamp_s'] for x in records])
    if not np.all(np.diff(time)>0):
        raise ValueError('Nonmonotonic exposure times')
    np.savez(output/'native_cameras.npz', frame_ego_to_world=frame_poses,
        camera_timestamp_ego_to_world=camera_poses, image_center_cv_camera_to_world=center_poses,
        R_world_to_cv_camera=rotations, cam_angvel_rotation6d=cam_angvel,
        frame_indices=sorted(observations), image_center_timestamps_s=time,
        K=K, distortion=intrinsic[4:], cv_to_waymo=cv_to_waymo)
    sheet = Image.new('RGB',(520*len(contacts),540))
    for i, contact in enumerate(contacts):sheet.paste(contact,(520*i,0))
    sheet.save(output/'distortion_contact.jpg',quality=95)
    report={'state':'raw_camera_audit_passed_not_motion_inference', 'review_id':args.review_id,
        'scene':row['scene'],'camera_index':cam,'track_id':row['track_id'],'frames':len(records),
        'raw_source':str(raw_path),'raw_stat':cache['raw_stat'],'tf_record_crc_checked':False,
        'processed_images_equal_decoded_raw':True,'processed_images_undistorted':False,
        'stored_extrinsic_axes':'Waymo forward/left/up; apply cv_to_waymo on right before OpenCV use',
        'center_pose_model':'constant global linear/angular velocity extrapolation from CameraImage.pose_timestamp',
        'gvhmr_cam_angvel_definition':'relative world-to-camera rotation first two rows, last step repeated; not rad/s',
        'time_interval_s_range':[float(np.diff(time).min()),float(np.diff(time).max())],
        'center_pose_npz_sha256':sha((output/'native_cameras.npz').read_bytes()),
        'limitations':['One representative pose per image does not correct rolling shutter',
            'No GVHMR inference or SMPL projection acceptance in this audit',
            'Native10Hz rotation6d cannot be reused for25Hz without time resampling',
            'Calibrated intrinsics require matching undistorted images and crop affine',
            'GVHMR consumes rotation increments but not measured camera translation; output is not registered Waymo world motion'],
        'sources':['https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto',
            'https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/camera_model.cc'],
        'records':records}
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':report['state'],'frames':len(records),'pixel_equality':True,
        'max_undistortion_shift_px':max(x['bbox_grid_undistortion_displacement_px_max'] for x in records),
        'max_frame_camera_pose_delta_m':max(x['frame_pose_vs_camera_pose_translation_m'] for x in records)}),flush=True)


if __name__ == '__main__':main()
