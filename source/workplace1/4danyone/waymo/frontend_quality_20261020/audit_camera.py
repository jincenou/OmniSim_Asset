"""Audit measured ego/label motion without treating labels as body pose GT."""
from pathlib import Path
import json, struct, sys, argparse
import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--sample', default='038/P08')
args = parser.parse_args()
P = OUT.parent/'scene20/r20261019/scenes'/args.sample
sys.path.insert(0, '/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
from waymo_open_dataset import dataset_pb2

cand = json.loads((P/'manifest.json').read_text())['candidate']
native = np.load(P/'input/native_cameras.npz')
cameras = native['image_center_cv_camera_to_world']
times = native['image_center_timestamps_s']
observations = {x['frame_index']:x for x in cand['observations']}
rows = []
with Path(cand['raw_tfrecord']).open('rb') as file:
    index = 0
    while index <= max(observations):
        header = file.read(12)
        assert len(header)==12
        size = struct.unpack('<Q',header[:8])[0]
        if index not in observations:
            file.seek(size+4,1);index+=1;continue
        data=file.read(size);assert len(data)==size and len(file.read(4))==4
        frame=dataset_pb2.Frame();frame.ParseFromString(data)
        assert frame.timestamp_micros == observations[index]['timestamp_micros']
        assert frame.context.name == cand['segment'].removeprefix('segment-').removesuffix('_with_camera_labels')
        matches=[label for label in frame.laser_labels if label.id==cand['associated_lidar_track_id']]
        if matches:
            assert len(matches)==1
            label=matches[0];box=label.box
            ego=np.asarray(frame.pose.transform).reshape(4,4)
            xyz=np.array([box.center_x,box.center_y,box.center_z])
            world=ego[:3,:3]@xyz+ego[:3,3]
            heading=ego[:3,:3]@np.array([np.cos(box.heading),np.sin(box.heading),0])
            rows.append(dict(frame=index,timestamp_micros=frame.timestamp_micros,
                center_ego=xyz.tolist(), center_world=world.tolist(),heading_world=heading.tolist(),
                label_type=int(label.type),difficulty=int(label.detection_difficulty_level)))
        index+=1
has_labels=len(rows)>=2
world=np.asarray([row['center_world'] for row in rows]);relative=np.asarray([row['center_ego'] for row in rows])
delta=world[-1]-world[0] if has_labels else None
result=dict(source=str(P), matched_labels=len(rows), requested_labels=len(observations),
    camera_world_displacement_m=float(np.linalg.norm(cameras[-1,:3,3]-cameras[0,:3,3])),
    camera_world_path_m=float(np.linalg.norm(np.diff(cameras[:,:3,3],axis=0),axis=1).sum()),
    camera_rotation_deg=float(Rotation.from_matrix(cameras[-1,:3,:3]@cameras[0,:3,:3].T).magnitude()*180/np.pi),
    camera_duration_s=float(times[-1]-times[0]),
    label_center_world_displacement_m=float(np.linalg.norm(delta)) if has_labels else None,label_center_world_delta_xyz_m=delta.tolist() if has_labels else None,
    label_center_world_span_xyz_m=np.ptp(world,axis=0).tolist() if has_labels else None,
    label_center_world_path_m=float(np.linalg.norm(np.diff(world,axis=0),axis=1).sum()) if has_labels else None,
    label_center_ego_delta_xyz_m=(relative[-1]-relative[0]).tolist() if has_labels else None,
    label_evidence_status="available_not_body_truth" if has_labels else "unknown_no_verified_associated_3d_track",
    global_motion_scope='Laser-box centers constrain approximate translation, not pelvis/joint pose or reliable front/back human heading.',
    time_scope='Laser labels use frame timestamps; cameras use exposure-center timestamps. Values are separately summarized, not falsely synchronized joint observations.',
    calibration=json.loads((P/'motion/calibration.json').read_text()),rows=rows)
(OUT/f'{P.name}_camera_motion.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','calibration')},indent=2))
