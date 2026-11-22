"""Read officially associated LiDAR boxes as independent diagnostic evidence."""
import argparse
import json
from pathlib import Path
import struct
import sys
import hashlib
import numpy as np

sys.path.insert(0, "/mnt/ojc/data/waymo_track_bbox_10/python_pkgs")
from waymo_open_dataset import dataset_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    raw = Path(candidate["raw_tfrecord"])
    cache = json.loads(Path(candidate["cache"]).read_text())
    stat = raw.stat()
    assert cache["raw_stat"] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    identity = candidate["associated_lidar_track_id"]
    observations = []
    with raw.open("rb") as stream:
        index = 0
        while True:
            header = stream.read(12)
            if not header:
                break
            if len(header) != 12:
                raise EOFError("Truncated TFRecord header")
            size = struct.unpack("<Q", header[:8])[0]
            payload = stream.read(size)
            if len(payload) != size or len(stream.read(4)) != 4:
                raise EOFError("Truncated TFRecord body")
            frame = dataset_pb2.Frame()
            frame.ParseFromString(payload)
            matches = [label for label in frame.laser_labels if label.id == identity]
            if len(matches) > 1:
                raise ValueError("Nonunique official identity")
            if matches:
                label = matches[0]
                if label.type != 2:
                    raise ValueError("Associated object is not PEDESTRIAN")
                box = label.box
                pose = np.asarray(frame.pose.transform).reshape(4, 4)
                center = np.array([box.center_x, box.center_y, box.center_z, 1.])
                # Box corners are for bbox comparison, not ground or foot contact.
                corners = np.asarray([[x, y, z] for x in (-box.length/2, box.length/2)
                                      for y in (-box.width/2, box.width/2)
                                      for z in (-box.height/2, box.height/2)])
                co, si = np.cos(box.heading), np.sin(box.heading)
                rot = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
                corners_vehicle = corners @ rot.T + center[:3]
                corners_world = corners_vehicle @ pose[:3, :3].T + pose[:3, 3]
                observations.append(dict(
                    frame=index, timestamp_s=frame.timestamp_micros/1e6,
                    center_world=(pose @ center)[:3].tolist(),
                    corners_world=corners_world.tolist(),
                    dimensions_length_width_height=[box.length, box.width, box.height],
                    num_lidar_points_in_box=label.num_lidar_points_in_box,
                    detection_difficulty=label.detection_difficulty_level,
                    tracking_difficulty=label.tracking_difficulty_level,
                    raw_record_sha256=hashlib.sha256(payload).hexdigest()))
            index += 1
    if not observations:
        raise ValueError("No associated 3D label; do not fabricate depth evidence")
    report = dict(
        identity=identity, source=str(raw), raw_stat=cache["raw_stat"],
        frames_read=index, observations=observations, crc_checked=False,
        scope="Official3D annotation diagnostic only; not used in motion fitting or hidden-action evaluation.",
        limitations=[
            "Frame/LiDAR annotation timestamp differs from camera exposure; interpolate only inside temporal support.",
            "Box center is not SMPL pelvis; bbox bottom is not a verified ground plane or foot contact.",
            "Heading is box heading, not true head/body facing; never force walking direction from it.",
            "3D labels may be uncertain or temporally annotated in occlusion; preserve point counts and difficulty.",
            "Using these boxes as future supervision would be a distinct3D-assisted experiment, not pure monocular reconstruction.",
        ])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(observations=len(observations),
                          first=observations[0]["frame"], last=observations[-1]["frame"])))


if __name__ == "__main__":
    main()
