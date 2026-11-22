"""Display the fixed moving-camera cases without estimating pose or generating RGB."""
import argparse
import json
import time

import numpy as np
from PIL import Image, ImageDraw
from prepare_person import ROOT, encode, save, selected, sha
from source_io import read_source, source_identity
from pathlib import Path


def main(ident):
    choice = selected(ident)
    assert choice['camera_class'] in ('moving', 'unknown')
    source = Path(choice['source_path'])
    assert source_identity(source) == choice['source_identity']
    track_path = ROOT / 'sources' / f'{choice["source_id"]}_tracks.json'
    tracks = json.loads(track_path.read_text())
    observations = {r['frame']: r for r in tracks['tracks'][str(choice['track_id'])]}
    timestamps = np.asarray(tracks['timestamps'], dtype=float)
    clock = choice['start_time'] + np.arange(121)/25
    picked = np.abs(timestamps[None, :] - clock[:, None]).argmin(1)
    assert len(set(picked)) == 121 and np.all(np.diff(picked) > 0)
    assert picked[0] >= choice['start_frame'] and picked[-1] <= choice['end_frame']
    assert all(int(i) in observations for i in picked)
    assert np.abs(timestamps[picked]-clock).max() <= np.diff(timestamps).max()/2+1e-6
    wanted = {int(value): slot for slot, value in enumerate(picked)}
    destination = ROOT / 'inputs' / ident
    destination.mkdir(parents=True, exist_ok=False)
    records = []

    def frames():
        for index, timestamp, rgb in read_source(source):
            if index in wanted:
                box = observations[index]['bbox']
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                draw.rectangle(tuple(box), outline='lime', width=4)
                draw.text((12, 12), f'{ident} REAL SOURCE; moving camera; t={timestamp:.3f}',
                          fill='white', stroke_width=2, stroke_fill='black')
                height = round(image.height * 960 / image.width / 2) * 2
                records.append(dict(display_frame=wanted[index], source_frame=index,
                                    source_timestamp_s=timestamp, target_time_s=float(clock[wanted[index]]),
                                    original_bbox_xyxy=box))
                yield image.resize((960, height))
            if index >= int(picked[-1]):
                break

    target = destination / 'source_clip.mp4'
    encode(target, frames())
    assert len(records) == 121
    save(destination / 'candidate.json', choice)
    save(destination / 'deferred_source.json', dict(
        at=time.time(), state='source_preview_only_camera_interface_not_applicable',
        source_identity=choice['source_identity'], selected_sha256=sha(ROOT / 'SELECTED_TRACKS.json'),
        track_file=str(track_path), track_sha256=sha(track_path),
        source_clip_sha256=sha(target), frames=records, display_fps=25,
        decoded_frames=121, camera_class=choice['camera_class'],
        pose_estimated=False, generation_started=False, asset_created=False,
        limitations='Annotated/downscaled source preview; nearest original frames without speed change. '
                    'No static camera assumption or model input package was applied.'))
    print(ident, 'source preview complete; camera interface remains deferred', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', choices=['W07', 'W08'])
    main(parser.parse_args().id)
