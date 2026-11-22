"""One fixed input/generated contact sheet per completed person, no quality label."""
import argparse
import json
import time

import av
from PIL import Image
from prepare_person import ROOT, save, sha
from publish_previews import tile

TIMES = (0, 30, 60, 90, 120)
VIEWS = (0, 6, 12, 18)


def sample(path):
    selected = {}
    count = 0
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 2
        for index, frame in enumerate(container.decode(video=0)):
            assert abs(float(frame.pts * frame.time_base) - index / 25) < 1e-7
            if index in TIMES:
                selected[index] = frame.to_image()
            count += 1
    assert count == 121 and len(selected) == len(TIMES), (path, count)
    return selected


def main(ident):
    base = ROOT / 'frontend' / ident
    validation = base / 'research/generated_validation.json'
    assert json.loads(validation.read_text())['state'] == 'all24_videos_decoded_same_retained_cameras'
    destination = ROOT / 'previews' / ident
    destination.mkdir(exist_ok=True)
    output = destination / 'input_generated_contact.jpg'
    assert not output.exists(), 'Review sheet already exists; do not overwrite evidence'
    inputs = ROOT / 'inputs' / ident
    sources = [('SOURCE CROP', inputs / 'crop_unsegmented.mp4'),
               ('MODEL INPUT', inputs / 'source_crop.mp4')]
    sources += [(f'GENERATED yaw {v*15}', base / 'outputs/turbo/videos' / f'{v:02d}.mp4') for v in VIEWS]
    canvas = Image.new('RGB', (1080, 1720), (128, 128, 128))
    evidence = []
    for column, (label, path) in enumerate(sources):
        images = sample(path)
        for row, frame in enumerate(TIMES):
            canvas.paste(tile(images[frame], (180, 344), f'{label}; f{frame}'), (column*180, row*344))
        evidence.append(dict(path=str(path), sha256=sha(path), decoded_frames=121))
    canvas.save(output, quality=94)
    save(destination / 'generated_review_media.json', dict(
        at=time.time(), views=list(VIEWS), frames=list(TIMES), evidence=evidence,
        sheet=str(output), sheet_sha256=sha(output),
        generation_validation_sha256=sha(validation), visual_review_performed=False,
        scope='Fixed source/input/four generated cameras; no postprocessing; other times/views unreviewed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', choices=['W03', 'W04', 'W05', 'W06'])
    main(parser.parse_args().id)
