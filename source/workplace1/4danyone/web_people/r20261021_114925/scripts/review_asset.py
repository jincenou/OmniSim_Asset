"""One final-asset contact sheet, only after independent replay has passed."""
import argparse
import json
import time
import av
from PIL import Image, ImageDraw
from prepare_person import ROOT, sha, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    args = parser.parse_args()
    assert args.id in {f'W{i:02d}' for i in range(1, 9)}
    folder = ROOT / 'previews' / args.id
    replay = json.loads((folder / 'independent_replay.json').read_text())
    assert replay['state'] == 'asset_only_replay_passed_quality_unreviewed'
    assert replay['source_reads'] == 0
    source = folder / 'asset_fixed_views.mp4'
    assert sha(source) == replay['videos'][source.name]['sha256']
    output = folder / 'asset_contact.jpg'
    assert not output.exists()
    chosen = (0, 60, 120)
    canvas = Image.new('RGB', (2112, 1360), (128, 128, 128))
    total = 0
    with av.open(str(source)) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            assert frame.width == 704 and frame.height == 1328
            assert abs(float(frame.pts * frame.time_base) - index / 25) < 1e-7
            if index in chosen:
                x = chosen.index(index) * 704
                canvas.paste(frame.to_image(), (x, 32))
                ImageDraw.Draw(canvas).text((x + 8, 10), f'{args.id} FINAL ASSET / frame {index} / {index / 25:.2f}s', fill='white')
            total += 1
    assert total == 121
    canvas.save(output, quality=94)
    save(folder / 'asset_review_media.json', dict(at=time.time(), source=str(source),
        source_sha256=sha(source), frames_decoded=total, sampled_frames=list(chosen),
        views=[0, 6, 12, 18], image=str(output), image_sha256=sha(output),
        full_image_preserved=True, exposure_or_color_adjustment=False,
        visual_review_performed=False, scope='A contact sheet is review material, not quality approval.'))
    print(str(output))


if __name__ == '__main__':
    main()
