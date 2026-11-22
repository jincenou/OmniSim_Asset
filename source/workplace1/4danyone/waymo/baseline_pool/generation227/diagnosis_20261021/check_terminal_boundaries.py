"""Read-only source/crop diagnosis for visually identified terminal truncations."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

D = Path(__file__).resolve().parent
SELECTED = [38, 41, 76, 84, 97, 131, 132, 141, 144, 154, 198, 201, 204]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(2)
    result_path = D / 'terminal_boundaries.json'
    assert not result_path.exists(), 'Preserve previous diagnosis'
    rows = json.loads((D / 'classification.json').read_text())['rows']
    results, panels = [], []
    for index in SELECTED:
        row = next(r for r in rows if r['index'] == index)
        native_path = Path(row['package']) / 'clip0/native.json'
        native = json.loads(native_path.read_text())
        frame = native['frames'][-1]
        audit_path = Path(native['source_camera_audit'])
        assert sha(audit_path) == native['source_camera_audit_sha256']
        audit = json.loads(audit_path.read_text())
        record = audit['records'][-1]
        assert record['frame_index'] == frame['frame_index']
        source_path = Path(frame['raw_source_path'])
        assert sha(source_path) == frame['raw_source_sha256']
        source = Image.open(source_path).convert('RGB')
        W, H = source.size
        box = np.array(record['bbox_xyxy_distorted'])
        edge_dist = np.array([box[0], box[1], W - box[2], H - box[3]])
        npz_path = audit_path.parent / 'native_cameras.npz'
        with np.load(npz_path) as cameras:
            K, dist = cameras['K'], cameras['distortion']
        affine = np.array(frame['undistorted_to_input'])
        width, height = native['size']
        maps = cv2.initUndistortRectifyMap(K, dist, np.eye(3), affine @ K,
                                         (width, height), cv2.CV_32FC1)
        crop = cv2.remap(np.array(source), *maps, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        output = io.BytesIO()
        Image.fromarray(crop).save(output, format='PNG')
        reconstructed_hash = hashlib.sha256(output.getvalue()).hexdigest()
        exact_replay = reconstructed_hash == frame['sha256']
        assert exact_replay, f'Crop replay differs: {row["id"]}'
        b = np.array(frame['bbox_input_xyxy'])
        margins = [b[0], b[1], width-b[2], height-b[3]]
        assert min(margins) >= 0
        # Fraction is source-camera coverage of the bbox rectangle, not body visibility.
        x0, y0 = np.floor(b[:2]).astype(int)
        x1, y1 = np.ceil(b[2:]).astype(int)
        valid = ((maps[0] >= 0) & (maps[0] <= W-1) &
                 (maps[1] >= 0) & (maps[1] <= H-1))
        rect_valid = valid[max(0,y0):min(height,y1), max(0,x0):min(width,x1)]
        assert rect_valid.size
        observed_path = native_path.parent / 'observed_native.mp4'
        with av.open(str(observed_path)) as container:
            container.streams.video[0].thread_count = 1
            count = 0
            for video_frame in container.decode(video=0):
                observed = video_frame.to_image()
                count += 1
        assert count == len(native['frames'])
        marked = source.copy()
        ImageDraw.Draw(marked).rectangle(tuple(box), outline='lime', width=4)
        # Zoom includes real camera boundary; black outside the source is labelled padding.
        center = (box[:2]+box[2:])/2
        side = max(128, float(max(box[2:]-box[:2]))*1.7)
        zoom_bounds = tuple(np.rint(np.r_[center-side/2, center+side/2]).astype(int))
        zoom = marked.crop(zoom_bounds)
        panel = Image.new('RGB', (960, 300), '#202020')
        draw = ImageDraw.Draw(panel)
        draw.text((5,3), f'{index:03d} {row["id"]} original f{frame["frame_index"]}', fill='white')
        for x, im, w, label in [(0, marked, 420, 'original camera + bbox'),
                                 (420, zoom, 270, 'source zoom; black = outside image'),
                                 (690, observed, 135, 'native video last'),
                                 (825, Image.fromarray(crop), 135, 'exact crop replay')]:
            panel.paste(ImageOps.pad(im, (w,255), color='#202020'), (x,25))
            draw.text((x+3,282),label,fill='white')
        panels.append(panel)
        results.append({'index':index,'id':row['id'],'source_path':str(source_path),
            'source_sha256':sha(source_path),'native_metadata_sha256':sha(native_path),
            'camera_npz_sha256':sha(npz_path),'source_frame_index':frame['frame_index'],
            'source_bbox_xyxy':box.tolist(),'source_edge_distance_ltrb_px':edge_dist.tolist(),
            'source_boundary_touch':bool(edge_dist.min()<=.5),
            'crop_bbox_margins_ltrb_px':margins,'bbox_inside_crop':True,
            'crop_replay_matches_saved_png_sha256':exact_replay,
            'reconstructed_crop_sha256':reconstructed_hash,
            'bbox_rectangle_source_coverage_fraction':float(rect_valid.mean()),
            'visual_reviewed':False})
    sheets=[]
    for start in range(0,len(panels),4):
        group=panels[start:start+4]
        sheet=Image.new('RGB',(960,300*len(group)),'#202020')
        for offset,panel in enumerate(group):sheet.paste(panel,(0,300*offset))
        path=D/f'terminal_boundaries_{start//4:02d}.jpg'
        assert not path.exists()
        sheet.save(path,quality=93)
        sheets.append({'path':str(path),'sha256':sha(path),'indices':SELECTED[start:start+4]})
    result={'utc':datetime.now(timezone.utc).isoformat(),'rows':results,'sheets':sheets,
        'scope':'13 visually selected terminal truncation cases, not a random pool sample.',
        'limits':['A bbox touching the original image boundary does not alone prove truncation.',
                  'Crop replay verifies implementation and saved output, not annotation correctness.',
                  'Source coverage fraction is not body occlusion or visible-body fraction.',
                  'No new model inference, input modification, filtering threshold or generation.']}
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':len(results),'original_boundary_touch':sum(r['source_boundary_touch'] for r in results),'exact_crop_replays':sum(r['crop_replay_matches_saved_png_sha256'] for r in results)}))


if __name__ == '__main__':
    main()
