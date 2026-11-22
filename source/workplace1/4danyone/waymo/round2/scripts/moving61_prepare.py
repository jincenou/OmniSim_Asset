"""Exact R01 subwindow, with original pixels, camera calibration and provenance."""
import copy
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image, ImageDraw
from moving_prepare import R, save, sha

sys.path.insert(0, str(R.parent/'scripts'))
from prepare import encode

OLD=R/'experiments/moving_R01_foreground_v2'
ROOT=R/'experiments/moving_R01_short61_v1'

def decode(path):
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count=1
        return [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]

def main():
    watched=[OLD/'inputs/main.mp4',OLD/'inputs/frame_mapping.json',OLD/'inputs/cameras.npz',OLD/'segmentation/target_masks.npz']
    before={str(p):sha(p) for p in watched}
    parent=json.loads((OLD/'inputs/frame_mapping.json').read_text())
    assert parent['video_sha256']==before[str(watched[0])]
    assert parent['camera_npz_sha256']==before[str(watched[2])]
    all_rgb=decode(watched[0]);assert len(all_rgb)==121
    rgb=all_rgb[60:121]
    ROOT.mkdir(exist_ok=False)
    for folder in ('inputs','segmentation','research','previews'):(ROOT/folder).mkdir()
    encode(ROOT/'inputs/main.mp4',[Image.fromarray(x) for x in rgb],25)
    decoded=decode(ROOT/'inputs/main.mp4')
    assert len(decoded)==61 and all(np.array_equal(a,b) for a,b in zip(rgb,decoded,strict=True))
    camera=np.load(watched[2]);arrays={k:camera[k][60:121].copy() for k in camera.files}
    for key in ('source_relative_exposure_seconds','video_seconds'):arrays[key]-=arrays[key][0]
    assert all(len(v)==61 for v in arrays.values())
    np.savez_compressed(ROOT/'inputs/cameras.npz',**arrays)
    masks=np.load(watched[3])['masks'][60:121].copy()
    np.savez_compressed(ROOT/'segmentation/target_masks.npz',masks=masks)
    mapping=copy.deepcopy(parent);first=parent['frames'][60]['source_relative_exposure_seconds']
    summary_keys=('native_source','native_source_sha256','native_frames',
        'direct_native_output_slots','interpolated_slots','sample_span_s',
        'real_exposure_span_s','max_time_regularization_s','device','peak_allocated_gib')
    mapping['parent_preprocessing_summary']={k:copy.deepcopy(parent[k]) for k in summary_keys if k in parent}
    mapping.pop('device',None);mapping.pop('peak_allocated_gib',None)
    rows=copy.deepcopy(parent['frames'][60:121])
    for slot,row in enumerate(rows):
        row.update(parent_slot=row['slot'],slot=slot,video_seconds=slot/25,
            parent_source_relative_exposure_seconds=row['source_relative_exposure_seconds'])
        row['source_relative_exposure_seconds']-=first
        row['left_native_index']-=24;row['right_native_index']-=24
    assert sum(not x['interpolated'] for x in rows)==13
    native=copy.deepcopy(json.loads((OLD/'native.json').read_text()));native['frames']=native['frames'][24:49]
    for i,row in enumerate(native['frames']):
        row.update(parent_native_index=row['native_index'],native_index=i)
        assert sha(Path(row['path']))==row['sha256']
    native_camera=np.load(R/'experiments/moving_R01_v1/native_cameras.npz')
    nc={k:native_camera[k][24:49].copy() for k in native_camera.files}
    np.savez_compressed(ROOT/'native_cameras.npz',**nc)
    native['camera_npz_sha256']=sha(ROOT/'native_cameras.npz')
    save(ROOT/'native.json',native)
    candidate=copy.deepcopy(json.loads((OLD/'candidate.json').read_text()))
    candidate['observations']=candidate['observations'][24:49]
    # Camera translation at exposure centers is independently calibrated; label precisely.
    translations=nc['image_center_cv_camera_to_world'][:,:3,3]
    camera_path=float(np.linalg.norm(np.diff(translations,axis=0),axis=1).sum())
    mapping.update(frames=rows,real_source_frames=25,output_frames=61,real_source_frame_range=[48,72],
        native_source=str(ROOT/'native.json'),native_source_sha256=sha(ROOT/'native.json'),
        native_frames=25,direct_native_output_slots=13,interpolated_slots=48,sample_span_s=2.4,
        real_exposure_span_s=rows[-1]['source_relative_exposure_seconds'],
        max_time_regularization_s=max(abs(x['source_relative_exposure_seconds']-x['video_seconds']) for x in rows),
        video_sha256=sha(ROOT/'inputs/main.mp4'),camera_npz_sha256=sha(ROOT/'inputs/cameras.npz'),
        parent_input_sha256=parent['video_sha256'],source_input_sha256=parent['video_sha256'],
        sample_span_seconds=2.4,encoded_duration_seconds=61/25,
        ego_translation_path_m=None,ego_rotation_max_deg=None,
        calibrated_camera_translation_path_m=camera_path,
        scope='Exact parent slots 60..120, reindexed and lossless; no padding or action retiming',
        limitations=['SAM propagation and fixed crop were computed using parent 49-image window; not a short-observation-only benchmark',
            'RIFE intermediate images are estimates; only 13 of 61 slots directly use native image times',
            'Exposure-center camera poses approximate rolling shutter; camera translation not consumed by GVHMR'])
    candidate.update(parent_candidate=str(OLD/'candidate.json'),observed_frame_range=[48,72],
        ego_path_m=None,ego_rotation_deg=None,calibrated_camera_translation_path_m=camera_path,
        native_frame_count=25,duration_s=(candidate['observations'][-1]['timestamp_micros']-candidate['observations'][0]['timestamp_micros'])/1e6,
        camera_calibration_status='Raw camera audit passed; exposure-center approximation')
    candidate['parent_summary_fields_not_recomputed']={k:candidate.pop(k) for k in
        ('height_median_px','height_min_px','bbox_relative_motion','selection_score','review_image',
         'preview_sources','native_preview','native_preview_validation') if k in candidate}
    save(ROOT/'candidate.json',candidate);save(ROOT/'inputs/frame_mapping.json',mapping)
    # All 25 native observations, paired with nearest generated-input time.
    panels=[]
    for index,row in enumerate(native['frames']):
        slot=round(index*2.5)
        panel=Image.new('RGB',(352*2,640),'white')
        panel.paste(Image.open(row['path']).convert('RGB'),(0,0));panel.paste(Image.fromarray(rgb[slot]),(352,0))
        d=ImageDraw.Draw(panel);d.rectangle((0,0,704,22),fill='black')
        d.text((4,4),f'native {row["frame_index"]} | input slot {slot}',fill='white')
        panels.append(panel.resize((352,320)))
    for start in (0,10,20):
        subset=panels[start:start+10];sheet=Image.new('RGB',(352*5,320*((len(subset)+4)//5)),'white')
        for i,p in enumerate(subset):sheet.paste(p,((i%5)*352,(i//5)*320))
        sheet.save(ROOT/'previews'/f'native_foreground_{start:02d}.jpg',quality=95)
    assert before=={str(p):sha(p) for p in watched}
    save(ROOT/'research/subwindow_validation.json',{'state':'prepared_visual_review_pending',
        'frames':61,'native_observations':25,'direct_native_slots':13,'interpolated_slots':48,
        'lossless_parent_pixel_equality':True,'parent_files_unchanged':before,
        'source_frame_range':[48,72],'parent_slot_range':[60,120],
        'calibrated_camera_path_m':camera_path,'source_exposure_span_s':float(arrays['source_relative_exposure_seconds'][-1]),
        'sample_span_seconds':2.4,'limitations':mapping['limitations']})
    print(json.dumps({'state':'prepared','camera_path_m':camera_path,'frames':61}),flush=True)

if __name__=='__main__':main()
