"""Prepare a distinct foreground-input experiment; preserve original observations."""
import json
import sys
from pathlib import Path
import av
from after_batch import ROUND, read, write, sha

OLD=ROUND/'experiments/moving_M02_short61_native'
ROOT=ROUND/'experiments/moving_M02_short61_v1'

def main():
    if sys.argv[1]=='prepare':
        ROOT.mkdir(exist_ok=False)
        for folder in ('inputs','inputs_rife','segmentation/frames','research','previews','logs'):
            (ROOT/folder).mkdir(parents=True,exist_ok=True)
        mapping=read(OLD/'inputs/frame_mapping.json');native=read(OLD/'native.json')
        candidate=next(x for x in read(ROUND/'moving_candidates_v2/manifest.json')['candidates'] if x['review_id']=='M02')
        candidate['native_frame_count']=25
        candidate['duration_s']=(candidate['observations'][-1]['timestamp_micros']-candidate['observations'][0]['timestamp_micros'])/1e6
        candidate['parent_summary_fields_not_recomputed']={k:candidate.pop(k) for k in ('height_median_px','height_min_px','bbox_relative_motion','selection_score','review_image','preview_sources','native_preview','native_preview_validation','ego_path_m','ego_rotation_deg') if k in candidate}
        mapping.update(scene=candidate['scene'],track_id=candidate['track_id'],camera_index=candidate['camera_index'],
            identity_key=candidate['identity_key'],source_split=candidate['split'],real_source_frames=25,output_frames=61,
            ego_translation_path_m=None,ego_rotation_max_deg=None,
            real_source_frame_range=[0,24],source_input_sha256=mapping['video_sha256'])
        with av.open(str(OLD/'inputs/main.mp4')) as container:
            stream=container.streams.video[0];stream.thread_count=1
            width,height=stream.width,stream.height;count=0
            for i,frame in enumerate(container.decode(video=0)):
                frame.to_image().save(ROOT/'segmentation/frames'/f'{i:05d}.jpg',quality=95,subsampling=0);count+=1
        if count!=61:raise ValueError('Unexpected M02 input frame count')
        for row in mapping['frames']:
            row['crop_xyxy']=[0,0,width,height]
            row['crop_field_scope']='SAM input canvas only; actual undistortion/crop affine remains undistorted_to_input'
            row['source_path']=native['frames'][row['left_native_index']]['raw_source_path']
        write(ROOT/'inputs_rife/frame_mapping.json',mapping)
        (ROOT/'inputs_rife/main.mp4').symlink_to(OLD/'inputs/main.mp4')
        (ROOT/'inputs/cameras.npz').symlink_to(OLD/'inputs/cameras.npz')
        (ROOT/'native.json').symlink_to(OLD/'native.json')
        write(ROOT/'candidate.json',candidate)
        x0,y0,x1,y1=mapping['frames'][0]['bbox_input_xyxy'];w=x1-x0;h=y1-y0
        write(ROOT/'segmentation/prompt.json',{'points':[[190,215],[174,318],[160,435],[265,484],[345,350]],
            'labels':[1,1,1,1,0],
            'extra_objects':[{'id':2,'points':[[244,331]],'box':[222,288,269,362]}],
            'rationale':'All 25 native frames reviewed: gray cardigan, red scarf, black leggings and black shoulder bag, continuous walking without foreground occluder. Person and visible bag explicitly included; road excluded.'})
    elif sys.argv[1]=='finalize':
        m=read(ROOT/'inputs/frame_mapping.json');m['video_sha256']=sha(ROOT/'inputs/main.mp4')
        m['scope']='New SAM foreground input; matching calibrated K(t)/rotations. Motion must be recomputed for these pixels.'
        write(ROOT/'inputs/frame_mapping.json',m)
    else:raise ValueError(sys.argv[1])

if __name__=='__main__':main()
