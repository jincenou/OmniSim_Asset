"""Prepare a distinct foreground-input experiment; preserve original R06."""
import json
import sys
from pathlib import Path
import av
from after_batch import ROUND, read, write, sha

OLD=ROUND/'experiments/moving_R06_v1'
ROOT=ROUND/'experiments/moving_R06_foreground_v2'

def main():
    if sys.argv[1]=='prepare':
        ROOT.mkdir(exist_ok=False)
        for folder in ('inputs','inputs_rife','segmentation/frames','research','previews','logs'):
            (ROOT/folder).mkdir(parents=True,exist_ok=True)
        mapping=read(OLD/'inputs/frame_mapping.json');native=read(OLD/'native.json')
        candidate=next(x for x in read(ROUND/'sample_manifest.json')['candidates'] if x['review_id']=='R06')
        mapping.update(scene=candidate['scene'],track_id=candidate['track_id'],camera_index=candidate['camera_index'],
            identity_key=candidate['identity_key'],source_split=candidate['split'],real_source_frames=49,output_frames=121,
            ego_translation_path_m=candidate['ego_path_m'],ego_rotation_max_deg=candidate['ego_rotation_deg'],
            real_source_frame_range=[0,48],source_input_sha256=mapping['video_sha256'])
        with av.open(str(OLD/'inputs/main.mp4')) as container:
            stream=container.streams.video[0];stream.thread_count=1
            width,height=stream.width,stream.height;count=0
            for i,frame in enumerate(container.decode(video=0)):
                frame.to_image().save(ROOT/'segmentation/frames'/f'{i:05d}.jpg',quality=95,subsampling=0);count+=1
        if count!=121:raise ValueError('Unexpected R06 input frame count')
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
        write(ROOT/'segmentation/prompt.json',{'points':[[x0+.5*w,y0+.2*h],[x0+.5*w,y0+.45*h],[x0+.4*w,y0+.8*h]],
            'labels':[1,1,1],'rationale':'Reviewed black sleeveless target with carried item and bag, native camera identity. Nearby pink-shirt person and bollards excluded; no invented hidden pixels.'})
    elif sys.argv[1]=='finalize':
        m=read(ROOT/'inputs/frame_mapping.json');m['video_sha256']=sha(ROOT/'inputs/main.mp4')
        m['scope']='New SAM foreground input; matching calibrated K(t)/rotations. Motion must be recomputed for these pixels.'
        write(ROOT/'inputs/frame_mapping.json',m)
    else:raise ValueError(sys.argv[1])

if __name__=='__main__':main()
