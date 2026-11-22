"""Verify the short experiment's actual frames, time span and native slots."""
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT'])


def decode(path):
    with av.open(str(path)) as con:
        stream=con.streams.video[0];stream.thread_count=1
        assert stream.average_rate==25
        frames=[]
        for index,frame in enumerate(con.decode(video=0)):
            assert frame.pts*stream.time_base==Fraction(index,25)
            frames.append(frame.to_ndarray(format='rgb24'))
        assert len(frames)==21,(path,len(frames))
    return frames


def main(mode):
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    native=json.loads((ROOT/'inputs_native/frame_mapping.json').read_text())
    assert len(native['frames'])==9 and len(mapping['frames'])==21
    assert abs(native['frames'][-1]['relative_seconds']-.8)<.01
    assert mapping['real_source_frames']==9 and mapping['output_frames']==21
    record={'frames':21,'real_source_frames':9,'sample_span_seconds':.8,'fps':25,'encoded_duration_seconds':.84,
            'scope':'Experimental short-sequence inference; interpolation is not additional observation'}
    if mode=='input':
        raw=decode(ROOT/'inputs_rife/main.mp4');decode(ROOT/'inputs/main.mp4')
        checked=[]
        for slot in (0,5,10,15,20):
            source=native['frames'][slot*2//5]
            expected=np.array(Image.open(source['source_path']).convert('RGB').crop(tuple(source['crop_xyxy'])))
            assert np.array_equal(raw[slot],expected),f'Native slot changed: {slot}'
            checked.append(slot)
        record.update(status='input_validated',native_slots_exact=checked,estimated_slots=16)
        path=ROOT/'research/input_validation.json'
    else:
        out=ROOT/'outputs/smoke';paths=sorted((out/'videos').glob('*.mp4'));assert len(paths)==6
        cameras=json.loads((out/'cameras.json').read_text())['cameras'];assert len(cameras)==6
        sheet=Image.new('RGB',(6*176,3*320),'white');draw=ImageDraw.Draw(sheet)
        for column,path in enumerate(paths):
            frames=decode(path);assert frames[0].shape==(1280,704,3)
            for row,index in enumerate((0,10,20)):
                sheet.paste(Image.fromarray(frames[index]).resize((176,320)),(column*176,row*320))
                draw.text((column*176+4,row*320+4),f'{path.stem} {index/25:.2f}s',fill='black')
        preview=out/'preview';preview.mkdir(exist_ok=True);sheet.save(preview/'time_views.jpg',quality=94)
        record.update(status='videos_decoded_camera_count_passed',views=6,dimensions=[704,1280],visual_review='pending')
        path=preview/'validation.json'
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main(sys.argv[1])
