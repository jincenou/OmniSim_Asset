"""Explicit camera split and counts of samples actually optimized, not prefetched."""
import hashlib
import json
from pathlib import Path
import numpy as np


def camera_split(parser, train_ids, validation_ids):
    ids=[int(value)-1 for value in parser.camera_ids]
    if len(set(ids))!=len(ids):raise ValueError('Camera records are not unique')
    if not train_ids or not validation_ids:raise ValueError('Explicit nonempty training and validation camera IDs required')
    train=set(train_ids);val=set(validation_ids)
    if len(train)!=len(train_ids) or len(val)!=len(validation_ids):raise ValueError('Duplicate camera IDs')
    if train & val:raise ValueError('Training/validation camera overlap')
    if (train|val)-set(ids):raise ValueError('Requested camera ID missing from actual parser')
    test=[index for index,value in enumerate(ids) if value in val]
    remove=[index for index,value in enumerate(ids) if value not in train|val]
    return test,remove,ids


def data_fingerprint(cfg):
    paths=[Path(cfg.init_npz_path),*sorted((Path(cfg.data_dir)/'sparse/0').glob('*'))]
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    return {'data_dir':str(Path(cfg.data_dir).resolve()),'frame_range':[cfg.start_frame,cfg.end_frame],
            'data_factor':cfg.data_factor,'files':hashes,
            'rgb_alpha_integrity':'Dataset content comes from fixed common master; this fingerprint covers initialization and COLMAP, not every RGBA byte'}


class SamplingLedger:
    def __init__(self,parser,cfg):
        self.test,self.remove,self.ids=camera_split(parser,cfg.train_view_ids,cfg.validation_view_ids)
        self.train=sorted(cfg.train_view_ids);self.validation=sorted(cfg.validation_view_ids)
        self.frames=cfg.end_frame-cfg.start_frame
        self.counts=np.zeros((len(self.ids),self.frames),dtype=np.int64)
        self.steps=0
        self.fingerprint=data_fingerprint(cfg)

    def consumed(self,data,step):
        if step!=self.steps:raise ValueError(f'Noncontiguous optimizer steps: {step} vs {self.steps}')
        camera=int(data['camera_id'].item());index=int(data['camera_idx'].item());frame=int(data['frame_offset'].item())
        if camera not in self.train or self.ids[index]!=camera or not 0<=frame<self.frames:
            raise ValueError('Actual optimized sample violates declared camera/time split')
        self.counts[index,frame]+=1;self.steps+=1

    def state(self):
        return {'version':1,'camera_ids':self.ids,'train_camera_ids':self.train,'validation_camera_ids':self.validation,
                'frame_count':self.frames,'optimized_steps':self.steps,'counts_by_camera_and_frame':self.counts.tolist(),
                'fingerprint':self.fingerprint,'count_scope':'Incremented after optimizer.step; excludes DataLoader prefetch and evaluation'}

    def restore(self,state,checkpoint_step):
        current=self.state()
        for key in ('version','camera_ids','train_camera_ids','validation_camera_ids','frame_count','fingerprint'):
            if state[key]!=current[key]:raise ValueError(f'Resume changes controlled experiment: {key}')
        counts=np.asarray(state['counts_by_camera_and_frame'])
        if counts.shape!=self.counts.shape or counts.dtype.kind not in 'iu' or (counts<0).any():
            raise ValueError('Invalid sampling counts')
        if int(counts.sum())!=checkpoint_step+1 or state['optimized_steps']!=checkpoint_step+1:
            raise ValueError('Sampling counts do not cover the saved optimizer steps')
        excluded=[index for index,value in enumerate(self.ids) if value not in self.train]
        if counts[excluded].any():raise ValueError('Saved counts show validation or excluded cameras were trained')
        self.counts=counts.astype(np.int64);self.steps=checkpoint_step+1

    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state(),indent=2,allow_nan=False)+'\n');tmp.replace(path)
