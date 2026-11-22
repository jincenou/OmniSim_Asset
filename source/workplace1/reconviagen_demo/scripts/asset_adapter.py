"""Adapt native SH0 Gaussians to the existing DriveLab curation contract.

The payload stores canonical normalized geometry; the original insertion
loader applies the shared metric length once. Orientation must be reviewed.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
PIPELINE=Path('/mnt/wmj/workspace4/pipeline')
sys.path.insert(0,str(PIPELINE))
from curate.core import load_rigid
from curate.stages import _export_ply,_grounding,_yaw_rotation,UP_FIX

FIELDS=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]

def inspect(path):
    vertex=PlyData.read(str(path))['vertex'].data
    missing=set(FIELDS)-set(vertex.dtype.names)
    if missing:raise ValueError(f'Missing Gaussian fields: {sorted(missing)}')
    if not len(vertex):raise ValueError('Empty Gaussian payload')
    for name in vertex.dtype.names:
        if not np.isfinite(vertex[name]).all():raise ValueError(f'Nonfinite field {name}')
    rest=[n for n in vertex.dtype.names if n.startswith('f_rest_')]
    if any(np.any(vertex[n]!=0) for n in rest):
        raise ValueError('Nonzero high-order SH requires a directional SH rotation implementation')
    gs=load_rigid(Path(path))
    norms=np.linalg.norm(gs.rot[0],axis=-1)
    if (norms<1e-8).any():raise ValueError('Degenerate quaternion')
    if not np.isfinite(gs.scale).all() or np.any(gs.scale<=0):raise ValueError('Invalid activated scales')
    if not np.isfinite(gs.opacity).all() or np.any((gs.opacity<0)|(gs.opacity>1)):
        raise ValueError('Invalid opacity')
    if not (gs.opacity>.01).any():raise ValueError('No visible Gaussians')
    lo,hi=np.quantile(gs.xyz[0],[.005,.995],axis=0)
    return dict(gaussian_count=len(vertex),fields=list(vertex.dtype.names),
        quaternion_norm_min_max=[float(norms.min()),float(norms.max())],
        scale_min_max=[float(gs.scale.min()),float(gs.scale.max())],
        opacity_min_max=[float(gs.opacity.min()),float(gs.opacity.max())],
        robust_bounds=[lo.tolist(),hi.tolist()],nonzero_high_order_sh=False),gs

def transform(source,target,rotation,translation,scale=1.):
    """Use the original serializer for rotation, then apply uniform scaling."""
    rotation=np.asarray(rotation,dtype=np.float64);translation=np.asarray(translation,dtype=np.float64)
    if rotation.shape!=(3,3) or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1,atol=1e-6):
        raise ValueError('Expected a proper rotation')
    if not np.isfinite(scale) or scale<=0:raise ValueError('Expected positive uniform scale')
    if translation.shape!=(3,) or not np.isfinite(translation).all():raise ValueError('Invalid translation')
    if Path(target).exists():raise FileExistsError(target)
    inspect(source)
    # Original exporter applies R to positions and quaternion/covariance.
    # t/scale makes subsequent position scaling exactly s R mu + t.
    _export_ply(Path(source),Path(target),rotation,translation/scale)
    ply=PlyData.read(str(target));v=ply['vertex'].data.copy()
    for name in ['x','y','z']:v[name]*=scale
    for i in range(3):v[f'scale_{i}']+=np.log(scale)
    q=np.stack([v[f'rot_{i}'] for i in range(4)],axis=1);q/=np.linalg.norm(q,axis=1,keepdims=True)
    for i in range(4):v[f'rot_{i}']=q[:,i]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(target))
    return inspect(target)[0]

def export(source,out,case,variant,raw_up,yaw_degrees=None,flip=False,heading_verified=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before,gs=inspect(source)
    up=UP_FIX if raw_up=='-y' else np.eye(3)
    yaw=_yaw_rotation(gs,'long_axis',up_rotation=up) if yaw_degrees is None else Rotation.from_euler('z',yaw_degrees,degrees=True).as_matrix()
    rotation=yaw@up
    if flip:rotation=Rotation.from_euler('z',180,degrees=True).as_matrix()@rotation
    translation,size=_grounding(gs.xyz[0].astype(np.float64)@rotation.T)
    target=out/'canonical.ply';after=transform(source,target,rotation,translation)
    length=float(case['metric_dimensions_lwh'][0]);factor=length/size[0]
    metric=np.asarray(size)*factor
    uid=f"car_reconviagen_{case['case_id']}_{variant}"
    if '_seed' in out.parent.name:
        uid+='_seed'+out.parent.name.rsplit('_seed',1)[1]
    entry=dict(uid=uid,global_uid='waymo:'+uid,dataset='waymo',category='car',**{'class':'vehicle'},
        source_path=str(Path(source).resolve()),evidence=str(out/'turntable.jpg'),quality_score=0.,
        status='review',defects=['Visual quality and scene insertion have not yet been verified'],
        category_match=True,pose_state='upright',ground_contact='wheels',symmetry_360_ok=False,
        scale=dict(object_desc=case['role'],est_length_m=length,est_width_m=float(case['metric_dimensions_lwh'][1]),
            est_height_m=float(case['metric_dimensions_lwh'][2]),scale_confidence='high'),
        label=dict(sub_type=case.get('subtype','van' if case['case_id']=='service_van' else 'sedan'),damaged=None,usable_as=[]),
        provenance=dict(source_obj_id=case['original_obj_id'],source_manifest=str(ROOT/'inputs'/case['case_id']/'manifest.json')),
        canonical=dict(canonical_path=str(target.resolve()),rotation=rotation.tolist(),translation=translation.tolist(),
            size_xyz=size,heading=dict(front='+X' if heading_verified else 'unverified',confidence=1. if heading_verified else 0.)))
    sidecar=dict(variant=variant,raw_up=raw_up,canonical_payload_units='normalized',metric_scale_baked=False,
        metric_scale_to_apply_once=factor,target_length_m=length,metric_dimensions_from_uniform_scale=metric.tolist(),
        collision_footprint_xy_m=[[-metric[0]/2,-metric[1]/2],[metric[0]/2,-metric[1]/2],
                                  [metric[0]/2,metric[1]/2],[-metric[0]/2,metric[1]/2]],
        scale_source=case['scale_source'],setup=case['setup'],floaters_filter='none; visual review pending',
        native_kernel='exported physical scales already include native 3D minimum kernel; no duplicate 3D filtering',
        quality_status='review',heading_verified=heading_verified,source_validation=before,reloaded_validation=after)
    (out/'metadata.json').write_text(json.dumps(entry,indent=2))
    (out/'library.json').write_text(json.dumps(dict(schema_version=2,dataset='waymo',
        source_manifests=[entry['provenance']['source_manifest']],total=1,
        counts={'car':1},assets=[entry]),indent=2))
    (out/'experiment.json').write_text(json.dumps(sidecar,indent=2))
    return entry,sidecar

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--case',required=True);p.add_argument('--variant',required=True);p.add_argument('--raw-up',choices=['z','-y'],required=True)
    p.add_argument('--yaw-degrees',type=float);p.add_argument('--flip',action='store_true');p.add_argument('--heading-verified',action='store_true')
    a=p.parse_args();case=json.loads((ROOT/'inputs'/a.case/'manifest.json').read_text())
    export(a.source,a.output,case,a.variant,a.raw_up,a.yaw_degrees,a.flip,a.heading_verified)
