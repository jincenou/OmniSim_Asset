"""Capture actual renderer camera arguments, stopping before rasterization; no image regeneration."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1]
source=R/'scripts/insert_scene_validation.py'
text=source.read_text(); marker='    # ---- patch rasterization to append asset gaussians ----'
assert text.count(marker)==1
payload='''    class CapturedCamera(Exception): pass
    captured={}
    def capture(*a,**kw):
        for k in ['viewmats','Ks','width','height','linear_velocity','angular_velocity','rolling_shutter_time','rolling_shutter','time_to_center_pixel']:
            if k in kw:
                v=kw[k];captured[k]=v.detach().cpu().tolist() if torch.is_tensor(v) else str(v) if not isinstance(v,(int,float,str,bool,type(None))) else v
        raise CapturedCamera()
    splatad_mod.rasterization=capture
    records=[]
    with torch.no_grad():
        for fi,anchor in enumerate(anchors):
            for si in layout_idx:
                cam=min(by_sensor[si],key=lambda c:abs(cam_time(c)-cam_time(anchor)))
                captured.clear()
                raw=cam.camera_to_worlds.detach().cpu().tolist()
                metadata={k:v.detach().cpu().tolist() if torch.is_tensor(v) else v for k,v in cam.metadata.items()}
                try:model.get_camera_outputs(cam.to(device))
                except CapturedCamera:pass
                else:raise RuntimeError('Rasterization capture was not reached')
                records.append(dict(frame=fi,sensor=layout[layout_idx.index(si)],camera_time_relative_s=cam_time(cam),epoch_s=cam_time(cam)+min_time,camera_to_model_raw_opengl=raw,metadata=metadata,rasterization=dict(captured)))
    result=dict(scene=seq,checkpoint=str(ckpt_path),step=step,dataparser_world_to_model=w2m.tolist(),dataparser_scale=float(dpo.dataparser_scale),min_sensor_epoch_s=min_time,camera_optimizer_in_eval=bool(model.config.use_camopt_in_eval),camera_coordinates='raw camera_to_model uses OpenGL; captured viewmats map model to OpenCV camera including evaluated camera optimization and rolling shutter center offset',native_render_vs_video='native render Ks/width/height; 6cam video tiles were downscaled by factor 2 after rendering',records=records)
    Path(args.out_dir,'cameras.json').write_text(json.dumps(result,indent=2))
    print('CAMERA_EXPORT_COMPLETE',seq,len(records),flush=True)

if __name__=='__main__':main()
'''
exec(compile(text.split(marker)[0]+payload,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
