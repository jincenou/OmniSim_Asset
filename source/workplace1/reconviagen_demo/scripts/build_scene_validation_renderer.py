from pathlib import Path
R=Path(__file__).resolve().parents[1];s=(R/'scripts/insert_scene.py').read_text()
s=s.replace('import json\n','import json\nimport hashlib,time\n')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument('--reference_width',type=float,required=True)
    ap.add_argument('--background_only',action='store_true')
    ap.add_argument('--save_diagnostics',action='store_true')
    args = ap.parse_args()
    if (Path(args.out_dir)/'meta.json').exists():raise FileExistsError(args.out_dir)
    started=time.monotonic()''')
s=s.replace('    # ---- color calibration ----','''    if human is not None:raise ValueError('This validation runner is rigid-only')
    # Same reference footprint and deterministic time-indexed contact poses for every route.
    contact_rows=[];last_z=None;last_n=None
    for ti,pose in zip(ts_grid,traj):
        if not np.isfinite(pose[:3]).all():continue
        gz,rg=ground.contact_pose(*pose[:3],length=args.target_len,width=args.reference_width)
        z=gz if last_z is None else .5*last_z+.5*gz
        n=rg[:,2] if last_n is None else .7*last_n+.3*rg[:,2];n=n/np.linalg.norm(n)
        contact_rows.append([float(ti),float(z),*n.tolist(),float(gz)])
        last_z=z;last_n=n
    contact_rows=np.asarray(contact_rows)
    with open(os.path.join(args.out_dir,'contact_trajectory.json'),'w') as f:json.dump(contact_rows.tolist(),f)
    def contact_at(t,yaw):
        z=np.interp(t,contact_rows[:,0],contact_rows[:,1]);n=np.array([np.interp(t,contact_rows[:,0],contact_rows[:,i]) for i in [2,3,4]]);n/=np.linalg.norm(n)
        fwd=np.array([np.cos(yaw),np.sin(yaw),0.]);fwd-=fwd.dot(n)*n;fwd/=np.linalg.norm(fwd)
        return z,np.column_stack([fwd,np.cross(n,fwd),n])
    diagnostics=[]
    # ---- color calibration ----''')
s=s.replace('    orig_rast = splatad_mod.rasterization','''    state['capture']=False
    state['diagnostic']=None
    state['velocity']=None
    orig_rast = splatad_mod.rasterization''')
s=s.replace('                     torch.zeros(len(a_opac), 3, device=device)], dim=0)','                     state["velocity"]], dim=0)')
s=s.replace('        return orig_rast(*a, **kw)','''        result=orig_rast(*a, **kw)
        if state['means'] is not None and state['capture']:
            # Visibility attribution with the exact renderer, depth sorting and camera.
            n_asset=len(a_opac);maskkw=dict(kw);maskkw['colors']=torch.zeros((len(kw['means']),1),device=device);maskkw['colors'][-n_asset:]=1
            maskkw['render_mode']='RGB';maskkw['backgrounds']=None
            visible,_,_=orig_rast(*a,**maskkw)
            only=dict(maskkw)
            for key in ['means','quats','scales','opacities','colors','velocities']:
                if only.get(key) is not None:only[key]=only[key][-n_asset:]
            isolated,_,_=orig_rast(*a,**only)
            va=visible[0,...,0].clamp(0,1);ia=isolated[0,...,0].clamp(0,1)
            mass=float(ia.sum());v=va.cpu().numpy();ii=ia.cpu().numpy();ys,xs=np.where(ii>.1)
            bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
            diag=dict(isolated_alpha_mass=mass,visible_alpha_mass=float(va.sum()),visible_mass_fraction=float(va.sum())/mass if mass>1e-6 else None,visible_pixels_01=int((va>.1).sum()),isolated_pixels_01=int((ia>.1).sum()),projected_bbox=bbox,edge_clipped=bool(len(xs) and (xs.min()==0 or ys.min()==0 or xs.max()==v.shape[1]-1 or ys.max()==v.shape[0]-1)))
            state['diagnostic']=(diag,v,ii)
        return result''')
a=s.index('                    gz, R_body = ground.contact_pose(px, py, yaw, length=args.target_len,');b=s.index('                    R = R_w2m @ R_g',a)
s=s[:a]+'''                    z_prev,R_g=contact_at(t_epoch,yaw)
                    if si == front_idx:
                        tilt_log.append((fi,float(np.degrees(np.arcsin(-R_g[2,0]))),float(np.degrees(np.arcsin(R_g[2,1])))))
'''+s[b:]
s=s.replace('                out = model.get_camera_outputs(cam.to(device))','''                    # Translational+yaw rigid velocity for the original rolling-shutter renderer.
                    dt=.02;pm=interp_traj(ts_grid,traj,max(ts_grid[0],t_epoch-dt));pp=interp_traj(ts_grid,traj,min(ts_grid[-1],t_epoch+dt))
                    vg=np.zeros((len(a_means_np),3))
                    if pm is not None and pp is not None:
                        duration=min(ts_grid[-1],t_epoch+dt)-max(ts_grid[0],t_epoch-dt)
                        linear=(np.array(pp[:2])-np.array(pm[:2]))/duration;rate=np.arctan2(np.sin(pp[2]-pm[2]),np.cos(pp[2]-pm[2]))/duration
                        off=a_means_np@R_g.T;vg[:,0]=linear[0]-rate*off[:,1];vg[:,1]=linear[1]+rate*off[:,0]
                    state['velocity']=torch.tensor(vg@R_w2m.T,dtype=torch.float32,device=device)
                if args.background_only:state['means']=None
                state['capture']=args.save_diagnostics;state['diagnostic']=None
                out = model.get_camera_outputs(cam.to(device))''')
s=s.replace('                tiles.append(rgb)','''                tiles.append(rgb)
                if state['diagnostic'] is not None:
                    diag,visible_alpha,isolated_alpha=state['diagnostic'];diag.update(frame=fi,sensor=layout[layout_idx.index(si)],epoch=t_epoch,pose=list(pose),ground_z=float(z_prev))
                    visible_mask=visible_alpha>.1
                    diag['visible_rgb_mean']=(rgb[visible_mask].mean(0)/255).tolist() if visible_mask.any() else None
                    diagnostics.append(diag)
                    if fi in [0,len(anchors)//2,len(anchors)-1]:
                        folder=Path(args.out_dir)/'diagnostics';folder.mkdir(exist_ok=True)
                        imageio.imwrite(folder/f'f{fi:03}_{diag["sensor"]}.jpg',rgb)
                        imageio.imwrite(folder/f'f{fi:03}_{diag["sensor"]}_visible.png',(visible_alpha*255).astype('uint8'))
                        imageio.imwrite(folder/f'f{fi:03}_{diag["sensor"]}_isolated.png',(isolated_alpha*255).astype('uint8'))''')
s=s.replace('    writer.close()','''    writer.close()
    with open(os.path.join(args.out_dir,'visibility_diagnostics.json'),'w') as f:json.dump(diagnostics,f,indent=2)''')
s=s.replace('                   "step": int(step),','''                   "step": int(step),
                   "fps":args.fps,"downscale":args.downscale,"reference_width_m":args.reference_width,
                   "background_only":args.background_only,"elapsed_seconds":time.monotonic()-started,
                   "trajectory_sha256":hashlib.sha256(Path(args.traj_npz).read_bytes()).hexdigest(),
                   "asset_sha256":hashlib.sha256(Path(args.asset_ply).read_bytes()).hexdigest(),
                   "contact_protocol":"shared reference footprint, 10Hz temporal smoothing before per-camera interpolation; no per-route retuning",
                   "motion_protocol":"append rigid translation+yaw velocity for rolling shutter; pitch/roll rates not modeled",
                   "color_protocol":"same seeded original signed-affine decoder calibration; no per-asset fitting",''')
s=s.replace('fps=args.fps,','fps=args.fps,')
(R/'scripts/insert_scene_validation.py').write_text(s)
print('Wrote scene validation renderer')
