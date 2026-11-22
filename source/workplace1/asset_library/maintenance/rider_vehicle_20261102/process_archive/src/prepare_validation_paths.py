from pathlib import Path
import pickle,json,numpy as np
W=Path(__file__).resolve().parents[1]
source=Path('/mnt/ojc/workplace1/reconviagen_demo/scene_validation/inputs/scene-0103.pkl');d=pickle.load(source.open('rb'));t0=1533151608.6124039;i=int(np.argmin(abs(d['timestamps']-t0)));e=d['ego_to_world'][i];base=np.load(W/'configs/scene0103_straight.npz');ts=base['ts_grid'];rot=e[:2,:2];ego=e[:2,3]
for mode,start,velocity in [('near_contact',[10.,0.],[2.,0.]),('occlusion',[28.,-6.],[.4,1.5])]:
 pos=np.array(start)[None,:]+(ts-t0)[:,None]*np.array(velocity)[None,:];xy=pos@rot.T+ego;vel=rot@velocity;yaw=float(np.arctan2(vel[1],vel[0]));traj=np.column_stack([xy,np.full(len(ts),yaw),np.full(len(ts),np.linalg.norm(vel))])
 meta={'purpose':mode,'source':str(source),'anchor_epoch':t0,'ego_reference_index':i,'start_relative_to_ego_m':start,'velocity_in_ego_axes_mps':velocity,'occluder_track':'3620feb00d744241a94855f3a8913a78' if mode=='occlusion' else None,'occluder_relative_position_m':[23.2,-4.59] if mode=='occlusion' else None,'scope':'Validation trajectory, not claimed as behavior-planner output'}
 np.savez(W/'configs'/('scene0103_'+mode+'.npz'),ts_grid=ts,traj=traj,meta=json.dumps(meta));print(mode,meta)
