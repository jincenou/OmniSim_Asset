"""Shared training and self-contained replay math; no changes to native sources."""
from common import *
from evq import EVQ,pyramid
from render_evq import render as render_final
from PIL import Image
import math

def specification():
    p=NEW/'PLAN.json'
    return json.loads((p if p.exists() else NEW/'PLAN_DRAFT.json').read_text())

def cold(label,arm):
    config=json.loads((NEW/'data'/label/'base_config.json').read_text())
    data=json.loads((NEW/'data'/label/'cameras.json').read_text());opt=SimpleNamespace(**config['optimization'])
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    g=GaussianModel(config['dataset']['sh_degree']);init=specification()['inputs'][label]['initialization']
    with np.load(init) as x:g.create_from_pcd(BasicPointCloud(points=x['points'],colors=x['colors'],normals=np.zeros_like(x['points'])),5.)
    g.training_setup(opt);d=new_deform(config,regularize=True);d.train_setting(opt)
    q=make_query(label,arm,d.spatial_dim+d.temporal_dim);qo=None if q is None else torch.optim.Adam(q.parameters(),lr=5e-4)
    return g,d,q,qo,config,data

def make_query(label,arm,dim):
    assert arm in ['G0','G1','G2']
    with np.load(NEW/'data'/label/'anchors.npz') as a:
        args=[torch.as_tensor(a['anchors'],device='cuda'),torch.as_tensor(a['center'],device='cuda'),float(a['length']),dim]
    state=rng_state();torch.manual_seed(1042);torch.cuda.manual_seed_all(1042)
    q=EVQ(*args,'R3',variant=arm,chunk=specification()['model']['local_chunk'])
    set_rng(state);return q

class Inputs:
    """Exact uint8 source caching, converted only when used; no trainable map cache."""
    def __init__(self,label,data,all_frames=True):
        self.label=label;self.data=data;self.images={};self.pyramids={}
        self.meta=json.loads(Path(specification()['inputs'][label]['manifest']).read_text())
        self.support_views=specification()['inputs'][label]['support_views']
        self.cameras={v:camera(data,v,0) for v in data['train_views']}
        self.projections=torch.stack([self.cameras[v].full_proj_transform for v in self.support_views])
        self.source_reads=0;self.support_requests=0
        if all_frames:
            for v in data['train_views']:
                for f in range(data['frame_count']):self.rgba(v,f)
    def rgba(self,v,f):
        assert v in self.data['train_views']
        if (v,f) not in self.images:
            p=Path(self.meta['source_data'])/'images'/f'cam{v:02d}'/f'{f:06d}.png'
            with Image.open(p) as im:
                assert im.mode=='RGBA' and im.size==(352,640)
                self.images[v,f]=torch.from_numpy(np.array(im)).permute(2,0,1).contiguous().cuda()
            self.source_reads+=1
        return self.images[v,f].float()/255.
    def target(self,v,f):
        x=self.rgba(v,f);c=self.cameras[v];c.fid=torch.tensor([f/self.data['time_denominator']],device='cuda');c.source_frame=f
        return c,x[:3]*x[3:]+.5*(1-x[3:]),x[3:]
    def support(self,f):
        self.support_requests+=1
        if f not in self.pyramids:
            with torch.no_grad():self.pyramids[f]=pyramid(torch.stack([self.rgba(v,f) for v in self.support_views]))
        return self.pyramids[f]

def geometry(g,d,q,data,frame,step,inputs=None,baked=None,regularize=False):
    t=g.get_xyz.new_tensor([[float(frame)/data['time_denominator']]])
    zero=g.get_xyz.sum()*0;reg=zero;gate=min(1.,max(0.,(step-2999)/1000))
    warp,feat=base_features(d,g.get_xyz.detach(),t,regularize)
    for key,weight in [('reg_spatial',.5),('reg_temporal',.5)]:
        if warp[key] is not None:reg=reg+weight*warp[key].mean()
    base=native_world(g.get_xyz,warp['d_xyz'])
    if baked is not None:
        z=q.latent_at(d,frame,data['frame_count'],baked=baked['Z_geo']);dp,_,_=q.decode(g,t,z,feat,gate)
        lo=int(math.floor(frame));hi=int(math.ceil(frame));frac=float(frame)-lo
        dc_rgb=baked['C_delta'][lo] if lo==hi else baked['C_delta'][lo]*(1-frac)+baked['C_delta'][hi]*frac
        means=base+dp;diag={}
    elif float(frame).is_integer():
        z,ed,context=q.encode_geo(d,t,inputs.support(int(frame)),inputs.projections)
        dp,dc_rgb,dd=q.decode_local(g,t,z,feat,base,context,inputs.support(int(frame)),inputs.projections,gate)
        means=base+dp
        reg=reg+1e-4*(dd['residual_regularizer']+ed['offset_regularizer'])
        diag={k:v for k,v in {**ed,**dd}.items() if k not in ['association']}
    else:
        # Same new continuous definition online: compute adjacent discrete RGB
        # residuals then interpolate. Geometry still decodes interpolated Z.
        lo=int(math.floor(frame));hi=int(math.ceil(frame));frac=float(frame)-lo
        states=[geometry(g,d,q,data,f,step,inputs) for f in [lo,hi]]
        dc_rgb=states[0]['dc_rgb_before_gate']*(1-frac)+states[1]['dc_rgb_before_gate']*frac
        z=q.latent_at(d,frame,data['frame_count'],inputs.support,inputs.projections)
        dp,_,_=q.decode(g,t,z,feat,gate);means=base+dp;diag={}
    sh=torch.cat([g.get_features[:,:1]+gate*dc_rgb[:,None]/C0,g.get_features[:,1:]],1)
    return dict(warp=warp,means=means,sh=sh,reg=reg,diag=diag,dc_rgb_before_gate=dc_rgb)

def raster(g,pipe,cam,state,background=.5,alpha=False):
    w=state['warp']
    return render_final(cam,g,pipe,torch.full((3,),float(background),device='cuda'),w['d_xyz'],w['d_rotation'],w['d_scaling'],override_color=torch.ones_like(g.get_xyz) if alpha else None,world_means_override=state['means'],features_override=state['sh'])

def checkpoint(g,d,q,qo,config,data,step,extra):
    return dict(schema='evq_primitive_recovery_v1',step=step,config=config,data=data,gaussians={k:getattr(g,k).detach() for k in PARAMETERS},statistics={k:getattr(g,k) for k in ['max_radii2D','xyz_gradient_accum','denom']},active_sh_degree=g.active_sh_degree,gaussian_optimizer=g.optimizer.state_dict(),hash=d.grid4d.state_dict(),deform=d.deform.state_dict(),deform_optimizer=d.optimizer.state_dict(),query=None if q is None else q.state_dict(),query_optimizer=None if qo is None else qo.state_dict(),rng=rng_state(),extra=extra)

def restore(state,g,d,q,qo,config,allow_cold=False):
    opt=SimpleNamespace(**config['optimization'])
    for k in PARAMETERS:setattr(g,k,torch.nn.Parameter(state['gaussians'][k].cuda()))
    g.training_setup(opt);g.optimizer.load_state_dict(state['gaussian_optimizer']);g.active_sh_degree=state['active_sh_degree']
    for k,v in state['statistics'].items():setattr(g,k,v.cuda())
    d.grid4d.load_state_dict(state['hash'],strict=True);d.deform.load_state_dict(state['deform'],strict=True);d.optimizer.load_state_dict(state['deform_optimizer'])
    if state['query'] is not None:
        assert q is not None;q.load_state_dict(state['query'],strict=True);qo.load_state_dict(state['query_optimizer'])
    elif q is not None:assert allow_cold and state['step']==2999
    set_rng(state['rng'])
    for k in PARAMETERS:equal(getattr(g,k),state['gaussians'][k])
    equal(g.optimizer.state_dict(),state['gaussian_optimizer']);equal(d.optimizer.state_dict(),state['deform_optimizer']);equal(d.grid4d.state_dict(),state['hash']);equal(d.deform.state_dict(),state['deform']);equal(rng_state(),state['rng'])
    if state['query'] is not None:equal(q.state_dict(),state['query']);equal(qo.state_dict(),state['query_optimizer'])
    return state['step']

@torch.no_grad()
def bake(q,d,g,data,inputs,step=32999):
    geo=[];colors=[]
    for f in range(data['frame_count']):
        t=g.get_xyz.new_tensor([[f/data['time_denominator']]])
        z,_,_=q.encode_geo(d,t,inputs.support(f),inputs.projections)
        state=geometry(g,d,q,data,f,step,inputs)
        geo.append(z.float());colors.append(state['dc_rgb_before_gate'].float())
    return dict(Z_geo=torch.stack(geo),C_delta=torch.stack(colors))
