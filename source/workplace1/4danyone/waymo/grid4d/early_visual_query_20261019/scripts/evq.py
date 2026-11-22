"""Early visual queries produce target-camera-independent 3D position/DC residuals.

R1: latent-only; R2: fixed cross samples + foreground-weighted mean;
R3: learned bounded offsets + two-head content Q/K/V attention.
All three share node embedding, geometric association and residual decoder.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import native_world,base_features,C0

def pe(x,levels=4):
    frequencies=(2.**torch.arange(levels,device=x.device,dtype=x.dtype))*math.pi
    phase=x[...,None]*frequencies
    return torch.cat([x,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)

def pyramid(rgba):
    """4x4xHxW straight RGBA -> two immutable 4x6xhxw input levels."""
    rgb=rgba[:,:3]*rgba[:,3:]+.5*(1-rgba[:,3:]);values=torch.cat([rgb,rgba[:,3:]],1);out=[]
    kernel=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],device=rgba.device,dtype=rgba.dtype)/8
    for scale in [2,4]:
        x=F.avg_pool2d(values,scale);gray=(x[:,:3]*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        gradients=F.conv2d(F.pad(gray,(1,1,1,1),mode='replicate'),torch.stack([kernel,kernel.T])[:,None])
        out.append(torch.cat([x,gradients],1))
    return out

def projected_samples(inputs,grid,projection):
    """Exactly sample a linear projected map, including zero-padded border bias."""
    values=F.grid_sample(inputs,grid,mode='bilinear',padding_mode='zeros',align_corners=False).permute(0,2,3,1)
    mass=F.grid_sample(torch.ones_like(inputs[:,:1]),grid,mode='bilinear',padding_mode='zeros',align_corners=False).permute(0,2,3,1)
    return values,F.linear(values,projection.weight,None)+mass*projection.bias

class EVQ(nn.Module):
    def __init__(self,anchors,center,length,base_dim,arm):
        super().__init__();assert arm in ['R1','R2','R3'];self.arm=arm;self.base_dim=base_dim
        self.register_buffer('anchors',anchors.clone());self.register_buffer('center',center.clone());self.register_buffer('length',torch.as_tensor(length,dtype=anchors.dtype,device=anchors.device));assert float(self.length)>0
        self.content=nn.Parameter(torch.randn(128,64,device=anchors.device)*.02)
        self.context=nn.Sequential(nn.Linear(64+27+9+base_dim,64),nn.SiLU(),nn.Linear(64,64));self.z_norm=nn.LayerNorm(64)
        self.gaussian_key=nn.Sequential(nn.Linear(38,64),nn.SiLU(),nn.Linear(64,64));self.anchor_key=nn.Linear(64,64)
        self.decoder=nn.Sequential(nn.Linear(64+27+9+base_dim,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU());self.head=nn.Linear(128,6);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if arm in ['R2','R3']:
            self.image_projection=nn.Linear(6,64);self.value=nn.Linear(64,64);self.output=nn.Linear(64,64);self.null_token=nn.Parameter(torch.zeros(64))
        if arm=='R3':
            self.query=nn.Linear(64,64);self.key=nn.Linear(64,64);self.offset=nn.Linear(64,4*2*4*2);nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        self.register_buffer('cross',torch.tensor([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]],device=anchors.device))
        self.to(anchors.device)

    def normalized(self,xyz):return (xyz-self.center)/self.length

    def encode(self,deform,t,pyramids=None,projection=None):
        warp,feature=base_features(deform,self.anchors,t,False)
        context=self.context(torch.cat([self.content,pe(self.normalized(self.anchors)),pe(t.expand(128,1)),feature],-1))
        zero=context.sum()*0
        if self.arm=='R1':return self.z_norm(context),dict(offset_regularizer=zero,offset_max=zero,null_mean=zero,hard_valid_fraction=zero+1.)
        assert pyramids is not None and projection is not None and len(pyramids)==2
        # This is the SAME native transform used by our renderer's world_means.
        world=native_world(self.anchors,warp['d_xyz']);hom=torch.cat([world,torch.ones_like(world[:,:1])],-1);clip=torch.einsum('mc,vcd->vmd',hom,projection)
        z=clip[...,3];safe=torch.where(z.abs()>1e-6,z,torch.full_like(z,1e-6));ndc=clip[...,:2]/safe[...,None]
        offsets=4*torch.tanh(self.offset(context)).reshape(128,4,2,4,2).permute(1,0,2,3,4) if self.arm=='R3' else torch.zeros(4,128,2,4,2,device=context.device)
        samples=[];validity=[];priors=[]
        for level,inputs in enumerate(pyramids):
            assert inputs.shape[:2]==(4,6)
            h,w=inputs.shape[-2:];grid=ndc[:,:,None,:]+(self.cross[None,None]+offsets[:,:,level])*inputs.new_tensor([2/w,2/h])
            valid=(z>1e-6)[:,:,None]&((grid>=-1)&(grid<=1)).all(-1)
            values,projected=projected_samples(inputs,grid,self.image_projection)
            # Zero padding also attenuates projection bias near image borders.
            # This avoids caching trainable maps without changing map-first semantics.
            samples.append(projected);priors.append(.05+.95*values[...,3].clamp(0,1));validity.append(valid)
        tokens=torch.stack(samples,2).permute(1,0,2,3,4).reshape(128,32,64);valid=torch.stack(validity,2).permute(1,0,2,3).reshape(128,32);prior=torch.stack(priors,2).permute(1,0,2,3).reshape(128,32)
        null_prior=.05+1-((prior-.05)/.95*valid).mean(-1,keepdim=True)
        tokens=torch.cat([tokens,self.null_token[None,None].expand(128,1,64)],1);prior=torch.cat([prior,null_prior],1);valid=torch.cat([valid,torch.ones(128,1,device=valid.device,dtype=torch.bool)],1)
        values=self.value(tokens).reshape(128,33,2,32).transpose(1,2)
        if self.arm=='R3':
            query=self.query(context).reshape(128,2,32);keys=self.key(tokens).reshape(128,33,2,32).transpose(1,2)
            logits=torch.einsum('mhd,mhkd->mhk',query,keys)/math.sqrt(32)+prior.log()[:,None]
        else:logits=prior.log()[:,None].expand(-1,2,-1)
        weights=logits.masked_fill(~valid[:,None],-torch.inf).softmax(-1);merged=(weights[...,None]*values).sum(2).reshape(128,64)
        z_out=self.z_norm(context+self.output(merged))
        return z_out,dict(offset_regularizer=(offsets/4).square().mean(),offset_max=offsets.abs().max(),null_mean=weights[:,:,-1].mean(),hard_valid_fraction=valid[:,:32].float().mean(),attention_entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean())

    def decode(self,gaussians,t,z,base_feature,gate=1.,chunk=8192):
        """Recompute association from current rows, including after split/clone/prune."""
        xyz=gaussians.get_xyz;attribute=torch.cat([pe(self.normalized(xyz)),gaussians._features_dc[:,0],gaussians._scaling,gaussians._rotation,gaussians._opacity],-1);assert attribute.shape[-1]==38
        keys=self.anchor_key(self.content);means=[];colors=[];top_ids=[]
        for start in range(0,len(xyz),chunk):
            stop=min(start+chunk,len(xyz));x=self.normalized(xyz[start:stop]);q=self.gaussian_key(attribute[start:stop]);distance=(x[:,None]-self.normalized(self.anchors)[None]).square().sum(-1)
            score=q@keys.T/math.sqrt(64)-distance/(2*.15**2);value,index=score.topk(4,dim=-1);weight=value.softmax(-1);latent=(z[index]*weight[...,None]).sum(1)
            h=self.decoder(torch.cat([latent,pe(x),pe(t.expand(stop-start,1)),base_feature[start:stop]],-1));raw=torch.tanh(self.head(h))
            means.append(.03*self.length*raw[:,:3]);colors.append(.1/C0*raw[:,3:]);top_ids.append(index)
        position=torch.cat(means);dc=torch.cat(colors)
        regularizer=(position/(.03*self.length)).square().mean()+(dc/(.1/C0)).square().mean()
        return position*gate,dc*gate,dict(residual_regularizer=regularizer,position_fraction_max=position.abs().max()/self.length,RGB_residual_max=dc.abs().max()*C0,association=torch.cat(top_ids))

    def latent_at(self,deform,frame,count,support=None,projections=None,baked=None):
        """One fixed continuous-time definition for online and baked replay."""
        frame=float(frame);assert 0<=frame<=count-1;lo=int(math.floor(frame));hi=int(math.ceil(frame));fraction=frame-lo
        def at(f):
            if baked is not None:return baked[f]
            t=self.anchors.new_tensor([[f/(count-1)]])
            return self.encode(deform,t,None if self.arm=='R1' else support(f),projections)[0]
        z=at(lo)
        return z if hi==lo else z*(1-fraction)+at(hi)*fraction
