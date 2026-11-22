"""Primitive-level image queries for BOTH world geometry and DC appearance.

Research adaptation, not an official GaussianFormer checkpoint. The projection
and native covariance contract remain those of the existing Grid4D pipeline.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def pe(x):
    f = 2. ** torch.arange(4, device=x.device, dtype=x.dtype) * math.pi
    phase = x[..., None] * f
    return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)


def pyramid(rgba):
    rgb = rgba[:, :3] * rgba[:, 3:] + .5 * (1-rgba[:, 3:])
    values = torch.cat([rgb, rgba[:, 3:]], 1)
    kernel = rgba.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    out = []
    for scale in (2, 4):
        x = F.avg_pool2d(values, scale)
        gray = (x[:, :3] * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        grad = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), torch.stack([kernel, kernel.T])[:, None])
        out.append(torch.cat([x, grad], 1))
    return out


class DirectQuery(nn.Module):
    def __init__(self, center, length, base_dim, arm, chunk=4096):
        super().__init__()
        assert arm in ('P1', 'P2', 'P3', 'P4')
        self.arm, self.base_dim, self.chunk = arm, base_dim, chunk
        self.use_checkpoint = True
        self.register_buffer('center', center.clone())
        self.register_buffer('length', center.new_tensor(float(length)))
        self.register_buffer('cross', center.new_tensor([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        # 38 Gaussian attributes + 9 time encoding + native hash features.
        self.context = nn.Sequential(nn.Linear(38+9+base_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.norm = nn.LayerNorm(64)
        self.decoder = nn.Sequential(nn.Linear(64+27+9+base_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.head = nn.Linear(128, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if arm != 'P1':
            self.image_projection = nn.Linear(6, 64)
            self.value = nn.Linear(64, 64)
            self.output = nn.Linear(64, 64)
            self.null_token = nn.Parameter(torch.zeros(64))
        if arm in ('P3', 'P4'):
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.offset = nn.Linear(64, 64)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        self.to(center.device)

    def sample(self, context, world, p0, p1, projection, depth_maps):
        n = len(world)
        clip = torch.einsum('nc,vcd->vnd', torch.cat([world, torch.ones_like(world[:, :1])], -1), projection)
        z = clip[..., 3]
        ndc = clip[..., :2] / torch.where(z.abs() > 1e-6, z, torch.full_like(z, 1e-6))[..., None]
        offsets = (4*torch.tanh(self.offset(context))).reshape(n, 4, 2, 4, 2).permute(1, 0, 2, 3, 4) if self.arm in ('P3', 'P4') else context.new_zeros(4, n, 2, 4, 2)
        tokens, priors, valids, vis_values = [], [], [], []
        for level, image in enumerate((p0, p1)):
            h, w = image.shape[-2:]
            grid = ndc[:, :, None] + (self.cross[None, None]+offsets[:, :, level])*image.new_tensor([2/w, 2/h])
            valid = (z > 1e-6)[:, :, None] & ((grid >= -1) & (grid <= 1)).all(-1)
            values = F.grid_sample(image, grid, align_corners=False, padding_mode='zeros').permute(0, 2, 3, 1)
            mass = F.grid_sample(torch.ones_like(image[:, :1]), grid, align_corners=False).permute(0, 2, 3, 1)
            token = F.linear(values, self.image_projection.weight, None) + mass*self.image_projection.bias
            prior = .05+.95*values[..., 3].clamp(0, 1)
            visibility = torch.ones_like(prior)
            if self.arm == 'P4':
                # Detached geometry prior: no gradient incentive to manipulate
                # the depth gate itself. Low rendered alpha stays unknown.
                with torch.no_grad():
                    moments = F.grid_sample(depth_maps, grid.detach(), align_corners=False).permute(0, 2, 3, 1)
                    alpha = moments[..., 1]
                    depth = moments[..., 0]/alpha.clamp_min(1e-6)
                    variance = (moments[..., 2]/alpha.clamp_min(1e-6)-depth.square()).clamp_min(0)
                    sigma = variance.sqrt().clamp_min(.01*self.length)
                    behind = (z.detach()[:, :, None]-depth-.01*self.length).clamp_min(0)
                    candidate = .1+.9*torch.exp(-.5*(behind/sigma).square())
                    visibility = torch.where(alpha >= .2, candidate, torch.ones_like(candidate))
                prior = prior*visibility
            tokens.append(token)
            priors.append(prior)
            valids.append(valid)
            vis_values.append(visibility)
        token = torch.stack(tokens, 2).permute(1, 0, 2, 3, 4).reshape(n, 32, 64)
        prior = torch.stack(priors, 2).permute(1, 0, 2, 3).reshape(n, 32)
        valid = torch.stack(valids, 2).permute(1, 0, 2, 3).reshape(n, 32)
        null_prior = .05+1-(prior*valid).mean(-1, keepdim=True)
        token = torch.cat([token, self.null_token[None, None].expand(n, 1, 64)], 1)
        prior = torch.cat([prior, null_prior], 1)
        valid = torch.cat([valid, torch.ones(n, 1, device=valid.device, dtype=torch.bool)], 1)
        value = self.value(token).reshape(n, 33, 2, 32).transpose(1, 2)
        if self.arm in ('P3', 'P4'):
            query = self.query(context).reshape(n, 2, 32)
            keys = self.key(token).reshape(n, 33, 2, 32).transpose(1, 2)
            logits = torch.einsum('nhd,nhkd->nhk', query, keys)/math.sqrt(32)+prior.log()[:, None]
        else:
            logits = prior.log()[:, None].expand(-1, 2, -1)
        weights = logits.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        merged = (weights[..., None]*value).sum(2).reshape(n, 64)
        latent = self.norm(context+self.output(merged))
        # Return normalized offset penalty ONCE, with point-weighted aggregation.
        info = torch.stack([(offsets/4).square().mean(), weights[:, :, -1].mean(), valid[:, :32].float().mean(), torch.stack(vis_values).mean()])
        return latent, info

    def block(self, attr, feature, world, t, p0, p1, projection, depth_maps):
        time = pe(t.expand(len(attr), 1))
        context = self.context(torch.cat([attr, time, feature], -1))
        if self.arm == 'P1':
            latent = self.norm(context)
            info = context.new_tensor([0., 0., 1., 1.])
        else:
            latent, info = self.sample(context, world, p0, p1, projection, depth_maps)
        raw = torch.tanh(self.head(self.decoder(torch.cat([latent, attr[:, :27], time, feature], -1))))
        delta = torch.cat([.03*self.length*raw[:, :3], .1*raw[:, 3:]], -1)
        return delta, info

    def forward(self, g, feature, world, t, pyramids, projection, depth_maps):
        x = (g.get_xyz-self.center)/self.length
        attr = torch.cat([pe(x), g._features_dc[:, 0], g._scaling, g._rotation, g._opacity], -1)
        assert attr.shape[1] == 38
        outputs, summaries, counts = [], [], []
        for start in range(0, len(x), self.chunk):
            sl = slice(start, start+self.chunk)
            args = (attr[sl], feature[sl], world[sl], t, pyramids[0], pyramids[1], projection, depth_maps)
            if self.use_checkpoint and torch.is_grad_enabled():
                delta, info = checkpoint(self.block, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                delta, info = self.block(*args)
            outputs.append(delta); summaries.append(info); counts.append(len(delta))
        delta = torch.cat(outputs)
        info = (torch.stack(summaries)*delta.new_tensor(counts)[:, None]).sum(0)/len(delta)
        regularizer = (delta[:, :3]/(.03*self.length)).square().mean()+(delta[:, 3:]/.1).square().mean()+info[0]
        return delta, regularizer, dict(offset_regularizer=info[0], null_mean=info[1], hard_valid_fraction=info[2], soft_visibility_mean=info[3], position_fraction_max=delta[:, :3].abs().max()/self.length, RGB_residual_max=delta[:, 3:].abs().max())
