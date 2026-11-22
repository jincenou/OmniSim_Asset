"""Exercise the actual author/local opacity-promotion statements on small graphs."""
import ast, hashlib, json, subprocess, textwrap
from pathlib import Path
from types import SimpleNamespace
import torch
from run_queue import ROOT, save, stamp

REPO=ROOT/'sources/cem4dgs'
PATH='scene/v_gaussian_model.py'

def statements(source, start):
    begin=source.index('        '+start)
    end=source.index('\n        d = {',begin)
    tree=ast.parse(textwrap.dedent(source[begin:end]))
    # The failing historical block explicitly allocates on CUDA. Exercise its
    # invalid indices on CPU so the expected failure cannot poison a GPU context.
    class CpuAllocation(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant('cpu'),node) if node.value=='cuda' else node
    return compile(ast.fix_missing_locations(CpuAllocation().visit(tree)),PATH,'exec')

def fixture(n,m,seed):
    g=torch.Generator().manual_seed(seed)
    levels=torch.randint(0,2,(n+m,1),generator=g)
    levels[0]=1
    parent=n+torch.randint(m,(n+m,1),generator=g)
    parent[0]=n+m-1
    selected=torch.rand(n,generator=g)>.4
    selected[0]=True
    model=SimpleNamespace(_xyz=torch.zeros(n,3),_xyz_motion=torch.zeros(m,3),
        _level=levels,_parent=parent,
        _opacity_duration_center=torch.rand(n,2,1,generator=g),
        _opacity_duration_var=torch.rand(n,2,1,generator=g))
    return dict(torch=torch,self=model,copy_to_dynamic_mask=selected,
        new_opacity_duration_center=torch.zeros(int(selected.sum()),2,1),
        new_opacity_duration_var=torch.ones(int(selected.sum()),2,1))

def evaluate(code,n,m,seed):
    ns=fixture(n,m,seed)
    exec(code,ns)
    return ns

def main():
    original=subprocess.check_output(['git','show','HEAD:'+PATH],cwd=REPO,text=True)
    current=(REPO/PATH).read_text()
    old=statements(original,'# Child splats inherit')
    new=statements(current,'# Temporal opacity belongs')
    cases=[(8,m,s) for m in [1,3,8] for s in range(40)]
    for n,m,s in cases:
        a,b=evaluate(old,n,m,s),evaluate(new,n,m,s)
        for key in ['new_opacity_duration_center','new_opacity_duration_var']:
            assert torch.equal(a[key],b[key]),(n,m,s,key)
    try:
        evaluate(old,8,12,42)
    except IndexError as exc:
        expected_error=str(exc)
    else:
        raise AssertionError('Historical code failed to reproduce out-of-bounds')
    fixed=evaluate(new,8,12,42)
    mask=fixed['copy_to_dynamic_mask']
    children=(fixed['self']._level[:8,0]>0)
    for suffix in ['center','var']:
        assert torch.equal(fixed['new_opacity_duration_'+suffix][children[mask]],
                           getattr(fixed['self'],'_opacity_duration_'+suffix)[mask & children])
    save(ROOT/'research/promotion_regression.json',dict(state='passed',updated_at=stamp(),
        tested_source_sha256=hashlib.sha256(current.encode()).hexdigest(),
        valid_graph_exact_equivalence_cases=len(cases),
        overflow_fixture=dict(rendered_splats=8,motion_groups=12,old_error=expected_error,
                              fixed_preserves_own_child_opacity=True),
        scope='Actual opacity-promotion source statements on CPU; full training remains separately required',
        reason='Removed parent-group lookup was overwritten by identical child masks; temporal opacity arrays index rendered splats, not virtual motion groups.'))
    print('PASS: 120 valid graphs exactly equal; historical out-of-bounds reproduced and repaired.')

if __name__=='__main__':main()
