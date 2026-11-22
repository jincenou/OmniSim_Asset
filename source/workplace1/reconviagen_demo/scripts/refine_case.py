"""Run the pinned app_fine registration + optimization with strict failures.

Extracts the complete official camera/pointcloud registration code from the
pinned source. Any registration or optimization exception terminates the run.
Only training inputs are accepted; base outputs remain separate.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys,time,traceback,zipfile
from runtime import ROOT,REPO,load_pipeline

def official_namespace(pipeline,mast3r_model):
    import torch,numpy as np,open3d as o3d,cv2,copy
    from PIL import Image
    from torchvision import transforms as TF
    from trellis.utils import render_utils
    from trellis.utils.general_utils import perform_rodrigues_transformation
    from wheels.vggt.vggt.utils.load_fn import load_and_preprocess_images
    from wheels.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from wheels.mast3r.fast_nn import fast_reciprocal_NNs
    from wheels.dust3r.dust3r.inference import inference
    from wheels.dust3r.dust3r.utils.image import load_images_new
    namespace=locals().copy();namespace['device']='cuda'
    source=REPO/'app_fine.py';tree=ast.parse(source.read_text())
    helper_names={'align_camera','refine_pose_mast3r','pointcloud_registration'}
    class Open3DArrayBoundary(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr in ('Vector3dVector','Vector2iVector'):
                dtype='float64' if node.func.attr=='Vector3dVector' else 'int32'
                node.args[0]=ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='ascontiguousarray',ctx=ast.Load()),args=[node.args[0]],keywords=[ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr=dtype,ctx=ast.Load()))])
            return node
    tree=ast.fix_missing_locations(Open3DArrayBoundary().visit(tree))
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helper_names]
    if len(helpers)!=3:raise ValueError('Unexpected upstream helper layout')
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),namespace)
    generate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_and_extract_glb')
    branch=next(n for n in generate.body if isinstance(n,ast.If) and ast.unparse(n.test)=="refine == 'Yes'")
    if len(branch.body)!=1 or not isinstance(branch.body[0],ast.Try):raise ValueError('Unexpected refine block')
    body=branch.body[0].body
    # Keep base and refined sampling parameters identical; UI defaults differ.
    for node in ast.walk(ast.Module(body=body,type_ignores=[])):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run_refine':
            for kw in node.keywords:
                if kw.arg in ('sparse_structure_sampler_params','slat_sampler_params'):
                    kw.value=ast.Attribute(value=ast.Name(id='pipeline',ctx=ast.Load()),attr=kw.arg,ctx=ast.Load())
    code=compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec')
    return namespace,code

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',choices=['R','F','H'],required=True)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=ROOT/'results'/a.case/f'{a.variant}_refined_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'raw.ply').exists():raise FileExistsError(out/'raw.ply')
    try:
        import torch,numpy as np
        from PIL import Image
        from preprocessing import preprocess_rgba
        sys.path.insert(0,str(REPO/'wheels'))
        from wheels.mast3r.model import AsymmetricMASt3R
        from dreamsim import dreamsim
        manifest=json.loads((ROOT/'inputs'/a.case/'segmented_manifest.json').read_text())
        types={'R':{'real'},'F':{'flux'},'H':{'real','flux'}}[a.variant]
        rows=[x for x in manifest['images'] if x['split']=='train' and x['source_type'] in types]
        if not rows:raise ValueError('No training views')
        multiimages=[]
        for i,row in enumerate(rows):
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen RGB changed')
            # Both registration and inference receive the same 518 RGBA
            # image, as in the official UI after explicit preprocessing.
            processed,_=preprocess_rgba(Image.open(row['rgba_path']))
            processed.save(out/f'processed_{i:02}.png')
            multiimages.append((processed.copy(),row['id']))
        images=[Image.open(out/f'processed_{i:02}.png') for i in range(len(rows))]
        (out/'input_manifest.json').write_text(json.dumps(rows,indent=2))
        pipeline=load_pipeline(include_mesh=True)
        ds=ROOT/'cache/refinement/dreamsim';ds.mkdir(exist_ok=True)
        if not (ds/'dino_vitb16_pretrain.pth').exists():
            with zipfile.ZipFile(ROOT/'cache/refinement/dreamsim_dino_vitb16_checkpoint.zip') as z:
                for member in z.infolist():
                    target=(ds/member.filename).resolve()
                    if not target.is_relative_to(ds.resolve()):raise ValueError('Unsafe archive member')
                z.extractall(ds)
        original_hub_load=torch.hub.load;original_hub_dir=torch.hub.get_dir()
        def local_dino(repo,name,*args,**kwargs):
            if repo=='facebookresearch/dino:main' and name=='dino_vitb16':
                model=original_hub_load(str(ROOT/'cache/refinement/dino_source'),name,source='local',pretrained=False)
                state=torch.load(ROOT/'cache/refinement/dino_vitbase16_pretrain.pth',map_location='cpu',weights_only=True)
                model.load_state_dict(state,strict=True)
                return model
            return original_hub_load(repo,name,*args,**kwargs)
        torch.hub.load=local_dino
        try:pipeline.dreamsim_model=dreamsim(pretrained=True,device='cpu',dreamsim_type='dino_vitb16',cache_dir=str(ds))[0].eval()
        finally:
            torch.hub.load=original_hub_load
            torch.hub.set_dir(original_hub_dir)
        # LPIPS must reuse the already present VGG weights, without writing to
        # the shared DINO cache used above by the pretrained pipeline.
        local_hub=ROOT/'cache/torch_refine/hub'
        vgg=local_hub/'checkpoints/vgg16-397923af.pth'
        if not vgg.is_file():raise FileNotFoundError(vgg)
        torch.hub.set_dir(str(local_hub))
        mast_path=ROOT/'cache/refinement/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
        # Official local checkpoint contains argparse metadata. Limit the
        # weights_only=False exception to this explicitly sourced file.
        original_load=torch.load
        def local_load(path,*args,**kwargs):
            if str(path)==str(mast_path):kwargs['weights_only']=False
            return original_load(path,*args,**kwargs)
        torch.load=local_load
        try:mast=AsymmetricMASt3R.from_pretrained(str(mast_path)).eval()
        finally:torch.load=original_load
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.no_grad():
            outputs,coords,noise=pipeline.run(images,seed=a.seed,formats=['gaussian','mesh'],preprocess_image=False)
        outputs['gaussian'][0].save_ply(str(out/'base_repeat.ply'))
        base_seconds=time.monotonic()-start
        namespace,code=official_namespace(pipeline,mast)
        namespace['o3d'].utility.random.seed(a.seed)
        namespace.update(multiimages=multiimages,image_files=images,outputs=outputs,coords=coords,ss_noise=noise,
            seed=a.seed,ss_refine='No',registration_num_frames=20,trellis_stage1_lr=.1,trellis_stage1_start_t=.5,
            trellis_stage2_lr=.1,trellis_stage2_start_t=.5,multiimage_algo='stochastic')
        start=time.monotonic()
        exec(code,namespace)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        refined=namespace['outputs']['gaussian'][0]
        refined.save_ply(str(out/'raw.ply'))
        np.savez(out/'registration.npz',extrinsics=namespace['target_extrinsics'].detach().cpu().numpy(),
            intrinsics=namespace['target_intrinsics'].detach().cpu().numpy(),fitness=np.array(namespace['target_fitnesses']))
        (out/'inference.json').write_text(json.dumps(dict(seed=a.seed,variant=a.variant+'_refined',
            base_repeat_seconds=base_seconds,refinement_seconds=elapsed,gaussian_count=len(refined.get_xyz),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),status='refinement_succeeded_downstream_pending',
            official_source_sha256=hashlib.sha256((REPO/'app_fine.py').read_bytes()).hexdigest(),
            registration_views=20,ss_refine='No',appearance_lr=.1,appearance_start_t=.5,
            sampling_params={'ss':pipeline.sparse_structure_sampler_params,'slat':pipeline.slat_sampler_params},
            heldout_used=False),indent=2))
        (out/'status.json').write_text(json.dumps({'status':'refinement_succeeded_downstream_pending','base_output_is_not_refined':True},indent=2))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        (out/'status.json').write_text(json.dumps({'status':'failed','base_output_is_not_refined':True}))
        raise

if __name__=='__main__':main()
