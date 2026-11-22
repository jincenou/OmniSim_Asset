"""Frozen, nested 1/2/3/4-real H ablation; never changes prior experiment outputs."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, math, os, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'h_count_ablation'
PY = ROOT / 'env/bin/python'
RENDER_PY = '/home/wmj/miniconda3/envs/sim/bin/python'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def freeze():
    target = WORK / 'registry.json'
    if target.exists():
        raise FileExistsError(target)
    tasks = []
    for case in json.loads((ROOT/'inputs/random10_cases.json').read_text()):
        cid = case['case_id']
        original = ROOT/'results'/cid/'H_seed42/input_manifest.json'
        rows = json.loads(original.read_text())['images']
        real = [r for r in rows if r['source_type'] == 'real']
        flux = [r for r in rows if r['source_type'] == 'flux']
        assert len(real) == 4 and len(flux) == 2
        selected = [next(r for r in real if r['id'] == case['flux_reference_id'])]
        while len(selected) < 4:
            def score(r):
                gap = min(abs(math.atan2(math.sin(r['view_azimuth_rad']-s['view_azimuth_rad']),
                                          math.cos(r['view_azimuth_rad']-s['view_azimuth_rad']))) for s in selected)
                return (gap, r.get('mask_score', 0), r.get('bbox_area', 0), r['id'])
            selected.append(max([r for r in real if r not in selected], key=score))
        for count in range(1, 5):
            ids = {r['id'] for r in selected[:count]}
            # Keep original input order so four-real is exactly the previous H run.
            chosen = [r for r in rows if r['source_type'] == 'flux' or r['id'] in ids]
            assert len(chosen) == count+2 and all(r['split'] == 'train' for r in chosen)
            for row in chosen:
                assert sha(row['path']) == row['sha256']
            variant = 'H' if count == 4 else f'H_real{count}'
            tasks.append(dict(case_id=cid, real_count=count, variant=variant, seed=42,
                images=chosen, selection_order=[r['id'] for r in selected],
                input_hashes=[dict(id=r['id'], rgb_sha256=sha(r['path']), rgba_sha256=sha(r['rgba_path'])) for r in chosen],
                baseline_input_manifest=str(original), baseline_input_sha256=sha(original),
                reused=count == 4))
    write(target, dict(version=1, frozen_unix=time.time(), seed=42,
        policy='Anchor is fixed FLUX source; greedy farthest azimuth among four previously reviewed training views; nested subsets; original input order retained. No output-based selection.',
        limitations='Ten existing Waymo metadata-assisted identities, not nuScenes validation; fixed training-source metrics. No new FLUX or Qwen changes in this count-only experiment.',
        tasks=tasks))
    print('FROZEN', len(tasks), 'tasks; 10 existing H4 and 30 new generations', flush=True)

def generate(gpus):
    tasks=[t for t in json.loads((WORK/'registry.json').read_text())['tasks'] if not t['reused']]
    def worker(item):
        gpu, assigned = item
        env=os.environ.copy()
        env['PATH']='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH']
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MAX_JOBS='2',
                   CPATH='/home/wmj/miniconda3/envs/splatad/include', TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'))
        for t in assigned:
            cid,v=t['case_id'],t['variant']; out=ROOT/'results'/cid/f'{v}_seed42'
            status=WORK/'status'/f'{cid}_{v}.json'; log=WORK/'logs'/f'{cid}_{v}.log'
            if (out/'asset/turntable.jpg').exists() and (out/'inference.json').exists():
                print('EXISTS', cid,v,flush=True);continue
            log.parent.mkdir(parents=True,exist_ok=True)
            write(status,dict(status='running',gpu=gpu,started_unix=time.time(),case_id=cid,variant=v))
            print('START',gpu,cid,v,flush=True)
            try:
                for row,h in zip(t['images'],t['input_hashes']):
                    assert sha(row['path'])==h['rgb_sha256'] and sha(row['rgba_path'])==h['rgba_sha256']
                out.mkdir(parents=True,exist_ok=True)
                manifest=dict(case_id=cid,variant=v,seed=42,images=t['images'],ablation_registry_sha256=sha(WORK/'registry.json'))
                if (out/'input_manifest.json').exists():
                    assert json.loads((out/'input_manifest.json').read_text())==manifest
                else: write(out/'input_manifest.json',manifest)
                commands=[]
                if not (out/'raw.ply').exists():
                    commands.append([str(PY),str(ROOT/'scripts/runtime.py'),'--output',str(out),'--seed','42','--images']+[r['rgba_path'] for r in t['images']])
                if not (out/'asset/canonical.ply').exists():
                    commands.append([str(PY),str(ROOT/'scripts/prepare_asset.py'),'--case',cid,'--variant',v])
                case=json.loads((ROOT/'inputs'/cid/'manifest.json').read_text())
                commands.append([RENDER_PY,str(ROOT/'scripts/render_orbit.py'),'--ply',str(out/'asset/canonical.ply'),'--output',str(out/'asset'),'--length',str(case['metric_dimensions_lwh'][0]),'--video-frames','32'])
                with log.open('a') as f:
                    for command in commands:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                write(status,dict(status='heading_review_pending',gpu=gpu,finished_unix=time.time(),case_id=cid,variant=v))
                print('DONE',gpu,cid,v,flush=True)
            except Exception as e:
                write(status,dict(status='failed',gpu=gpu,error=str(e),case_id=cid,variant=v,log=str(log)))
                print('FAILED',gpu,cid,v,str(e),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,[(g,tasks[i::len(gpus)]) for i,g in enumerate(gpus)]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','generate']);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:generate([int(g) for g in a.gpus.split(',')])
