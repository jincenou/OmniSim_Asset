"""Run the frozen 24-view root-local conditioning on this batch's reviewed pose."""
import os,sys,subprocess,importlib.util,json
from prepare_person import ROOT,PROJECT,selected,sha
ident=sys.argv[1];selected(ident)
dest=ROOT/'inputs'/ident
gate=json.loads((dest/'research/motion_gate.json').read_text())
assert gate['visual_review']!='pending' and gate['variants']['static_estimated']['allow_exploratory_generation']
assert gate['input_sha256']==sha(dest/'inputs/main.mp4')
uuid=os.environ['FD_EXPECTED_GPU_UUID']
gpu=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=index','--format=csv,noheader'],text=True).strip()
os.environ.update(CUDA_VISIBLE_DEVICES=gpu,FDANYONE_RUNTIME=str(PROJECT/'repo'),FDANYONE_FRAMES='121',FDANYONE_ROOT_VIEWS='24',
 FDANYONE_WAYMO_ROOT=str(ROOT/'frontend'/ident),FDANYONE_SOURCE_EXPERIMENT=str(dest),FDANYONE_MOTION_VARIANT='static_estimated')
e=ROOT/'research/baseline_evidence';sys.path.insert(0,str(e))
spec=importlib.util.spec_from_file_location('frozen_conditions',e/'conditioning.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
(ROOT/'frontend').mkdir(exist_ok=True)
m.main()
