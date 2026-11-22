from pathlib import Path
import json,zipfile,hashlib,concurrent.futures
R=Path(__file__).resolve().parents[1];S=R/'scene_validation';O=R/'deliveries';D=O/'insertion_handoff_20261023'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pack(name,entries,readme):
 path=O/(name+'.zip');rows=[]
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  z.writestr(name+'/README.txt',readme)
  for src,dest in entries:
   z.write(src,name+'/'+dest);rows.append(dict(path=dest,bytes=src.stat().st_size,sha256=sha(src)))
  z.writestr(name+'/MANIFEST.json',json.dumps(rows,indent=2))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 digest=sha(path);path.with_suffix('.zip.sha256').write_text(digest+'  '+path.name+'\n');print('ARCHIVE',name,path.stat().st_size,flush=True)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=digest,files=len(rows),verification='All ZIP members CRC-tested; per-file SHA256 recorded')
deps=json.loads((D/'EXTERNAL_DEPENDENCIES.json').read_text());bg=[]
for d in deps:
 bg.extend([(Path(d['checkpoint_original']),d['scene']+'/nerfstudio_models/'+Path(d['checkpoint_original']).name),(Path(d['config_original']),d['scene']+'/config.original.yml')])
frames=[]
for r in json.loads((D/'INSERTIONS.json').read_text()):
 src=S/'results'/r['scene']/r['case_id']/r['variant']/'frames'
 ff=sorted(src.glob('*.png'));assert len(ff)==60
 frames.extend((p,'results/'+r['scene']+'/'+r['case_id']+'/'+r['variant']+'/frames/'+p.name) for p in ff)
for d in deps:
 src=S/'results'/d['scene']/'random10_04_white_suv/background/frames';ff=sorted(src.glob('*.png'));assert len(ff)==60
 frames.extend((p,'results/'+d['scene']+'/random10_04_white_suv/background/frames/'+p.name) for p in ff)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 jobs=[ex.submit(pack,'insertion_backgrounds_20261023',bg,'3 original step30000 SplatAD checkpoints and original configs. Use with the core insertion_handoff_20261023 package. nuScenes data and environments remain external; configs retain original paths.\n'),ex.submit(pack,'insertion_frames_20261023',frames,'Original PNG mosaics: 24 insertions +3 backgrounds, 60 frames each =1620 PNG. Matches the native 6-camera videos in the core package. Not per-camera full alpha sequences.\n')]
 results=[j.result() for j in jobs]
(D/'SUPPLEMENT_ARCHIVES.json').write_text(json.dumps(results,indent=2));print('SUPPLEMENTS_COMPLETE',flush=True)
