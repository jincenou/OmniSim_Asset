"""Independent diagnostic review; never bypass original or final admission."""
import asyncio,importlib.util,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('rider_worker',ROOT/'run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
SCHEMA=r.schema(camera_change_only={'type':'boolean'},source_complete={'type':'boolean'},same_rider_and_vehicle={'type':'boolean'},physical_pose_consistent={'type':'boolean'},plausible_contacts={'type':'boolean'},major_geometry_defect={'type':'boolean'},recommend_reconstruction={'type':'boolean'},reason={'type':'string'})
async def main():
 out=ROOT/'rejection_diagnostics_v2';out.mkdir(exist_ok=True)
 client=r.AsyncOpenAIClient(r.CONFIG,None);sem=asyncio.Semaphore(3);results=[]
 async def one(chunk,a):
  image=chunk/'assembly_review'/a['obj_id']/'view_0/comparison.jpg'
  if not image.exists():return
  dest=out/(a['obj_id']+'.json')
  async with sem:
   d=await r.ask(client,'Compare A reference and B generated rider with bicycle. Diagnostic review: CAMERA/viewpoint change is not by itself a physical pose change. Compare 3D posture/contact plausibility allowing foreshortening and self-occlusion. Do not demand both hidden feet be visible. Require full head and both wheels in source, consistent rider appearance and vehicle, seated pelvis, plausible hand/handlebar and feet/pedal or footrest support. Reject real identity substitution, changed clothing/body appearance, removed rider, detached limbs, major missing geometry. Minor text/texture differences are allowed. Set recommend_reconstruction true only if source_complete, same_rider_and_vehicle, physical_pose_consistent and plausible_contacts are all true and major_geometry_defect false. Explain visible evidence. Do not assume prior rejection correct.',image,SCHEMA,dest)
  results.append(dict(obj_id=a['obj_id'],chunk=str(chunk),original_error=a.get('assembly_review_error'),decision=d))
 try:
  await client.check_endpoints();jobs=[]
  for f in sorted((ROOT/'chunks').glob('*/completed.json')):
   g=f.parent/'generation/generation.json'
   for a in r.read_json(g)['assets']:
    if a.get('assembly_review_terminal')=='rejected':jobs.append(one(f.parent,a))
  await asyncio.gather(*jobs)
  r.write_json(out/'summary.json',dict(updated=time.time(),reviewed=len(results),recommended=sum(x['decision']['recommend_reconstruction'] for x in results),automatic_admission=False,assets=results))
 finally:await client.close()
if __name__=='__main__':asyncio.run(main())
