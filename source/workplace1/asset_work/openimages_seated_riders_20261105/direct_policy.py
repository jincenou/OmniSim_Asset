"""User-requested direct publication: source resemblance is informational."""
from pathlib import Path
import copy

def install(r):
 original=r.gen.prepare_jobs
 def jobs(*args,**kwargs):
  bundle=original(*args,**kwargs)
  for a in bundle['assets']:
   assert not a['real_images']
   a['assembly']['admission_policy']='user_direct_flux_v1'
  bundle['signature']=r.gen.digest(dict(parent=bundle['signature'],policy='user_direct_flux_v1'))
  return bundle
 r.gen.prepare_jobs=jobs
 async def flux(client,row,result,folder):
  folder.mkdir(parents=True,exist_ok=True);segmented=[]
  for i,img in enumerate(result['images']):
   sub=folder/f'view_{i}';sub.mkdir(exist_ok=True)
   boxes=await r.check(client,'Locate the full generated rider and full generated bicycle/scooter separately. Return normalized 0..1000 XYXY boxes including head, both wheels, handlebars. Do not compare to any original photograph.',img['path'],r.LOCATE,sub/'direct_boxes.json')
   await r.segment([dict(image=img['path'],output=str(sub/'direct_segmentation'),**boxes)],folder.name+f'_direct_{i}')
   s=r.read_json(sub/'direct_segmentation/result.json')
   if s['status']!='segmented':raise ValueError(s)
   evidence=sub/'direct_policy.json';r.write_json(evidence,dict(policy='user_direct_flux_v1',source_identity_review=False,admission='user-authorized generation; union alpha extraction only'))
   segmented.append(dict(path=s['rgba'],sha256=r.gen.sha256(s['rgba']),flux_sha256=img['sha256'],complete_assembly=True,review=str(evidence),completeness_basis='union of detected generated rider and vehicle; not a visual quality approval'))
  return dict(flux_assembly_reviews=[],flux_segmented_images=segmented)
 r.flux_review=flux
 original_curate=r.curate
 async def curate(*args,**kwargs):
  state=await original_curate(*args,**kwargs)
  from asset_extraction.rigid_publication import entry_from_metadata,validate_ply
  for record in state['assets'].values():
   if 'entry' not in record:continue
   e=record['entry'];q=e['quality'];q['original_visual_status']=record['status'];q['admission_policy']='user_direct_flux_v1';q['visual_quality_approved']=False
   try:
    if not q['rendering']['all_16_nonempty']:raise ValueError('Empty render')
    validate_ply(e['canonical']['canonical_path'],e['canonical']['size_xyz'])
    proposed=copy.deepcopy(e);proposed['status']='keep';entry_from_metadata(proposed)
    record['status']=e['status']='keep';q['publication_basis']='User requested direct inclusion; technical format checks passed; visual scores informational'
   except Exception as exc:record['technical_publication_error']=repr(exc)
  return state
 r.curate=curate
 # Skip extra source/appearance and support admission gates after technical curation.
 r._verdict=lambda *args,**kwargs:'informational'
