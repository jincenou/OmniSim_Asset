"""Read-only batch readiness and publication dependency inventory.

Writes reports only; no payload conversion, original deletion, or producer edit.
Metadata eligibility is not full PLY validation or visual acceptance.
"""
from pathlib import Path
from collections import Counter,defaultdict
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import hashlib,json,math

ROOT=Path('/mnt/ojc/workplace1/asset_library')
OUT=Path(__file__).parent
DIRECTIONAL={'car','truck','bus','van','vehicle','bicycle','motorcycle','traffic_sign','traffic_light'}

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0

central=read(ROOT/'manifests/asset_library_rigid.json')['assets']
legacy=read(ROOT/'rigid/asset_library_rigid.json')['assets']
central_ids=Counter(x.get('global_uid') or x.get('uid') for x in central)
legacy_ids={x.get('global_uid') or x.get('uid') for x in legacy}
file_classes=defaultdict(lambda:{'files':0,'bytes':0})
file_inventory=[];assets=[];archived=[];known_paths=set();receipt_protected=set()
for file in sorted((ROOT/'rigid').rglob('asset.json')):
 rel=file.relative_to(ROOT)
 if '_archive' in rel.parts:
  archived.append(str(rel));continue
 folder=file.parent;meta=read(file);identity=meta.get('global_uid') or meta.get('uid')
 c=meta.get('canonical') or {};category=meta.get('category');status=meta.get('status','missing')
 reasons=[];observations=[];size=c.get('size_xyz');axis=None;scale=None;basis=None
 if identity not in central_ids:reasons.append('not_in_central_manifest')
 if central_ids[identity]!=1:reasons.append('central_identity_not_unique')
 if status!='keep':reasons.append('review_or_non_keep_not_approved_for_publication')
 if not (folder/'asset.ply').is_file():reasons.append('missing_payload')
 if c.get('up')!='+Z':reasons.append('up_axis_not_confirmed_Z')
 if c.get('metric_scale_baked') is not False:reasons.append('metric_baking_state_not_explicit_false')
 if not isinstance(size,list) or len(size)!=3 or not all(valid(x) for x in size):
  reasons.append('invalid_canonical_size')
 else:
  axis=max(range(3),key=lambda i:size[i])
  if category in DIRECTIONAL and c.get('heading_verified') is not True:
   reasons.append('direction_requires_review')
  if category in {'car','truck','bus','van','vehicle','bicycle','motorcycle'} and size[1]>size[0]:
   reasons.append('long_horizontal_axis_Y_for_directional_asset')
  binding=folder/'binding.json';placement=folder/'placement.json'
  if binding.exists():
   b=read(binding);scale=b.get('scale');basis='legacy_binding_fixed_scale'
   if b.get('coordinate',{}).get('metric_scale_baked'):
    reasons.append('binding_metric_baked_conflicts_with_canonical')
   dims=b.get('size_lwh_m')
   if valid(scale) and isinstance(dims,list) and len(dims)==3:
    if any(abs(size[i]*scale-dims[i])>max(1e-5,abs(dims[i])*1e-5) for i in range(3)):
     reasons.append('binding_dimensions_disagree_with_canonical')
   else:reasons.append('invalid_binding_scale_or_dimensions')
  elif placement.exists() and 'metric_scale' in read(placement):
   p=read(placement);scale=p.get('metric_scale');basis='placement_fixed_scale'
  else:
   estimates=meta.get('scale') or {};target=estimates.get(['est_length_m','est_width_m','est_height_m'][axis]);basis='dominant_axis_metric_estimate'
   if not valid(target):reasons.append('missing_selected_axis_metric_estimate')
   else:scale=target/size[axis]
   if estimates.get('scale_confidence') not in {'medium','high'}:reasons.append('metric_estimate_confidence_low_or_missing')
   if axis==1:reasons.append('dominant_Y_metric_width_semantics_need_review')
  if scale is not None and not valid(scale):reasons.append('invalid_scale')
 if not c.get('sha256'):observations.append('no_canonical_source_hash_in_metadata')
 if meta.get('defects'):observations.append('producer_defects_recorded')
 if meta.get('quality_approved') is not True:observations.append('not_certified_by_quality_approved')
 receipt=folder/'package.json'
 if receipt.exists():
  rec=read(receipt)
  for name in (rec.get('files') or {}):receipt_protected.add(str((folder/name).resolve()))
 owned=[]
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.name=='asset.ply' and p.parent==folder:group='canonical_payload_runtime'
  elif p.suffix.lower()=='.ply':group='raw_or_auxiliary_geometry_archive'
  elif p.suffix.lower() in {'.pt','.pth','.npz','.ckpt'}:group='training_or_runtime_archive_requires_review'
  elif p.name in {'asset.json','binding.json','placement.json','package.json','package_complete.json','asset_library_rigid.json','transform.json','assembly.json'}:group='migration_metadata_and_publisher_receipts'
  elif p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.mp4','.gif'}:group='source_review_and_preview_archive'
  else:group='other_provenance_or_workflow_files_review_before_deletion'
  n=p.stat().st_size;file_classes[group]['files']+=1;file_classes[group]['bytes']+=n
  row={'path':str(p.relative_to(ROOT)),'bytes':n,'group':group,'symlink':p.is_symlink(),'referenced_by_preview_pages':[]}
  known_paths.add(str(p.resolve()));file_inventory.append(row);owned.append(row['path'])
 assets.append(dict(id=identity,package=str(folder.relative_to(ROOT)),category=category,status=status,
   metadata_sha256=digest(file),payload=str((folder/'asset.ply').relative_to(ROOT)),
   appears_in_legacy_index=identity in legacy_ids,scale_axis='xyz'[axis] if axis is not None else None,
   proposed_scale=scale,scale_basis=basis,proposed_dimensions_m=[x*scale for x in size] if axis is not None and valid(scale) else None,
   blockers=sorted(set(reasons)),observations=observations,
   readiness='metadata_candidate_needs_payload_and_render_validation' if not reasons else 'needs_review',
   files=owned))

class Links(HTMLParser):
 def __init__(self):super().__init__();self.targets=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in {'src','href','poster'} and v:self.targets.append(v)
references=defaultdict(set)
for p in (ROOT/'previews').rglob('*.html'):
 parser=Links()
 try:parser.feed(p.read_text())
 except (UnicodeDecodeError,OSError):continue
 for url in parser.targets:
  parsed=urlsplit(url)
  if parsed.scheme or parsed.netloc or not parsed.path:continue
  target=str((p.parent/unquote(parsed.path)).resolve())
  if target in known_paths:references[target].add(str(p.relative_to(ROOT)))
for row in file_inventory:
 resolved=str((ROOT/row['path']).resolve())
 row['referenced_by_preview_pages']=sorted(references.get(resolved,[]))
 row['publisher_receipt_reference']=resolved in receipt_protected
 row['delete_authorized']=False
category=defaultdict(Counter)
for a in assets:category[a['category']][a['readiness']]+=1
summary=dict(scope='All current rigid package directories, excluding archives; no human data',
 central_entries=len(central),central_status=dict(Counter(x.get('status','missing') for x in central)),
 legacy_index_entries=len(legacy),package_count=len(assets),archived_packages=len(archived),
 package_status=dict(Counter(a['status'] for a in assets)),
 readiness=dict(Counter(a['readiness'] for a in assets)),
 keep_metadata_candidates=sum(a['status']=='keep' and not a['blockers'] for a in assets),
 keep_needs_review=sum(a['status']=='keep' and bool(a['blockers']) for a in assets),
 blockers=dict(Counter(b for a in assets for b in a['blockers'])),
 categories={k:dict(v) for k,v in sorted(category.items())},file_classes=dict(file_classes),
 publisher_receipt_referenced_files=sum(x['publisher_receipt_reference'] for x in file_inventory),
 preview_referenced_files=sum(bool(x['referenced_by_preview_pages']) for x in file_inventory),
 source_metadata_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'manifests/asset_library_rigid.json',ROOT/'rigid/asset_library_rigid.json']},
 limitations=['No bulk payload read/hash check performed; metadata readiness is not visual quality approval.',
 'Review entries are inventoried, not promoted to keep.',
 'Omitting a file from release is not authorization or evidence that it may be deleted from the production tree.',
 'Source publisher verifies package.json file hashes on rerun; deleting preview/source files can break resume.',
 'HTML reference detection covers literal src/href/poster, not dynamic JavaScript, CSS, external scripts or all workflow consumers.'])
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'asset_readiness.json').write_text(json.dumps({'assets':assets},ensure_ascii=False,indent=2)+'\n')
with (OUT/'file_retention_review.jsonl').open('w') as f:
 for row in file_inventory:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
