from pathlib import Path
import importlib.util,json,tempfile,unittest,copy,builtins,io,os
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('compact_humans','/mnt/ojc/workplace1/human_minimal_toolkit_20261105/compact_humans.py');tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
REPO='/mnt/ojc/workplace1/Hasim_trajectory_assets'
class ToolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.index,cls.entries=tool.records(ROOT/'input');cls.byid={r['id']:r for r in cls.entries}
 def test_shared_index_exclusion(self):
  Library,_,_=tool.hasim_api(REPO);library=Library(ROOT/'release')
  self.assertEqual({a.id for a in library.assets},{'waymo_grid4d_H0007','waymo_grid4d_H1132','nuscenes_grid4d_HN0127'})
  self.assertEqual(len(list((ROOT/'release').rglob('library.json'))),1)
  for p in (ROOT/'release').glob('*/*'):
   self.assertEqual({x.name for x in p.iterdir()},{'asset.json','runtime.npz'})
 def test_independent_load_with_source_access_denied(self):
  Library,_,load=tool.hasim_api(REPO);oldopen,oldio=builtins.open,io.open
  def guard(fn):
   def wrapper(path,*a,**kw):
    if isinstance(path,(str,bytes,os.PathLike)):
     resolved=str(Path(os.fsdecode(path)).resolve())
     if resolved.startswith('/mnt/ojc/workplace1/asset_library/human/') or resolved.startswith(str(ROOT/'input')):raise RuntimeError('source access denied')
    return fn(path,*a,**kw)
   return wrapper
  builtins.open,io.open=guard(oldopen),guard(oldio)
  try:
   for asset in Library(ROOT/'release').assets:
    runtime=load(asset);self.assertGreater(len(runtime.sample(.5).means),0)
    self.assertTrue(all(p.is_relative_to(ROOT/'release') for p in runtime.files_used))
  finally:builtins.open,io.open=oldopen,oldio
 def test_qualified_cycle_preserved(self):
  for aid in ['waymo_grid4d_H0007','waymo_grid4d_H1132']:
   meta=tool.read_json(ROOT/'release/assets'/aid/'asset.json')
   self.assertTrue(meta['cycle_profile']['qualified']);self.assertEqual(meta['playback'],'once')
 def test_resume(self):
  r=self.byid['nuscenes_grid4d_HN0127'];p=ROOT/'release/assets'/r['id']/'runtime.npz';before=tool.digest_file(p)
  result=tool.process_one((r,str(ROOT/'release'),str(ROOT/'resume-report'),tool.BUILTIN_CYCLES,REPO,False,True))
  self.assertTrue(result['roundtrip_exact']);self.assertEqual(tool.digest_file(p),before)
 def test_refuse_changed_existing_metadata(self):
  r=self.byid['nuscenes_grid4d_HN0127'];original=ROOT/'release/assets'/r['id']
  with tempfile.TemporaryDirectory(dir=ROOT) as td:
   dst=Path(td)/'assets'/r['id'];dst.mkdir(parents=True)
   os.link(original/'runtime.npz',dst/'runtime.npz')
   meta=tool.read_json(original/'asset.json');meta['duration_s']+=1;tool.write_json(dst/'asset.json',meta)
   with self.assertRaisesRegex(ValueError,'metadata differs'):
    tool.process_one((r,td,str(ROOT/'tamper-report'),tool.BUILTIN_CYCLES,None,False,True))
 def test_invalid_arrays_rejected(self):
  r=self.byid['waymo_grid4d_H0007'];a=tool.read_arrays(r);m=tool.metadata(r);tool.validate(a,m)
  bad=dict(a);bad['ground_offset']=tool.np.array([0.]);
  with self.assertRaises(ValueError):tool.validate(bad,m)
  bad=dict(a);bad['timestamps']=a['timestamps'][::-1]
  with self.assertRaises(ValueError):tool.validate(bad,m)
 def test_cycle_fingerprint_mismatch_rejected(self):
  r=self.byid['waymo_grid4d_H0007'];a=tool.read_arrays(r);a['ground_offset']=a['ground_offset']+.01
  with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):tool.attach_cycle(r,tool.metadata(r),a,tool.BUILTIN_CYCLES)
 def test_package_traversal_rejected(self):
  with self.assertRaises(ValueError):tool.safe_path(ROOT,'../../human/H0007')
 def test_no_implicit_cycle(self):
  m=tool.read_json(ROOT/'release/assets/nuscenes_grid4d_HN0127/asset.json');self.assertNotIn('cycle_profile',m);self.assertEqual(m['playback'],'once')
if __name__=='__main__':unittest.main(verbosity=2)
