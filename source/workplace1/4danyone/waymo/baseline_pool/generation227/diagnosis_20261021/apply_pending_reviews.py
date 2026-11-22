"""Apply only explicit inspected observations, preserving original move receipts."""
import argparse,collections,csv,fcntl,hashlib,json,os,time
from pathlib import Path
from organize import page,LABELS
D=Path(__file__).resolve().parent;G=D.parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,a):
 tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--observations',default='pending_review_observations.json')
 parser.add_argument('--receipt',default='pending_review_applied.json')
 args=parser.parse_args()
 observation_path=D/args.observations
 result=D/args.receipt
 assert observation_path.parent.resolve()==D.resolve() and result.parent.resolve()==D.resolve()
 lock=(D/'classification_update.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not result.exists(),'Already applied'
 obs=json.loads(observation_path.read_text())
 c=json.loads((D/'classification.json').read_text());old_sha=sha(D/'classification.json');byid={x['index']:x for x in c['rows']};moves=[]
 assert len({x['index'] for x in obs['rows']})==len(obs['rows'])
 # Validate the whole review set before any move; retain historical receipts.
 for review in obs['rows']:
  row=byid[review['index']]
  assert row['id']==review['id'] and row['category']=='pending_review'
  assert review['category'] in LABELS
  assert sha(Path(review['evidence']))==review['evidence_sha256']
  if review.get('contact_evidence'):
   assert sha(Path(review['contact_evidence']))==review['contact_sha256']
  old=Path(row['output'])
  assert sha(old/'cameras.json')==review['camera_sha256']
  if review['category']!='pending_review':
   new=G/'reviewed'/review['category']/old.relative_to(G/'reviewed/pending_review')
   assert not new.exists() and not old.is_symlink()
   manifest=json.loads((old/'manifest.json').read_text())
   assert {name:sha(old/name) for name in manifest['retained']}==manifest['retained']
 for review in obs['rows']:
  row=byid[review['index']];assert row['id']==review['id'] and row['category']=='pending_review'
  assert sha(Path(review['evidence']))==review['evidence_sha256']
  old=Path(row['output']);assert sha(old/'cameras.json')==review['camera_sha256']
  if review['category']!='pending_review':
   new=G/'reviewed'/review['category']/old.relative_to(G/'reviewed/pending_review');assert not new.exists() and not old.is_symlink()
   manifest=json.loads((old/'manifest.json').read_text());before={name:sha(old/name) for name in manifest['retained']};assert before==manifest['retained']
   stats={name:(old/name).stat().st_ino for name in before}
   new.parent.mkdir(parents=True,exist_ok=True);assert new.parent.stat().st_dev==old.stat().st_dev
   old.rename(new)
   try:old.symlink_to(os.path.relpath(new,old.parent),target_is_directory=True)
   except BaseException:new.rename(old);raise
   assert old.resolve()==new.resolve() and Path(row['old_output']).resolve()==new.resolve()
   assert all(sha(new/name)==digest and (new/name).stat().st_ino==stats[name] for name,digest in before.items())
   entry={'id':row['id'],'old':str(old),'new':str(new),'hashes':before,'copied_video_bytes':0}
   with (D/'pending_review_moves.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
   moves.append(entry);row['output']=str(new)
  row.update(category=review['category'],stage=review['stage'],note=review['note'],scope=review['scope'],evidence=review['evidence'])
 c.update(updated_at=time.time(),counts=dict(collections.Counter(x['category'] for x in c['rows'])),latest_review_evidence=observation_path.name)
 save(D/'classification.json',c)
 with (D/'classification.csv').open('w',newline='') as f:
  out=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in c['rows'] for k in row)));out.writeheader();out.writerows(c['rows'])
 page(G/'index.html',c['rows'],'4DAnyone 227条生成结果：问题诊断与分类')
 for key,label in LABELS.items():page(G/'reviewed'/key/'index.html',[x for x in c['rows'] if x['category']==key],label)
 from html.parser import HTMLParser
 from urllib.parse import unquote,urlsplit
 class Check(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ('src','href','poster') and v and not urlsplit(v).scheme:assert (self.root/unquote(urlsplit(v).path)).exists(),v
 for path in [G/'index.html',*(G/'reviewed'/k/'index.html' for k in LABELS)]:
  parser=Check();parser.root=path.parent;parser.feed(path.read_text())
 save(result,{'at':time.time(),'reviewed':len(obs['rows']),'moved':len(moves),'counts':c['counts'],'previous_classification_sha256':old_sha,'current_classification_sha256':sha(D/'classification.json'),'observation_sha256':sha(observation_path),'copied_video_bytes':0,'all_gallery_links_exist':True,'original_move_receipts_preserved':True})
 print(c['counts'])
if __name__=='__main__':main()
