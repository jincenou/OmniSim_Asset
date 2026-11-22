"""Download official code and release checkpoints in an isolated directory."""
import asyncio,hashlib,json,time,zipfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'vendor';MODEL=ROOT/'models';PROXY='http://127.0.0.1:17904'
STATE={'state':'running','sources':{'code':'https://github.com/sanweiliti/RoHM.git','weights':'https://github.com/sanweiliti/RoHM/releases/download/v0/checkpoints.zip'}}
def save():
 p=ROOT/'rohm_download.json';q=p.with_suffix('.writing');q.write_text(json.dumps(STATE,indent=2)+'\n');q.replace(p)
async def command(name,args):
 with (ROOT/'logs'/f'rohm_{name}.log').open('x') as log:
  begin=time.time();p=await asyncio.create_subprocess_exec(*args,stdout=log,stderr=log)
  STATE[name]={'state':'running','pid':p.pid,'started':begin};save();rc=await p.wait()
  STATE[name].update(state='downloaded' if rc==0 else 'failed',exit_code=rc,seconds=time.time()-begin);save()
  if rc:raise RuntimeError((name,rc))
async def main():
 DEST.mkdir(exist_ok=True);MODEL.mkdir(exist_ok=True);save()
 results=await asyncio.gather(command('code',['git','-c',f'http.proxy={PROXY}','clone','--depth','1',STATE['sources']['code'],str(DEST/'RoHM')]),
  command('weights',['curl','--proxy',PROXY,'-fL','--retry','2','--connect-timeout','15','--max-time','3600','-o',str(MODEL/'rohm_checkpoints.zip.part'),STATE['sources']['weights']]),return_exceptions=True)
 if any(isinstance(x,Exception) for x in results):STATE['state']='download_failed';STATE['errors']=[repr(x) for x in results if isinstance(x,Exception)];save();return
 archive=MODEL/'rohm_checkpoints.zip.part'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  members=z.namelist()
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in members)
  STATE['zip_members']=members
  unpack=MODEL/'rohm';unpack.mkdir(exist_ok=False);z.extractall(unpack)
 STATE['weights']['sha256']=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest();STATE['weights']['bytes']=archive.stat().st_size
 archive.rename(MODEL/'rohm_checkpoints.zip')
 p=await asyncio.create_subprocess_exec('git','-C',str(DEST/'RoHM'),'rev-parse','HEAD',stdout=asyncio.subprocess.PIPE);stdout,_=await p.communicate();assert p.returncode==0
 STATE['code']['revision']=stdout.decode().strip();STATE['state']='downloaded_zip_crc_verified_not_inference_tested';STATE['integrity_scope']='Official HTTPS source, zip CRC and local SHA256 recorded; no independently published digest was supplied.';save()
if __name__=='__main__':asyncio.run(main())
