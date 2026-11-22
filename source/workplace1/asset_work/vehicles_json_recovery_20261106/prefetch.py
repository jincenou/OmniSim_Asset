import asyncio,importlib.util
from pathlib import Path
p=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recovery',p/'run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
async def main():
 client=r.AsyncOpenAIClient(r.w.CONFIG,None)
 try:
  await client.check_endpoints()
  rows=r.read_json(p/'inventory.json')['assets'][6:]
  for a in rows:
   await r.prepare(client,a)
   print('Prepared',a['obj_id'],flush=True)
 finally:await client.close()
if __name__=='__main__':asyncio.run(main())
