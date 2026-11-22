import os,time,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
while Path('/proc/668932').exists():time.sleep(3)
f=p/'control.json';d=json.loads(f.read_text());d.update(status='running',reason='User direct FLUX generation and final publication',updated=time.time());f.write_text(json.dumps(d,indent=2)+'\n')
print('Direct policy enabled; restarting managed supervisor',flush=True)
subprocess.run(['systemctl','--user','restart','rider-seated-resume-20261106.service'],check=True)
