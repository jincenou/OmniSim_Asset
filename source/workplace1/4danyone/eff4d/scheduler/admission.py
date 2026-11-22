import argparse,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--min-free-mib',type=int,required=True);a=p.parse_args();assert a.gpu in [4,5,6,7]
assert shutil.disk_usage('/mnt').free>=35*1024**3,'Disk reserve below35GiB'
free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=a.min_free_mib,f'Insufficient current free GPU memory: {free}'
print(f'Admission GPU{a.gpu}: free {free}MiB; snapshot only, not a reservation',flush=True)
