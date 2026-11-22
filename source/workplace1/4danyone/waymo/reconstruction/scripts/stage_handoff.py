"""One-time scheduler handoff at a subprocess boundary, before GPU work."""
import json
import os
from pathlib import Path
import signal
import time

B=Path(__file__).resolve().parents[2]/'batch100'

def gate():
    control=B/'handoff.json'
    if not control.exists():return
    request=json.loads(control.read_text())
    if request['state']!='armed' or os.getppid()!=request['old_controller_pid']:return
    # The old scheduler awaits this child. Detach the child from its terminal
    # so ending the old scheduler cannot terminate a stage doing real work.
    os.setsid()
    marker=B/'handoff_waiters'/f'{os.getpid()}.json'
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps({'pid':os.getpid(),'parent':os.getppid(),'state':'before_stage'})+'\n')
    while os.getppid()==request['old_controller_pid']:time.sleep(.2)
    # No stage ran, so do not produce a success marker for it.
    os._exit(75)

gate()
