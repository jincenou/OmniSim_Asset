"""Independent native checkpoint replay after the short training, before formal runs."""
import argparse,json
import torch
from common import ROOT,save,stamp,device
from evaluate_case import CEM

@torch.inference_mode()
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--label',required=True);args=cli.parse_args()
    gpu=device();root=ROOT/'smoke'/args.label/'stage1'
    complete=json.loads((root/'complete.json').read_text())
    assert complete['kind']=='smoke_not_quality' and complete['updates_this_stage']==1199
    model=CEM(args.label,1,root=root);model.validate()
    assert len(model.replay)==9
    result={'state':'passed','device':gpu,'checked_at':stamp(),'replay':model.replay,
        'motion_groups':len(model.model._xyz_motion),
        'scope':'short-run native checkpoint replay only; no quality metric selection'}
    save(root/'independent_replay.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
