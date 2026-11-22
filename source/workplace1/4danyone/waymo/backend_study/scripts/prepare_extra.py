"""Freeze additional existing baseline cases before any CEM heldout quality result."""
from pathlib import Path
from common import ROOT,manifest,save,sha,stamp
from train_case import configuration

def main():
    rows=[]
    for label in ['N01','N02','N03','N06','N07']:
        data,path=manifest(label)
        for item in data['input_hashes']:
            p=Path(data['source_data'])/'images'/f"cam{item['view']:02d}"/f"{item['frame']:06d}.png"
            assert sha(p)==item['sha256'],str(p)
        for stage in [1,2]:save(ROOT/'configs'/f'{label}_stage{stage}.json',configuration(label,stage)[-1])
        rows.append({'label':label,'manifest_sha256':sha(path),'images_checked':len(data['input_hashes']),
            'frame_count':data['frame_count'],'train_views':data['train_views'],'heldout_views':data['heldout_views']})
        print(label,'input hashes and both stage configurations checked',flush=True)
    save(ROOT/'research/extra_inputs.json',{'state':'passed','checked_at':stamp(),'cases':rows,
        'selection':'All other completed original Random8 baselines; no CEM heldout metrics seen at selection'})

if __name__=='__main__':main()
