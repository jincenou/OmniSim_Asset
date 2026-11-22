"""Pure training-domain sampling rules. No renderer or heldout data access."""
import numpy as np

def normalize(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid sampling weights')
    return w/w.sum()

class Selector:
    def __init__(self,candidates,base_uids,arm,seed=42):
        assert arm in ['U','P','E','Q'];self.arm=arm;self.rows=candidates['observations'];self.base=list(base_uids)
        assert all(r['view'] in candidates['train_views'] and r['view'] not in [0,8,16] for r in self.rows)
        self.uid_to_index={r['uid']:r['index'] for r in self.rows};assert len(self.uid_to_index)==len(self.rows)
        assert all(u in self.uid_to_index for u in self.base)
        self.exposure=np.zeros(len(self.rows),np.int64);self.rng=np.random.default_rng(seed);self.cells={}
        for b in range(6):
            for k in range(8):
                ids=np.asarray([r['index'] for r in self.rows if r['bin']==k and r['areas'][b]>=64],dtype=int)
                if len(ids):self.cells[b,k]=ids
        self.table=None

    def update(self,table):
        assert table['domain']=='training_only' and set(table['views']).isdisjoint([0,8,16]);self.table=table

    def secondary(self,cell):
        ids=self.cells[cell];areas=np.asarray([self.rows[i]['areas'][cell[0]] for i in ids],float)
        return ids,normalize(np.minimum(areas/(np.median(areas)+1e-6),2)/np.sqrt(1+self.exposure[ids]))

    def choose(self,step):
        assert self.table is not None and 1<=step<=len(self.base)
        basic=self.uid_to_index[self.base[step-1]];cell=None;prob=1.;reason='uniform_base';index=basic
        if self.arm!='U' and step%2==1:
            if self.arm=='E':
                valid=[];weights=[]
                for r in self.rows:
                    d=self.table['full_difficulty'].get(f"{r['view']}:{r['bin']}")
                    if d is not None:valid.append(r['index']);weights.append(d+1e-3)
                if valid:
                    probs=normalize(weights);pos=int(self.rng.choice(len(valid),p=probs));index=valid[pos];prob=float(probs[pos]);reason='foreground_error_proxy'
                else:reason='empty_error_candidates_uniform'
            else:
                cells=list(self.cells)
                if self.arm=='Q':cells=[c for c in cells if self.table['demand'][c[0]][c[1]] is not None]
                if cells:
                    probs=normalize(np.ones(len(cells)) if self.arm=='P' else [self.table['demand'][b][k]+1e-3 for b,k in cells]);pos=int(self.rng.choice(len(cells),p=probs));cell=cells[pos]
                    ids,second=self.secondary(cell);j=int(self.rng.choice(len(ids),p=second));index=int(ids[j]);prob=float(probs[pos]*second[j]);reason='part_time_uniform' if self.arm=='P' else 'part_time_demand'
                else:reason='empty_part_candidates_uniform'
        self.exposure[index]+=1;r=self.rows[index]
        return dict(candidate_index=index,uid=r['uid'],view=r['view'],frame=r['frame'],bin=r['bin'],base_uid=self.base[step-1],selected_part=None if cell is None else cell[0],selected_bin=None if cell is None else cell[1],selection_path_probability=prob,probability_scope='cell-and-observation path,not marginal when observation belongs to multiple cells',selection_reason=reason,actual_exposure=int(self.exposure[index]))

    def state(self):return dict(exposure=self.exposure.tolist(),rng=self.rng.bit_generator.state,arm=self.arm)

    def restore(self,state):
        assert state['arm']==self.arm;self.exposure=np.asarray(state['exposure'],np.int64);self.rng.bit_generator.state=state['rng']
