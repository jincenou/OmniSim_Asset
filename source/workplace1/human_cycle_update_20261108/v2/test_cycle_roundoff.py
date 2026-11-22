import numpy as np
from hasim.shared.assets.human_cycle import CyclicRuntimeAsset, CycleProfile

def test_transition_weights_remain_in_unit_interval():
    obj=object.__new__(CyclicRuntimeAsset)
    obj.profile=CycleProfile("synthetic",.55,2.55,.2,4.,"0"*64,"stationary",qualified=True)
    for n in (0,1,17,100):
        for offset in np.linspace(-1e-7,1e-7,101):
            _,_,weight=obj.source_samples(n*obj.period_s+.2+offset)
            assert 0 <= weight <= 1
