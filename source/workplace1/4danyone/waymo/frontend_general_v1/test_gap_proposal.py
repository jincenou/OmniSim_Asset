import numpy as np
from scipy.spatial.transform import Rotation
from gap_proposal import bounded_rotation_gaps

def main():
    t=np.arange(31)/30;p=np.zeros((31,2,3));p[:,0,2]=np.deg2rad(np.linspace(170,190,31))
    valid=np.ones((31,2),dtype=bool);valid[8:23,0]=False;valid[:5,1]=False
    corrupt=p.copy();corrupt[8:23,0]=[1.,1.,1.]
    q,changed,records=bounded_rotation_gaps(corrupt,t,valid,1.)
    error=(Rotation.from_rotvec(q[:,0])*Rotation.from_rotvec(p[:,0]).inv()).magnitude().max()
    assert error<1e-10 and changed[:,0].sum()==15 and not changed[:,1].any()
    np.testing.assert_array_equal(q[valid],corrupt[valid])
    # No fake boundary observation and no all-joint smoothing.
    np.testing.assert_array_equal(q[:5,1],corrupt[:5,1])
    q2,c2,_=bounded_rotation_gaps(corrupt,t,valid,.2)
    assert not c2.any();np.testing.assert_array_equal(q2,corrupt)
    print('SO(3) branch-crossing, observed-anchor preservation, unknown-boundary and long-gap tests passed; no real-motion quality claim.')
if __name__=='__main__':main()
