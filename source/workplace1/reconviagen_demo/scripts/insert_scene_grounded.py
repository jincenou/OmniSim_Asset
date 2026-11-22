"""Explicit lower-road surface fit for the audited curbside trajectory.

Uses the original renderer and LiDAR+GS grid; removes high curb/vehicle grid
contamination before fitting the contact plane. No alternate path on failure.
"""
import numpy as np
import insert_scene as original

def road_contact(self,x,y,yaw,length=4.2,width=2.0):
    ix=int((x-self.x0)/self.CELL);iy=int((y-self.y0)/self.CELL)
    radius=int(np.ceil(max(length*.7,4.)/self.CELL))
    xlo=max(ix-radius,0);xhi=min(ix+radius+1,self.grid.shape[1])
    ylo=max(iy-radius,0);yhi=min(iy+radius+1,self.grid.shape[0])
    yy,xx=np.mgrid[ylo:yhi,xlo:xhi]
    P=np.stack([xx.ravel()*self.CELL+self.x0-x,yy.ravel()*self.CELL+self.y0-y,self.grid[yy,xx].ravel()],axis=-1)
    if len(P)<50 or not np.isfinite(P).all():raise ValueError('Insufficient local ground evidence')
    cutoff=np.quantile(P[:,2],.25)+.12
    keep=P[:,2]<=cutoff
    if keep.sum()<30:raise ValueError('Insufficient lower-road cells')
    A=np.column_stack([P[:,:2],np.ones(len(P))])
    a,b,d=np.linalg.lstsq(A[keep],P[keep,2],rcond=None)[0]
    if np.hypot(a,b)>np.tan(np.radians(self.MAX_TILT_DEG)):raise ValueError('Unreliable road plane slope')
    n=np.array([-a,-b,1.]);n/=np.linalg.norm(n)
    fwd=np.array([np.cos(yaw),np.sin(yaw),0.]);fwd-=fwd.dot(n)*n;fwd/=np.linalg.norm(fwd)
    return float(d),np.column_stack([fwd,np.cross(n,fwd),n])

original.GroundModel.contact_pose=road_contact
if __name__=='__main__':original.main()
