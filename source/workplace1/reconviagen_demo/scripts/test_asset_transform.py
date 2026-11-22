"""Check actual PLY roundtrip and projected covariance under a similarity."""
from pathlib import Path
import tempfile,json
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
from asset_adapter import transform,inspect,ROOT

def covariance(gs):
    q=gs.rot[0][:,[1,2,3,0]]
    r=Rotation.from_quat(q).as_matrix();s=gs.scale[0]
    return (r*s[:,None,:]**2)@r.transpose(0,2,1)

def project(mu,cov,eye,world_to_cam,focal=700.):
    m=(mu-eye)@world_to_cam.T
    c=world_to_cam[None]@cov@world_to_cam.T[None]
    uv=focal*m[:,:2]/m[:,2,None]
    j=np.zeros((len(mu),2,3));j[:,0,0]=focal/m[:,2];j[:,1,1]=focal/m[:,2]
    j[:,0,2]=-focal*m[:,0]/m[:,2]**2;j[:,1,2]=-focal*m[:,1]/m[:,2]**2
    return uv,j@c@j.transpose(0,2,1)

def main():
    rng=np.random.default_rng(314159);n=31
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    v=np.zeros(n,dtype=[(k,'f4') for k in fields]);mu=rng.normal(size=(n,3))*.3
    for i,k in enumerate(['x','y','z']):v[k]=mu[:,i]
    q=Rotation.random(n,random_state=rng).as_quat()[:,[3,0,1,2]]
    for i in range(4):v[f'rot_{i}']=q[:,i]
    scales=rng.uniform(.005,.04,(n,3))
    for i in range(3):v[f'scale_{i}']=np.log(scales[:,i]);v[f'f_dc_{i}']=rng.normal(size=n)
    v['opacity']=rng.normal(size=n)
    r=Rotation.from_euler('xyz',[27,-39,73],degrees=True).as_matrix();s=4.3;t=np.array([.7,-1.1,2.0])
    with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='transform_check_') as td:
        a=Path(td)/'source.ply';b=Path(td)/'transformed.ply';PlyData([PlyElement.describe(v,'vertex')]).write(str(a))
        transform(a,b,r,t,s);_,g0=inspect(a);_,g1=inspect(b)
        c0=covariance(g0);c1=covariance(g1);expected=s*s*(r[None]@c0@r.T[None])
        np.testing.assert_allclose(g1.xyz[0],s*g0.xyz[0]@r.T+t,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(c1,expected,rtol=3e-5,atol=1e-8)
        eye=np.array([0.,0.,-4.]);u0,p0=project(g0.xyz[0],c0,eye,np.eye(3))
        u1,p1=project(g1.xyz[0],c1,s*(r@eye)+t,r.T)
        np.testing.assert_allclose(u1,u0,rtol=1e-5,atol=3e-5)
        np.testing.assert_allclose(p1,p0,rtol=3e-5,atol=1e-5)
        np.testing.assert_array_equal(g0.opacity,g1.opacity);np.testing.assert_array_equal(g0.color,g1.color)
        result=dict(status='passed',kind='synthetic CPU serializer/projection check, not a generated asset or renderer test',
                    original_loader='curate.core.load_rigid',original_serializer='curate.stages._export_ply',
                    gaussian_count=n,max_covariance_error=float(abs(c1-expected).max()),
                    max_projection_pixel_error=float(abs(u1-u0).max()),max_projected_covariance_error=float(abs(p1-p0).max()))
        (ROOT/'docs/transform_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
