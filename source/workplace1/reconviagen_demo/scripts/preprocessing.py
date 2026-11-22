"""Official RGBA preprocessing, with explicit transforms and invalid-mask guards."""
import numpy as np
from PIL import Image

def preprocess_rgba(image, resolution=518):
    import torch
    import torch.nn.functional as F
    if image.mode!='RGBA':raise ValueError('An explicit foreground alpha is required')
    arr=np.array(image);alpha=arr[:,:,3]
    points=np.argwhere(alpha>.8*255)
    if len(points)<16:raise ValueError('Empty or degenerate foreground mask')
    if np.all(alpha==255):raise ValueError('All-opaque image does not establish foreground')
    xmin,ymin=points[:,1].min(),points[:,0].min();xmax,ymax=points[:,1].max(),points[:,0].max()
    center=[(xmin+xmax)/2,(ymin+ymax)/2];size=int(max(xmax-xmin,ymax-ymin)*1.1)
    box=[max(0,int(center[0]-size//2)),max(0,int(center[1]-size//2)),
         min(image.width,int(center[0]+size//2)),min(image.height,int(center[1]+size//2))]
    crop=image.crop(box);w,h=crop.size
    if min(w,h)<2:raise ValueError('Degenerate crop')
    side=max(w,h);px=(side-w)//2;py=(side-h)//2
    padded=Image.new('RGBA',(side,side),(0,0,0,0));padded.paste(crop,(px,py))
    tensor=torch.from_numpy(np.array(padded).astype(np.float32))/255
    resized=F.interpolate(tensor[None].permute(0,3,1,2),(resolution,resolution),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
    resized[:,:,:3]*=resized[:,:,3:4]>.8
    output=Image.fromarray((resized*255).astype(np.uint8),'RGBA')
    s=resolution/side
    # Pixel center convention for align_corners=False.
    A=[[s,0,s*(px-box[0])+(s-1)/2],[0,s,s*(py-box[1])+(s-1)/2],[0,0,1]]
    return output,dict(original_size=list(image.size),foreground_bbox=[int(xmin),int(ymin),int(xmax),int(ymax)],
        crop_box=box,pad_xy=[px,py],resize=[resolution,resolution],A_pixel_centers=A,
        clipped_foreground_at_border=bool(xmin==0 or ymin==0 or xmax==image.width-1 or ymax==image.height-1))

def selftest():
    import sys
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]/'vendor/ReconViaGen'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'wheels/vggt'))
    from trellis.pipelines import TrellisVGGTTo3DPipeline
    for dims,bbox in [((220,90),(30,10,190,80)),((95,210),(0,30,80,200)),((100,80),(5,2,98,65))]:
        a=np.zeros((dims[1],dims[0],4),np.uint8);x0,y0,x1,y1=bbox;a[y0:y1,x0:x1]=[210,40,90,255]
        im=Image.fromarray(a);result,record=preprocess_rgba(im)
        official=TrellisVGGTTo3DPipeline.preprocess_image(None,im)
        assert np.array_equal(np.array(result),np.array(official)),record
    for a in [np.zeros((20,40,4),np.uint8),np.full((20,40,4),255,np.uint8)]:
        try:preprocess_rgba(Image.fromarray(a))
        except ValueError:pass
        else:raise AssertionError('Invalid foreground accepted')
    print('PASS: bitwise equivalent official RGBA preprocessing; empty/opaque rejected')

if __name__=='__main__':selftest()
