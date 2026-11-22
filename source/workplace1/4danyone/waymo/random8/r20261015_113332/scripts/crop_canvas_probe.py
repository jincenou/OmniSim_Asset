"""Verify that source pixels are lost by an artificial rectified-canvas clamp."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
R=Path(__file__).resolve().parents[1];cv2.setNumThreads(2)
def main():
 results=[]
 for label in ['N02','N07','N04']:
  audit=R/'camera_audit'/label;rows=json.loads((audit/'audit.json').read_text())['records'];z=np.load(audit/'native_cameras.npz');K=z['K'];dist=z['distortion'];boxes=[]
  for row in rows:
   x0,y0,x1,y1=row['bbox_xyxy_distorted'];x,y=np.meshgrid(np.linspace(x0,x1,17),np.linspace(y0,y1,17));uv=cv2.undistortPointsIter(np.stack([x,y],-1).reshape(-1,1,2),K,dist,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-10)).reshape(-1,2);boxes.append(np.r_[uv.min(0),uv.max(0)])
  boxes=np.asarray(boxes);extent=(boxes[:,2:]-boxes[:,:2]).max(0)*1.3;unit=int(np.ceil(max(extent[0]/11,extent[1]/20)/2)*2);width,height=11*unit,20*unit;assert width<=1920 and height<=1280;clipped=[];errors=[];hash_equal=[];all_inside=True
  for i,(row,b) in enumerate(zip(rows,boxes)):
   center=np.rint((b[:2]+b[2:]-[width,height])/2).astype(int);old=np.maximum(0,np.minimum(center,[1920-width,1280-height]));oldb=b-np.tile(old,2);inside=(oldb[0]>=0 and oldb[1]>=0 and oldb[2]<=width and oldb[3]<=height);lo=old if inside else center
   if not inside:clipped.append({'frame':row['frame_index'],'old_lo':old.tolist(),'correct_lo':lo.tolist(),'bbox_undistorted':b.tolist()})
   newb=b-np.tile(lo,2);all_inside &= bool(newb[0]>=0 and newb[1]>=0 and newb[2]<=width and newb[3]<=height);A=np.eye(3);A[:2,2]=-lo;maps=cv2.initUndistortRectifyMap(K,dist,np.eye(3),A@K,(width,height),cv2.CV_32FC1)
   if label=='N04':
    rgb=np.asarray(Image.open(row['source_path']).convert('RGB'));current=cv2.remap(rgb,*maps,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);original=np.asarray(Image.open(R/'samples/N04/source/native'/f'{i:05d}.png'));errors.append(int(abs(current.astype(int)-original.astype(int)).max()));hash_equal.append(bool(np.array_equal(current,original)))
  result={'label':label,'old_clipped_frames':len(clipped),'clipping':clipped,'all_new_boxes_inside':all_inside,'size':[width,height],'N04_existing_native_max_pixel_difference':max(errors) if errors else None,'N04_all_native_exact_equal':all(hash_equal) if hash_equal else None};results.append(result);print({k:v for k,v in result.items() if k!='clipping'},flush=True)
 (R/'research/crop_canvas_probe.json').write_text(json.dumps({'state':'probe_complete_no_resize_no_threshold_change','change':'Direct remap to A@K crop; retain old integer origin whenever it includes the undistorted target; otherwise allow translated crop past the arbitrary1920x1280 rectified canvas. Original raw sensor remains unchanged.','results':results},indent=2)+'\n')
if __name__=='__main__':main()
