"""Locate archived SAM2 source crops in original frames using SIFT geometry.

Candidate correspondence only; native Waymo association and visual review are
required before using it as object identity.
"""
from pathlib import Path
import cv2
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)

def match_case(row):
    scene = row['obj_id'].split('_')[2]
    source = Path(row['archive_root'])/row['source_copy']
    src = cv2.imread(str(source), 0)
    sift = cv2.SIFT_create(nfeatures=1600)
    kp, des = sift.detectAndCompute(src, None)
    frames = sorted((Path('/mnt/public_data/waymo/val')/scene/'images').glob('*.jpg'))
    results = []
    for path in frames:
        if int(path.stem.split('_')[0]) % 5: continue
        image = cv2.imread(str(path), 0)
        if image is None: continue
        ratio = 0.6
        small = cv2.resize(image, None, fx=ratio, fy=ratio)
        k2,d2 = sift.detectAndCompute(small,None)
        if d2 is None: continue
        matches=cv2.BFMatcher().knnMatch(des,d2,k=2)
        good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<0.72*b.distance]
        if len(good)<7: continue
        a=np.float32([kp[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good])/ratio
        H, mask=cv2.findHomography(a,b,cv2.RANSAC,4.0)
        if H is None or mask.sum()<7:continue
        corners=np.float32([[0,0],[src.shape[1],0],[src.shape[1],src.shape[0]],[0,src.shape[0]]]).reshape(1,4,2)
        projected=cv2.perspectiveTransform(corners,H)[0]
        if not np.isfinite(projected).all(): continue
        area=abs(cv2.contourArea(projected))
        if area<1000 or area>image.size*0.5:continue
        results.append(dict(path=str(path),inliers=int(mask.sum()),matches=len(good),
                            homography=H.tolist(),corners=projected.tolist()))
    results.sort(key=lambda x:x['inliers'],reverse=True)
    out=dict(candidate=row,source_path=str(source),source_keypoints=len(kp),matches=results[:30],
             status='candidate_needs_native_association_and_visual_review')
    (ROOT/'inputs'/f'{row["name"]}_source_matches.json').write_text(json.dumps(out,indent=2))
    print(row['name'],results[:1],flush=True)
    if results:
        best=cv2.imread(results[0]['path']);cv2.polylines(best,[np.int32(results[0]['corners'])],True,(0,255,0),4)
        cv2.imwrite(str(ROOT/'inputs'/f'{row["name"]}_source_match.jpg'),best)

if __name__=='__main__':
    rows=json.loads((ROOT/'docs/candidate_sources.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(match_case,rows))
