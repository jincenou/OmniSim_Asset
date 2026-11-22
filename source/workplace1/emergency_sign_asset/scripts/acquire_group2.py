from acquire import acquire
import concurrent.futures
sources=[('V03','https://www.etrailer.com/tv-review-erickson-emergency-warning-triangles-em05310.aspx','https://www.etrailer.com/static/images/video/review-erickson-emergency-warning-triangles-em05310_web.mp4','etrailer'),('V05','https://www.youtube.com/watch?v=oxTTrZzDn0I',None,'The Traffic Safety Store'),('V07','https://www.youtube.com/watch?v=UQSeO-NCwIs',None,'The Cortina Companies'),('V08','https://www.youtube.com/watch?v=hLfh_kOoNLY',None,'The Cortina Companies')]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(acquire,sources))
