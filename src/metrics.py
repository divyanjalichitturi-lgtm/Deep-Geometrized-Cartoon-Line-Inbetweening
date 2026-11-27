import math
import numpy as np
from skimage.metrics import structural_similarity

def metrics(pred,target):
    p=np.clip(pred.astype(np.float32),0,1); t=np.clip(target.astype(np.float32),0,1)
    mse=float(np.mean((p-t)**2)); mae=float(np.mean(np.abs(p-t)))
    psnr=float('inf') if mse==0 else 10*math.log10(1.0/mse)
    ssim=float(structural_similarity(p,t,data_range=1.0))
    # Chamfer distance between nonzero line pixels.
    pb=(p>0.5).astype(np.uint8); tb=(t>0.5).astype(np.uint8)
    if pb.any() and tb.any():
        import cv2
        dt=np.where(tb>0,0,1).astype(np.uint8); dt=cv2.distanceTransform(dt,cv2.DIST_L2,3)
        dp=np.where(pb>0,0,1).astype(np.uint8); dp=cv2.distanceTransform(dp,cv2.DIST_L2,3)
        cd=float(dt[pb>0].mean()+dp[tb>0].mean())/2.0
    else: cd=None
    return {"MAE":mae,"MSE":mse,"PSNR_dB":psnr,"SSIM":ssim,"ChamferDistance":cd}
