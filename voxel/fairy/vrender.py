import numpy as np, math
from PIL import Image
def vrender(occ,col,az,el=0.0,S=3,bg=(250,244,236),outline=(60,40,34),size=None):
    """복셀 정사영 미리보기. az: 0=정면, 90=왼쪽 옆. el: 내려다보는 각도."""
    H,D,W=occ.shape; zz,yy,xx=np.nonzero(occ)
    cx,cy,cz=W/2,D/2,H/2
    ph=math.radians(az); e=math.radians(el)
    X=xx+0.5-cx; Y=yy+0.5-cy; Z=zz+0.5-cz
    u=-X*math.cos(ph)+Y*math.sin(ph)
    d=-X*math.sin(ph)-Y*math.cos(ph)          # 카메라 쪽 깊이
    v=Z*math.cos(e)-d*math.sin(e)
    dep=d*math.cos(e)+Z*math.sin(e)
    R=int(max(W,D,H)*0.9) if size is None else size
    n=2*R*S
    order=np.argsort(dep)
    img=np.zeros((n,n,3),np.uint8); img[:]=bg; msk=np.zeros((n,n),bool)
    ui=((u[order]+R)*S).astype(int); vi=((R-v[order])*S).astype(int); c=col[zz[order],yy[order],xx[order]]
    for du in range(S+1):
        for dv in range(S+1):
            a=np.clip(vi+dv,0,n-1); b=np.clip(ui+du,0,n-1); img[a,b]=c; msk[a,b]=True
    if outline is not None:
        e_=np.zeros_like(msk); e_[1:]|=msk[:-1]; e_[:-1]|=msk[1:]; e_[:,1:]|=msk[:,:-1]; e_[:,:-1]|=msk[:,1:]
        img[e_&~msk]=outline
    ys,xs=np.nonzero(msk)
    return Image.fromarray(img).crop((xs.min()-3,ys.min()-3,xs.max()+4,ys.max()+4))
