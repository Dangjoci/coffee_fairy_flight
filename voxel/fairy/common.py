import numpy as np, math, pickle
from PIL import Image
from scipy import ndimage as ndi
ANG=[0,45,90,135,180,225,270,315]
def load_views():
    V=[]
    for k in range(8):
        c=np.asarray(Image.open(f'view{k}.png')).copy()
        rgb=c[...,:3].astype(int); m=c[...,3]>0
        sat=rgb.max(2)-rgb.min(2); h=m.shape[0]
        sh=np.zeros_like(m); sh[h-30:]=True
        m&=~(sh&(sat<45)&(rgb.min(2)>150))                 # 발밑 그림자 제거
        m=ndi.binary_opening(m,iterations=1)
        lab,n=ndi.label(m); 
        if n>1:
            sz=ndi.sum(m,lab,range(1,n+1)); m=lab==1+int(np.argmax(sz))
        rows=np.nonzero(m.any(1))[0]; top,bot=rows[0],rows[-1]
        # 축: 머리 윗부분(꽃 위, 잔 아래)의 가운데
        r0=top+int((bot-top)*0.13); r1=top+int((bot-top)*0.17)
        cs=[]
        for r in range(r0,r1):
            xs=np.nonzero(m[r])[0]; cs.append((xs[0]+xs[-1])/2)
        V.append(dict(rgb=c[...,:3],m=m,top=top,bot=bot,ax=float(np.median(cs))))
    return V
def proj(V,k,x,y,z,cx,cy,H):
    """복셀 좌표 -> 뷰 k의 픽셀 좌표(u,v)."""
    v=V[k]; ph=math.radians(ANG[k]); s=(v['bot']-v['top'])/H
    u=(-(x-cx)*math.cos(ph)+(y-cy)*math.sin(ph))*s+v['ax']
    w=v['bot']-(z)*s
    return u,w
def silhouette(V,k,xx,yy,zz,cx,cy,H):
    m=V[k]['m']; s=(V[k]['bot']-V[k]['top'])/H; r=int(math.ceil(s/2))+1
    u,w=proj(V,k,xx+0.5,yy+0.5,zz+0.5,cx,cy,H)
    sil=np.zeros(m.shape,bool)
    ui=np.rint(u).astype(int); wi=np.rint(w).astype(int)
    ok=(ui>=0)&(ui<m.shape[1])&(wi>=0)&(wi<m.shape[0]); sil[wi[ok],ui[ok]]=True
    return ndi.binary_dilation(sil,structure=np.ones((2*r+1,2*r+1),bool))
