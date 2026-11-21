"""서 있는 모델을 비행 자세로: 팔은 앞으로, 발은 뒤로 접고, 몸을 목 기준 35도 눕힌다. 날개는 3프레임(위·가운데·아래)."""
import numpy as np, pickle, math
from scipy import ndimage as ndi
SC=1.5
d=pickle.load(open('standing3.pkl','rb'))
occ0,wing0,col0,lab0,names=d['occ'],d['wing'],d['col'],d['lab'],d['names']
PADB,PADX=34,10                                    # 뒤쪽·좌우 여유
def pad(a):
    w=[(0,0),(0,PADB),(PADX,PADX)]+([(0,0)] if a.ndim==4 else [])
    return np.pad(a,w)
occ0,wing0,col0,lab0=pad(occ0),pad(wing0),pad(col0),pad(lab0)
H,D,W=occ0.shape; cx=d['cx']+PADX; cy=d['cy']
Z,Y,X=np.mgrid[0:H,0:D,0:W].astype(float)
BE=names.index('BEARD')+1
def rot_x(mask,piv,deg,occ,col,lab):
    """mask 부분을 x축 기준(piv=(y,z))으로 deg만큼 돌려 옮긴다. +면 아래쪽이 뒤로 간다."""
    a=math.radians(deg); dy,dz=Y-piv[0],Z-piv[1]
    sy=dy*math.cos(a)+dz*math.sin(a)+piv[0]; sz=-dy*math.sin(a)+dz*math.cos(a)+piv[1]
    iy,iz,ix=np.rint(sy).astype(int),np.rint(sz).astype(int),X.astype(int)
    ok=(iy>=0)&(iy<D)&(iz>=0)&(iz<H); src=np.zeros_like(mask); src[ok]=mask[iz[ok],iy[ok],ix[ok]]
    o2=occ&~mask; c2=col.copy(); c2[mask]=0; l2=np.where(mask,0,lab)
    put=src&~o2
    c2[put]=col[iz[put],iy[put],ix[put]]; l2[put]=lab[iz[put],iy[put],ix[put]]
    return o2|put,c2,l2
occ,col,lab=occ0.copy(),col0.copy(),lab0.copy()
head=(np.abs((X-cx)/(18.8*SC))**2.2+np.abs((Y-cy)/(16.3*SC))**2.2+np.abs((Z-39*SC)/(16.2*SC))**2.2<=1)
brown=(col[...,0]<150)&(col[...,0]>col[...,2]+20)&(Z>16*SC)&(np.abs(X-cx)<6*SC)&(Y<cy-8*SC)   # 정면에서 찍어 온 수염
beard=occ&((lab==BE)|brown)&~((lab==names.index('SHORT')+1))
beard=ndi.binary_dilation(beard,structure=np.ones((1,3,1),bool))&occ|beard
# 1) 팔을 앞으로
arms=occ&(np.abs(X-cx)>12.3*SC)&(Z>13*SC)&(Z<26.5*SC)&~head
occ,col,lab=rot_x(arms,(cy-1*SC,23.5*SC),-58,occ,col,lab)
# 2) 발을 뒤로 접기
feet=occ&(Z<4.6*SC)
occ,col,lab=rot_x(feet,(cy+1*SC,6.5*SC),62,occ,col,lab)
# 3) 몸을 눕히기(머리·수염·꽃·잔은 그대로)
body=occ&~head&~beard&(Z<27*SC)
occ,col,lab=rot_x(body,(cy+2*SC,26*SC),35,occ,col,lab)
# 수염을 두껍게(뒤로 2칸) — 가슴이 물러나 홀로 늘어지므로
bz,by,bx=np.nonzero(beard&occ)
for t in (1,2):
    yy=np.clip(by+t,0,D-1); free=~occ[bz,yy,bx]
    occ[bz[free],yy[free],bx[free]]=True; col[bz[free],yy[free],bx[free]]=col[bz[free],by[free],bx[free]]
l_,n=ndi.label(occ); sz=ndi.sum(occ,l_,range(1,n+1)); occ&=np.isin(l_,[i+1 for i,s in enumerate(sz) if s>60])
# 4) 날개: 뿌리를 눕힌 등 위로 옮기고, 뿌리 축(앞뒤 방향)으로 올리고 내린다
ROOT=(cy+11.5*SC,25.5*SC)                         # (y,z)
wsrc_root=(cy+9.0*SC,22.0*SC)
def wings(deg):
    o=np.zeros_like(occ); c=np.zeros_like(col)
    for side in (-1,1):
        a=math.radians(deg)*side
        dx,dz=X-cx,Z-ROOT[1]
        sx=dx*math.cos(a)+dz*math.sin(a)+cx; sz=-dx*math.sin(a)+dz*math.cos(a)+wsrc_root[1]; sy=Y-ROOT[0]+wsrc_root[0]
        ix,iy,iz=np.rint(sx).astype(int),np.rint(sy).astype(int),np.rint(sz).astype(int)
        ok=(ix>=0)&(ix<W)&(iy>=0)&(iy<D)&(iz>=0)&(iz<H)&((X-cx)*side>0)
        ok2=ok.copy(); ok2[ok]=wing0[iz[ok],iy[ok],ix[ok]]&((ix[ok]-cx)*side>0)
        o|=ok2; c[ok2]=col0[iz[ok2],iy[ok2],ix[ok2]]
    o&=~occ
    return o,c
W3=[wings(a) for a in (24,8,-12)]
pickle.dump(dict(occ=occ,col=col,wings=W3,cx=cx,cy=cy,SC=SC),open('flight.pkl','wb'))
# ── 내보내기: 모델마다 꽉 차게 잘라 층을 격자(아틀라스)로 깐 PNG + 메타
import json
from PIL import Image
OUT='..'
ORG=(cx,cy+6*SC,30*SC)                              # 게임에서 요정 위치가 될 점(몸 한가운데)
FV=0.123                                            # 복셀 한 칸의 게임 단위 길이
meta={}
def export(name,o,c,extra=None):
    zz,yy,xx=np.nonzero(o); z0,z1,y0,y1,x0,x1=zz.min(),zz.max()+1,yy.min(),yy.max()+1,xx.min(),xx.max()+1
    o=o[z0:z1,y0:y1,x0:x1]; c=c[z0:z1,y0:y1,x0:x1]; h,d_,w=o.shape; cols=int(math.ceil(math.sqrt(h)))
    rows=int(math.ceil(h/cols)); im=np.zeros((rows*d_,cols*w,4),np.uint8)
    for z in range(h):
        r,k=divmod(z,cols); im[r*d_:(r+1)*d_,k*w:(k+1)*w,:3]=c[z]; im[r*d_:(r+1)*d_,k*w:(k+1)*w,3]=o[z]*255
    Image.fromarray(im,'RGBA').save(f'{OUT}/{name}_strip.png')
    meta[name]=dict(w=int(w),d=int(d_),h=int(h),cols=cols,vox=FV,o=[round(float(ORG[0]-x0),2),round(float(ORG[1]-y0),2),round(float(ORG[2]-z0),2)],**(extra or {}))
    print(name,o.shape,im.shape)
wz,wy,wx=np.nonzero(W3[1][0]); i=int(np.argmax(wx)); tip=[float(wx[i]-ORG[0]),float(wy[i]-ORG[1]),float(wz[i]-ORG[2])]
SHc=names.index('SHORT')+1
sz_,sy_,sx_=np.nonzero(occ&(lab==SHc)); j=int(np.argmax(sy_)); butt=[0.0,float(sy_[j]-ORG[1]),float(sz_[sy_==sy_[j]].mean()-ORG[2])]
export('fairy_body',occ,col,dict(tip=[round(t,1) for t in tip],butt=[round(t,1) for t in butt]))
for k,(o,c) in enumerate(W3): export(f'fairy_wing{k}',o,c)
json.dump(meta,open(f'{OUT}/fairy_meta.json','w'),indent=1)
print(json.dumps(meta)[:400])
print('grid',occ.shape,'body',occ.sum(),'wings',[int(w[0].sum()) for w in W3])
