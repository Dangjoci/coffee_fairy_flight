"""커피의 요정 서 있는 모델(1.5배 해상도).
형태: 기존 치수 모델을 부풀려 원화 8뷰 실루엣으로 깎는다.
색: 부위별 4단계 색(원화에서 고른 값)을 법선·그늘로 나눠 칠하고, 얼굴·수염·허리끈은 정면 그림에서 찍어 온다. 날개는 뒷모습 그림에서 떼어 세운다."""
import numpy as np, pickle, math
from scipy import ndimage as ndi
from common import *
SC=1.5
g0,pal0=pickle.load(open('prior.pkl','rb')); pal0=[tuple(c[:3]) for c in pal0]
NAME={(247,198,152):'SKIN',(238,172,112):'FOOT',(96,60,44):'SHORT',(222,170,110):'TIE',(112,64,38):'BEARD',(66,40,30):'SKIN',
      (248,242,230):'CUP',(84,50,30):'COFFEE',(238,104,150):'PINK',(250,160,190):'PINK',(250,205,80):'YEL',(92,150,68):'GREEN',
      (246,243,238):'WING',(224,198,150):'WING',(236,222,196):'WING',(246,150,142):'SKIN',(232,150,130):'SKIN',(232,176,130):'SKIN'}
RAMP=dict(  # 밝은 쪽 -> 어두운 쪽. 밝을수록 노랗게, 어두울수록 붉은 보라로 색상을 튼다
 SKIN=[(255,231,190),(250,205,160),(236,170,134),(206,132,118)],
 FOOT=[(255,206,140),(242,174,110),(220,138,92),(184,104,88)],
 SHORT=[(138,92,62),(104,64,46),(80,44,42),(58,32,40)],
 TIE=[(246,206,138),(226,172,110),(196,134,88),(160,102,80)],
 BEARD=[(158,100,56),(124,70,42),(96,48,38),(70,34,36)],
 CUP=[(255,253,244),(244,236,222),(222,208,200),(190,176,184)],
 COFFEE=[(120,76,44),(90,54,32),(70,38,30),(54,28,30)],
 PINK=[(255,188,198),(246,126,164),(224,84,146),(180,58,124)],
 YEL=[(255,236,132),(250,206,82),(232,164,56),(198,128,52)],
 GREEN=[(150,196,88),(100,156,68),(66,124,66),(44,92,66)])
DARK=(70,42,30); BLUSH=(250,158,150); BLUSH2=(252,182,166)
LAB=np.array([0]+[list(RAMP).index(NAME[c])+1 if NAME[c]!='WING' else 99 for c in pal0[1:]])
lab0=LAB[g0]
body0=(lab0>0)&(lab0!=99)
occP=ndi.zoom(body0.astype(float),SC,order=1)>0.5
labP=ndi.zoom(np.where(body0,lab0,0),SC,order=0)
H,D,W=occP.shape; cx,cy=32.5*SC,28.5*SC; HT=61.8*SC
V=load_views(); PX=[(v['bot']-v['top'])/HT for v in V]
# ── 형태: 2칸 부풀려 8개 실루엣으로 깎기
infl=ndi.binary_dilation(occP,iterations=2)
zz,yy,xx=np.nonzero(infl); keep=np.ones(len(zz),bool)
for k in range(8):
    m=ndi.binary_dilation(V[k]['m'],iterations=1)
    u,w=proj(V,k,xx+0.5,yy+0.5,zz+0.5,cx,cy,HT); ui=np.rint(u).astype(int); wi=np.rint(w).astype(int)
    ok=(ui>=0)&(ui<m.shape[1])&(wi>=0)&(wi<m.shape[0]); ins=np.zeros(len(zz),bool); ins[ok]=m[wi[ok],ui[ok]]; keep&=ins
occ=np.zeros_like(infl); occ[zz[keep],yy[keep],xx[keep]]=True
occ=ndi.binary_closing(ndi.binary_opening(occ,iterations=1),iterations=1)
l_,n=ndi.label(occ); occ=l_==1+int(np.argmax(ndi.sum(occ,l_,range(1,n+1))))
# ── 부위: 가장 가까운 기존 부위를 물려받는다
idx=ndi.distance_transform_edt(labP==0,return_distances=False,return_indices=True)
lab=np.where(occ,labP[tuple(idx)],0)
# ── 명암 단계: 위·앞 왼쪽에서 오는 빛 + 파묻힌 정도
sm=ndi.gaussian_filter(occ.astype(float),1.8); gz,gy,gx=np.gradient(sm)
nl=np.sqrt(gx**2+gy**2+gz**2)+1e-9
Ldir=np.array([-0.32,-0.05,0.95]); Ldir/=np.linalg.norm(Ldir)        # (x,y,z): 왼쪽·앞·위
lam=-(gx*Ldir[0]+gy*Ldir[1]+gz*Ldir[2])/nl
ao=ndi.gaussian_filter(occ.astype(float),5.0)                         # 클수록 주변이 막혀 있음
val=0.74*(lam*0.5+0.5)+0.26*(1.0-np.clip((ao-0.25)/0.45,0,1))
X_,Y_,Z_=np.meshgrid(np.arange(W),np.arange(D),np.arange(H),indexing='xy'); X_=X_.transpose(2,0,1); Y_=Y_.transpose(2,0,1); Z_=Z_.transpose(2,0,1)
dith=(((X_+Y_+Z_)%2)-0.5)*0.03                                        # 단계 경계의 바둑판 디더
tone=np.clip(np.digitize(val+dith,[0.36,0.50,0.70]),0,3)             # 0=가장 어두움 … 3=가장 밝음
tone=3-tone
col=np.zeros(occ.shape+(3,),np.uint8)
names=list(RAMP)
for i,nm in enumerate(names):
    r=np.array(RAMP[nm],np.uint8); m=lab==i+1; col[m]=r[tone[m]]
# ── 얼굴·수염·허리끈을 복셀 크기에 맞춰 직접 그린다(위치는 정면 원화에서 잰 값, 선 모양은 2D 치비 시트 기준)
SK,SH,BE=names.index('SKIN')+1,names.index('SHORT')+1,names.index('BEARD')+1
first=np.argmax(occ,axis=1); anyc=occ.any(axis=1)
def dot(xf,zf,c,dep=2):
    """정면에서 본 (x,z) 자리의 가장 앞 복셀에 색을 놓는다."""
    x,z=int(round(cx-0.5+xf)),int(round(zf))
    if 0<=x<W and 0<=z<H and anyc[z,x]:
        y=first[z,x]; col[z,y:y+dep,x]=c
# 기존 수염 덩어리는 일단 살색으로 되돌린다(수염은 아래에서 새로 그림)
col[lab==BE]=np.array(RAMP['SKIN'],np.uint8)[tone[lab==BE]]
for s_ in (-1,1):
    for t in np.linspace(-1,1,25):                         # 웃는 눈: 위로 굽은 호, 두께 2
        ex=s_*(9.6+t*4.2); ez=60.6+2.4*(1-t*t)
        dot(ex,ez,DARK); dot(ex,ez-1,DARK)
    for t in np.linspace(-1,1,15):                         # 눈썹: 짧고 얇은 호
        dot(s_*(9.8+t*2.6),68.0+0.9*(1-t*t),DARK,1)
    for dx in np.arange(-3.0,3.01,0.5):                    # 볼터치: 꽉 찬 타원 + 밝은 점
        for dz in np.arange(-1.8,1.81,0.5):
            if (dx/3.0)**2+(dz/1.8)**2<=1: dot(s_*17.6+dx,54.6+dz,BLUSH,1)
    dot(s_*17.0,55.4,(255,206,196),1); dot(s_*17.0+1,55.4,(255,206,196),1)
    for t in np.linspace(0,1,22):                          # 콧수염: 가운데서 내려갔다가 끝이 말려 올라감
        mx=s_*(0.6+t*8.6); mz=53.6-2.0*math.sin(t*math.pi*0.9)+2.6*t*t
        dot(mx,mz,DARK,3); dot(mx,mz-1,DARK,3)
    dot(s_*9.6,56.4,DARK,3)                                # 말린 끝
dot(0,56.0,(232,150,132),1); dot(-1,56.0,(232,150,132),1)             # 코
# 턱수염: 콧수염 아래에서 시작해 가슴까지, 아래로 갈수록 좁게. 밝은 갈색 바탕에 결과 테두리
bd=np.array(RAMP['BEARD'],np.uint8)
ZT,ZB=51.5,29.0
for z in range(int(ZB),int(ZT)+1):
    f=(z-ZB)/(ZT-ZB); half=0.6+4.6*min(1.0,f*1.25)**0.8
    y0=None
    for xo in np.arange(-half,half+0.01,1.0):
        x=int(round(cx-0.5+xo)); 
        if not (0<=x<W): continue
        ys=np.nonzero(occ[z,:,x])[0]
        yf=min(ys[0] if len(ys) else D, int(cy-14.6*SC))          # 가슴 앞으로 떨어져 늘어짐
        edge=abs(xo)>half-1.0
        strand=(int(round(xo))%3==0)
        c=bd[2] if edge else (bd[0] if strand else bd[1])
        occ[z,yf-2:yf+1,x]=True; col[z,yf-2:yf+1,x]=c; lab[z,yf-2:yf+1,x]=BE
# 허리끈: 리본 매듭과 늘어진 두 가닥
T1,T2=RAMP['TIE'][0],RAMP['TIE'][2]
for (dx,dz) in ((-3,20),(-2,21),(-1,20),(-2,19),(1,20),(2,21),(3,20),(2,19)): dot(dx,dz,T1)       # 고리 두 개
dot(0,20,T2); dot(-0.5,20,T2)
for k in range(5): dot(-1.5-k*0.2,18-k,T1); dot(1.0+k*0.2,18-k,T1)
# 가슴 점, 배꼽
for s_ in (-1,1): dot(s_*8.6,35.5,(232,150,132),1)
dot(0,24.5,RAMP['SKIN'][3],1); dot(-1,24.5,RAMP['SKIN'][3],1)
# 반바지 허리단(밝은 줄)과 뒤 솔기(어두운 줄)
ys_,zs_,xs_=None,None,None
er=ndi.binary_erosion(occ); surf=occ&~er
wb=surf&(lab==SH)&(Z_>=19.5)&(Z_<=20.6)
col[wb]=RAMP['SHORT'][0]
seam=surf&(lab==SH)&(np.abs(X_-(cx-0.5))<=0.6)&(Y_>cy)&(Z_<19)
col[seam]=RAMP['SHORT'][3]
# 꽃: 꽃잎 5장의 경계 골과 가운데 진한 색, 잎의 가운데 맥
PK,YL,GR=names.index('PINK')+1,names.index('YEL')+1,names.index('GREEN')+1
for s_ in (-1,1):
    a18=math.radians(18); n_=np.array([s_*math.cos(a18),-math.sin(a18),0.0]); e1=np.array([s_*math.sin(a18),math.cos(a18),0.0])
    c0=np.array([cx-0.5+0.0,cy-0.5,46.5*SC])+n_*17.6*SC
    px_=(X_-c0[0])*e1[0]+(Y_-c0[1])*e1[1]; pz_=Z_-c0[2]
    rr=np.sqrt(px_**2+pz_**2); ang=np.degrees(np.arctan2(pz_,px_))
    side=((X_-(cx-0.5))*s_>0)
    fm=surf&(lab==PK)&side
    groove=(np.abs(((ang-90+36)%72)-36)>31)&(rr>2.2)
    col[fm&groove]=RAMP['PINK'][2]
    col[fm&(rr<=3.0)]=RAMP['PINK'][3]
    col[fm&(rr>6.4)&~groove]=RAMP['PINK'][0]
    col[surf&(lab==YL)&side]=RAMP['YEL'][1]
    lm=surf&(lab==GR)&side
    for a_ in (20,140,260):
        d_=np.abs(-(px_)*math.sin(math.radians(a_))+pz_*math.cos(math.radians(a_)))
        along=px_*math.cos(math.radians(a_))+pz_*math.sin(math.radians(a_))
        col[lm&(d_<0.7)&(along>0)]=RAMP['GREEN'][0]
# 잔: 테두리는 가장 밝게, 받침 아래는 그늘
CU=names.index('CUP')+1
col[surf&(lab==CU)&(Z_>=H-4)]=RAMP['CUP'][0]
# ── 날개: 뒷모습 그림의 흰 날개를 떼어 20도 젖힌 판으로 세운다(3단계 색)
v4=V[4]; rgb=v4['rgb'].astype(int); sat=rgb.max(2)-rgb.min(2); wingpx=v4['m']&(sat<60)&(rgb.min(2)>150)
wacc=np.zeros(occ.shape+(3,)); wn=np.zeros(occ.shape); SW=math.tan(math.radians(20)); ROOTY=cy+9.0*SC
ys,xs=np.nonzero(wingpx)
for py,pxx in zip(ys,xs):
    z=(v4['bot']-py)/PX[4]; x=(pxx-v4['ax'])/PX[4]+cx
    if z>36*SC or z<6*SC or abs(x-cx)<3.5*SC: continue
    y=ROOTY+abs(x-cx)*SW; zi,xi=int(z),int(x)
    for t in range(3):
        yi=int(y)+t
        if 0<=zi<H and 0<=yi<D and 0<=xi<W: wacc[zi,yi,xi]+=rgb[py,pxx]; wn[zi,yi,xi]+=1
wing=(wn>0)&~occ
wl,nw=ndi.label(wing); szs=ndi.sum(wing,wl,range(1,nw+1)); wing=np.isin(wl,[i+1 for i,s_ in enumerate(szs) if s_>200])
wm=(wacc/np.maximum(wn,1)[...,None]).mean(-1)
WT=np.array([(222,198,152),(238,228,210),(251,248,243)],np.uint8)
q=np.quantile(wm[wing],[0.2,0.5]); col[wing]=WT[np.digitize(wm[wing],q)]
print('voxels',occ.sum(),'wing',wing.sum(),'grid',occ.shape)
pickle.dump(dict(occ=occ,wing=wing,col=col,lab=lab,cx=cx,cy=cy,HT=HT,names=names),open('standing3.pkl','wb'))
