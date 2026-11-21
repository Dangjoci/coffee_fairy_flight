"""향기 순찰 — 게임 요소 복셀 모델. 층 아틀라스(왼쪽 위가 맨 아래층, 가로 cols장씩)와 메타데이터를 만든다.
좌표: x=오른쪽, y=뒤(작을수록 앞), z=위."""
import math, json, sys
import numpy as np
from stackpix import *

META = {}

import colorsys
from scipy import ndimage as ndi

def ramp(rgb):
    """바탕색 하나에서 4단계 색을 만든다. 밝은 쪽은 노랗게, 어두운 쪽은 붉은 보라로 색상을 튼다."""
    h, sat, val = colorsys.rgb_to_hsv(*[c / 255 for c in rgb])
    def tw(h, target, amt):                                              # 색상을 target 쪽으로 amt만큼
        d = ((target - h + 0.5) % 1.0) - 0.5
        return (h + max(-amt, min(amt, d))) % 1.0
    out = []
    for dv, ds, tgt, amt in ((1.14, 0.84, 1 / 6, 0.025), (1.0, 1.0, h, 0), (0.83, 1.08, 0.86, 0.03), (0.66, 1.14, 0.86, 0.06)):
        hh = tw(h, tgt, amt) if sat > 0.08 else h
        r, g, b = colorsys.hsv_to_rgb(hh, min(1, sat * ds), min(1, val * dv))
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out

def shaded(v):
    """복셀마다 빛 방향과 파묻힌 정도로 4단계 중 하나를 골라 칠한 RGB 배열."""
    occ = v.g > 0
    sm = ndi.gaussian_filter(occ.astype(float), 1.2); gz, gy, gx = np.gradient(sm)
    nl = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-9
    L = np.array([-0.32, -0.05, 0.95]); L /= np.linalg.norm(L)
    lam = -(gx * L[0] + gy * L[1] + gz * L[2]) / nl
    ao = ndi.gaussian_filter(occ.astype(float), 3.0)
    val = 0.76 * (lam * 0.5 + 0.5) + 0.24 * (1.0 - np.clip((ao - 0.25) / 0.45, 0, 1))
    dith = (((v._x + v._y + v._z) % 2) - 0.5) * 0.03
    tone = 3 - np.clip(np.digitize(val + dith, [0.36, 0.50, 0.70]), 0, 3)
    col = np.zeros(occ.shape + (3,), np.uint8)
    for i, c in enumerate(v.pal):
        if i == 0: continue
        m = v.g == i
        if m.any(): col[m] = np.array(ramp(c[:3]), np.uint8)[tone[m]]
    return occ, col

def export(name, v, vox, origin, extra=None):
    """층들을 격자(아틀라스)로 깐 PNG와 메타를 만든다. vox=복셀 한 칸의 게임 단위 길이, origin=게임 원점이 될 복셀 좌표(x,y,z)."""
    occ, col = shaded(v)
    h, d, w = occ.shape; cols = int(math.ceil(math.sqrt(h))); rows = int(math.ceil(h / cols))
    im = np.zeros((rows * d, cols * w, 4), np.uint8)
    for z in range(h):
        r, k = divmod(z, cols)
        im[r * d:(r + 1) * d, k * w:(k + 1) * w, :3] = col[z]; im[r * d:(r + 1) * d, k * w:(k + 1) * w, 3] = occ[z] * 255
    Image.fromarray(im, "RGBA").save(f"{name}_strip.png")
    META[name] = dict(w=v.w, d=v.d, h=v.h, cols=cols, vox=vox, o=list(origin), **(extra or {}))
    return occ, col

# 커피요정은 fairy/ 폴더의 스크립트가 따로 만든다(원화 시트가 필요). 여기서는 그 결과(fairy_*_strip.png, fairy_meta.json)를 메타에 합치기만 한다.
META.update(json.load(open("fairy_meta.json")))

# ───────────────────────── 무궁화 고리 ─────────────────────────
def flower_ring():
    v = Vox(43, 9, 43)
    X, Y, Z = v._x, v._y, v._z
    P1 = v.color((250, 167, 188)); P2 = v.color((252, 192, 208)); PC = v.color((226, 96, 140))
    YL = v.color((250, 210, 90)); GR = v.color((92, 150, 68)); RING = v.color((255, 251, 232))
    cx, cy, cz = 21, 4, 21
    r = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)
    v.g[(np.abs(r - 19.0) <= 1.1) & (np.abs(Y - cy) <= 1)] = RING        # 고리
    for ang in (54, 126, 270):                                           # 잎
        ax, az = cx + 9.5 * math.cos(math.radians(ang)), cz + 9.5 * math.sin(math.radians(ang))
        v.ellipsoid((ax, cy, az), (3.2, 1.0, 3.2), GR)
    for k in range(5):                                                   # 꽃잎 5장
        ang = math.radians(90 + 72 * k)
        px, pz = cx + 5.8 * math.cos(ang), cz + 5.8 * math.sin(ang)
        v.ellipsoid((px, cy, pz), (5.4, 1.6, 5.4), P1)
        v.ellipsoid((cx + 7.6 * math.cos(ang), cy, cz + 7.6 * math.sin(ang)), (2.6, 1.9, 2.6), P2)
    v.ellipsoid((cx, cy, cz), (3.4, 2.2, 3.4), PC)                       # 진한 가운데
    v.g[(np.abs(X - cx) <= 0.6) & (np.abs(Z - cz) <= 0.6) & (np.abs(Y - cy) <= 4)] = YL   # 꽃술(앞뒤로)
    return v
fl = flower_ring()
export("flower", fl, 0.8, (21, 4, 21))

# ───────────────────────── 졸음 구름 (기존 게임의 덩어리 배치 그대로) ─────────────────────────
def sleeper():
    S = 2.0
    v = Vox(34, 18, 18)
    L = v.color((0x7a, 0x71, 0x98)); Dk = v.color((0x5d, 0x55, 0x80))
    ox, oy, oz = 17, 9, 7                                                # 게임 원점(0,0,0)의 복셀 좌표
    def ball(col, x, y, z, sx, sy, sz):                                  # 게임 좌표(x, y위, z) -> 복셀
        v.ellipsoid((ox + x / S, oy + z / S, oz + y / S), (sx / S, sz / S, sy / S), col)
    ball(L, 0, 0, 0, 20, 13, 15); ball(L, -17, -3, 2, 13, 9, 11); ball(L, 16, -2, -2, 14, 10, 11); ball(L, 4, 9, 0, 12, 9, 10)
    ball(Dk, 0, -7, 0, 24, 6, 13)
    return v
sl = sleeper()
export("sleeper", sl, 2.0, (17, 9, 7))

# ───────────────────────── 원두 ─────────────────────────
def bean():
    v = Vox(13, 9, 8)
    X, Y, Z = v._x, v._y, v._z
    B = v.color((108, 72, 55)); Dk = v.color((59, 38, 25)); Cr = v.color((77, 49, 35)); Hi = v.color((137, 107, 88))
    v.ellipsoid((6, 4, 3.5), (6.2, 4.3, 3.8), B)
    for x in range(2, 11):                                               # 가운데 홈(굽은 선): 기둥마다 맨 위 복셀
        yc = int(round(4 + 0.9 * math.sin((x - 6) / 1.6)))
        zs = np.nonzero(v.g[:, yc, x])[0]
        if len(zs): v.g[zs[-1], yc, x] = Dk
    v.g[(v.g == B) & (Z <= 1)] = Cr
    v.dot(3, 2, 6, Hi); v.dot(4, 2, 6, Hi)
    return v
bn = bean()
export("bean", bn, 1.0 / 12, (6, 4, 3.5))


# ───────────────────────── 건물 (시안: 세계관 글에 외형 묘사가 없어 캐릭터 시트의 색과 소품에서 따옴) ─────────────────────────
def cafe():
    """카페 요정의집: 크림색 벽, 갈색 박공지붕, 지붕 위 커피잔."""
    v = Vox(40, 32, 34)
    X, Y, Z = v._x, v._y, v._z
    WALL = v.color((242, 226, 196)); BASE = v.color((150, 100, 70)); ROOF = v.color((120, 70, 48)); ROOF_D = v.color((96, 60, 44))
    DOOR = v.color((84, 50, 30)); GLASS = v.color((170, 215, 230)); FRAME = v.color((250, 244, 232))
    CUP = v.color((248, 242, 230)); COFFEE = v.color((84, 50, 30)); PINK = v.color((238, 104, 150)); AWN = v.color((250, 160, 190))
    v.box((3, 4, 0), (36, 27, 15), WALL); v.box((3, 4, 0), (36, 27, 1), BASE)
    for z in range(16, 29):                                              # 박공지붕(용마루가 x 방향)
        k = (z - 16) * 1.1
        y0, y1 = int(round(2 + k)), int(round(29 - k))
        if y1 < y0: break
        v.box((2, y0, z), (37, y1, z), ROOF if z % 2 == 0 else ROOF_D)
        v.box((3, y0 + 1, z), (3, y1 - 1, z), WALL) if y1 - y0 >= 2 else None
        v.box((36, y0 + 1, z), (36, y1 - 1, z), WALL) if y1 - y0 >= 2 else None
    v.box((18, 4, 2), (21, 4, 9), DOOR)                                  # 앞문
    for x0 in (7, 26):                                                   # 앞 창문
        v.box((x0, 4, 5), (x0 + 6, 4, 10), FRAME); v.box((x0 + 1, 4, 6), (x0 + 5, 4, 9), GLASS)
    for x in range(5, 35):                                               # 분홍·흰 줄무늬 차양
        v.box((x, 2, 12), (x, 3, 12), AWN if (x // 2) % 2 == 0 else FRAME)
    for sx in (3, 36):                                                   # 옆 창문
        v.box((sx, 12, 6), (sx, 19, 10), GLASS)
    v.cylinder((20, 15.5), 5.2, 28, 28, CUP)                             # 지붕 위 커피잔(받침·잔·커피)
    v.cylinder((20, 15.5), 3.8, 29, 32, CUP); v.cylinder((20, 15.5), 2.8, 32, 32, COFFEE)
    v.box((24, 15, 30), (25, 16, 31), CUP)
    for (x, z) in ((19, 12), (20, 13), (21, 12), (20, 11), (20, 12)):    # 문 위 꽃 장식
        v.box((x, 3, z + 1), (x, 3, z + 1), PINK)
    return v
cf = cafe()
export("cafe", cf, 3.0, (20, 16, 0), dict(hx=57, hz=42, top=84))

def roastery():
    """로스팅 공장: 벽돌 벽, 톱니지붕, 굴뚝, 앞벽의 원두 표시."""
    v = Vox(48, 36, 44)
    X, Y, Z = v._x, v._y, v._z
    BR = v.color((176, 96, 72)); BR_D = v.color((140, 74, 58)); ROOF = v.color((118, 112, 118)); ROOF_D = v.color((92, 88, 96))
    GLASS = v.color((170, 215, 230)); DOOR = v.color((70, 66, 74)); CREAM = v.color((242, 226, 196)); BEAN = v.color((108, 72, 55)); BEAN_D = v.color((59, 38, 25))
    v.box((2, 4, 0), (45, 33, 17), BR); v.box((2, 4, 0), (45, 33, 2), BR_D)
    for z in (6, 12):                                                    # 벽돌 줄눈
        v.g[(v.g == BR) & (Z == z)] = BR_D
    v.box((2, 4, 18), (45, 33, 18), ROOF_D)
    for t in range(3):                                                   # 톱니지붕 3칸: 한쪽은 비탈, 한쪽은 유리
        x0 = 2 + t * 14
        for x in range(x0, x0 + 14):
            top = 19 + int((x - x0) * 6 / 13)
            v.box((x, 5, 19), (x, 32, top), ROOF)
        v.box((x0 + 13, 6, 20), (x0 + 13, 31, 24), GLASS)
    v.box((20, 4, 0), (27, 4, 10), DOOR)                                 # 큰 문
    for x0 in (5, 11, 32, 38):                                           # 앞 창문
        v.box((x0, 4, 9), (x0 + 3, 4, 13), GLASS)
    v.box((7, 3, 2), (15, 3, 7), CREAM)                                  # 원두 표시판
    v.ellipsoid((11, 3, 4.5), (3.4, 0.6, 2.2), BEAN); v.box((9, 3, 4), (13, 3, 4), BEAN_D)
    v.cylinder((40, 26), 3.2, 18, 41, BR_D)                              # 굴뚝
    v.cylinder((40, 26), 3.2, 36, 37, CREAM); v.cylinder((40, 26), 3.6, 42, 42, ROOF_D); v.cylinder((40, 26), 2.0, 42, 42, 0)
    return v
rs = roastery()
export("roastery", rs, 3.5, (24, 18, 0), dict(hx=77, hz=53, top=88, chim=[56, 150.5, 28]))


# ───────────────────────── 부술 것들 (세계관 [본편]에 나오는 물건: 결점두, 디카페인, 원두수리검, 원두탄환, 중절모) ─────────────────────────
def rnd(i, j, k):                                                         # 무늬용 고정 난수(매번 같은 결과)
    n = (i * 374761393 + j * 668265263 + k * 2147483647) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0

def badbean():
    """결점두(상한콩): 누렇게 뜬 원두에 곰팡이 점과 깨진 자리."""
    v = Vox(21, 15, 13)
    X, Y, Z = v._x, v._y, v._z
    B = v.color((126, 118, 72)); Dk = v.color((62, 54, 36)); Mold = v.color((170, 192, 124)); Rot = v.color((44, 40, 34)); Lo = v.color((98, 90, 56))
    v.ellipsoid((10, 7, 6), (10.2, 7.2, 6.2), B)
    v.g[(v.g == B) & (Z <= 2)] = Lo
    for x in range(2, 19):                                               # 가운데 홈
        yc = int(round(7 + 1.4 * math.sin((x - 10) / 2.6)))
        zs = np.nonzero(v.g[:, yc, x])[0]
        if len(zs): v.g[zs[-1], yc, x] = Dk
    solid = v.g > 0
    for z in range(13):                                                  # 곰팡이·썩은 점(겉면)
        for y in range(15):
            for x in range(21):
                if not solid[z, y, x]: continue
                edge = (z == 12 or not solid[z + 1, y, x]) or x in (0, 20) or not solid[z, y, x - 1] or not solid[z, y, min(20, x + 1)] or not solid[z, max(0, y - 1), x] or not solid[z, min(14, y + 1), x]
                if not edge or v.g[z, y, x] == Dk: continue
                r = rnd(x // 2, y // 2, z // 2)
                if r > 0.86: v.g[z, y, x] = Mold
                elif r < 0.07: v.g[z, y, x] = Rot
    v.ellipsoid((17.5, 3, 9.5), (3.2, 3.2, 3.2), 0)                      # 깨져 나간 자리
    return v
export("badbean", badbean(), 2.2, (10, 7, 6))

def decaf():
    """디카페인 잔: 푸르스름한 띠가 둘린 잔에 묽은 커피."""
    v = Vox(21, 21, 16)
    X, Y, Z = v._x, v._y, v._z
    CUP = v.color((226, 230, 236)); BAND = v.color((112, 138, 176)); TEA = v.color((196, 176, 146)); SAUC = v.color((204, 210, 220))
    c = (10, 10)
    v.cylinder(c, 9.6, 0, 0, SAUC); v.cylinder(c, 6.0, 1, 1, SAUC)
    for z in range(2, 15):
        v.cylinder(c, 5.6 + (z - 2) * 0.16, z, z, BAND if z in (7, 8, 9) else CUP)
    v.cylinder(c, 6.6, 14, 14, TEA)
    r = np.sqrt((X - 10) ** 2 + (Y - 10) ** 2)
    v.g[(Z == 14) & (r > 6.6) & (r <= 7.6)] = CUP
    v.g[(np.abs(Y - 10) <= 1.2) & (((X - 18.6) / 2.8) ** 2 + ((Z - 8.5) / 3.6) ** 2 <= 1)] = CUP      # 손잡이
    v.g[(np.abs(Y - 10) <= 1.2) & (((X - 18.6) / 1.3) ** 2 + ((Z - 8.5) / 2.1) ** 2 <= 1)] = 0
    v.cylinder(c, 5.2, 2, 13, CUP)                                        # 속을 채워 둔다(부서질 때 판이 되게)
    return v
export("decaf", decaf(), 2.2, (10, 10, 8))

def shuriken():
    """원두수리검: 원두 네 알이 날이 된 표창. 세워서 만든다(면이 앞을 본다)."""
    v = Vox(31, 3, 31)
    X, Y, Z = v._x, v._y, v._z
    B = v.color((108, 72, 55)); Dk = v.color((59, 38, 25)); ST = v.color((156, 158, 170)); ST_D = v.color((104, 106, 120)); Hi = v.color((137, 107, 88))
    cx = cz = 15
    for k in range(4):
        a = math.radians(90 * k)
        ux, uz = math.cos(a), math.sin(a)
        px, pz = (X - cx) * ux + (Z - cz) * uz, -(X - cx) * uz + (Z - cz) * ux
        m = (((px - 9) / 6.4) ** 2 + ((pz - (px - 9) * 0.25) / 3.0) ** 2 <= 1)            # 살짝 휜 원두 날
        v.g[m] = B
        v.g[m & (np.abs(pz - (px - 9) * 0.25) < 0.55) & (np.abs(px - 9) < 5) & (Y != 1)] = Dk
        v.g[m & (pz - (px - 9) * 0.25 > 1.8) & (Y == 1)] = Hi
    r = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2)
    v.g[r <= 4.4] = ST; v.g[(r <= 4.4) & (r > 3.2)] = ST_D; v.g[r <= 1.4] = 0
    return v
export("shuriken", shuriken(), 1.6, (15, 1, 15))

def bullet():
    """원두탄환: 앞이 뾰족한 원두 탄두와 놋쇠색 탄피. 앞(-y)을 향한다."""
    v = Vox(11, 24, 11)
    X, Y, Z = v._x, v._y, v._z
    B = v.color((108, 72, 55)); Dk = v.color((59, 38, 25)); BR = v.color((206, 166, 84)); BR_D = v.color((160, 122, 58))
    r = np.sqrt((X - 5) ** 2 + (Z - 5) ** 2)
    v.g[(Y >= 12) & (Y <= 22) & (r <= 4.6)] = BR                         # 탄피
    v.g[(Y >= 22) & (Y <= 23) & (r <= 5.2)] = BR_D
    v.g[(Y == 12) & (r <= 4.6)] = BR_D
    v.g[(Y < 12) & (r <= 4.4 * np.sqrt(np.clip((Y + 0.5) / 12.0, 0, 1)))] = B           # 원두 탄두
    v.g[(v.g == B) & (np.abs(X - 5) < 0.6) & (Z >= 5)] = Dk              # 원두 홈
    return v
export("bullet", bullet(), 2.0, (5, 12, 5))

def fedora():
    """중절모: 넓은 챙, 띠, 가운데가 눌린 꼭대기."""
    v = Vox(31, 27, 13)
    X, Y, Z = v._x, v._y, v._z
    F = v.color((74, 70, 80)); F_D = v.color((54, 50, 60)); BAND = v.color((34, 30, 38)); F_L = v.color((96, 92, 104))
    cx, cy = 15, 13
    e = ((X - cx) / 15.0) ** 2 + ((Y - cy) / 12.6) ** 2
    v.g[(Z == 1) & (e <= 1)] = F                                          # 챙
    v.g[(Z == 0) & (e <= 0.86)] = F_D
    v.g[(Z == 2) & (e <= 1) & (e > 0.8)] = F                              # 챙 끝이 살짝 말림
    for z in range(2, 12):                                               # 모자 몸통
        k = 1 - (z - 2) * 0.022
        m = (((X - cx) / (8.6 * k)) ** 2 + ((Y - cy) / (7.4 * k)) ** 2 <= 1) & (Z == z)
        v.g[m] = BAND if z in (2, 3) else (F_L if z >= 10 else F)
    v.g[(Z >= 10) & (np.abs(X - cx) <= 1.6) & (np.abs(Y - cy) <= 5.0)] = 0     # 꼭대기 눌린 골
    v.g[(Z == 9) & (np.abs(X - cx) <= 1.6) & (np.abs(Y - cy) <= 5.0)] = F_D
    return v
export("fedora", fedora(), 2.6, (15, 13, 5))

json.dump(META, open("meta.json", "w"), indent=1)
print(json.dumps(META))
