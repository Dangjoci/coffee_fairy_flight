"""커피의 요정(귀여운 3D 버전) — 서 있는 자세. 치수는 360° 턴어라운드 시트 정면/측면에서 잰 값
(시트 px * 0.1756 = 복셀, 머리 폭 205px -> 36복셀)."""
import sys, math
import numpy as np
import sys; sys.path.insert(0, '..')
from stackpix import *

W, D, H = 65, 57, 64
v = Vox(W, D, H)
cx, cy = 32, 28
X, Y, Z = v._x, v._y, v._z
SKIN = v.color((247, 198, 152)); FOOT = v.color((238, 172, 112))
SHORT = v.color((96, 60, 44)); TIE = v.color((222, 170, 110))
BEARD = v.color((112, 64, 38)); DARK = v.color((66, 40, 30))
CUP = v.color((248, 242, 230)); COFFEE = v.color((84, 50, 30))
PINK = v.color((238, 104, 150)); PINK_L = v.color((250, 160, 190)); YEL = v.color((250, 205, 80))
GREEN = v.color((92, 150, 68)); WING = v.color((246, 243, 238)); WING_E = v.color((224, 198, 150))
WING_V = v.color((236, 222, 196)); BLUSH = v.color((246, 150, 142)); NIP = v.color((232, 150, 130)); SKIN_SH = v.color((232, 176, 130))

def put(mask, col): v.g[mask] = col

# ── 날개: 뒤-바깥으로 33도 벌어진 수직 판 두 쌍 (몸보다 먼저)
def wing(side, root_z, u0, z0, a, b, tilt_deg, sweep=33, t=2.3):
    sw = math.radians(sweep); ry_back = cy + 7.5
    dx, dy = side * math.cos(sw), math.sin(sw)
    rx_, ry_ = X - cx, Y - ry_back
    u = rx_ * dx + ry_ * dy                     # 날개 방향 거리
    n = -rx_ * dy + ry_ * dx                    # 판 두께 방향
    tl = math.radians(tilt_deg)
    pu, pz = u - u0, Z - z0
    a_ = pu * math.cos(tl) + pz * math.sin(tl)
    b_ = -pu * math.sin(tl) + pz * math.cos(tl)
    e = (a_ / a) ** 2 + (b_ / b) ** 2
    slab = (np.abs(n) <= t) & (u >= 0)
    put(slab & (e <= 1), WING_E)
    put(slab & (((a_) / (a - 1.0)) ** 2 + (b_ / (b - 1.0)) ** 2 <= 1), WING)
    put(slab & (e <= 0.7) & (np.abs(b_) < 0.5) & (a_ < a * 0.5), WING_V)      # 가운데 맥
for s in (-1, 1):
    wing(s, 24, 16, 28, 15.5, 7.5, 14)          # 위 날개(큰 쌍)
    wing(s, 22, 11, 15, 11.5, 5.2, -36)         # 아래 날개(작은 쌍)

# ── 발, 반바지, 몸통, 팔
for s in (-1, 1):
    v.ellipsoid((cx + s * 5.5, cy - 1, 2.2), (4.2, 4.6, 2.8), FOOT)
v.ellipsoid((cx, cy, 10.5), (14.0, 10.5, 7.5), SHORT)
put((v.g == SHORT) & (Z > 14), 0)                                        # 허리선에서 자름
v.ellipsoid((cx, cy - 0.5, 19), (11.8, 9.6, 9.5), SKIN)                  # 배·가슴
put((v.g == SKIN) & (Z < 12) & (((X - cx) / 13.6) ** 2 + ((Y - cy) / 10.2) ** 2 <= 1) & (Z >= 4), SHORT)
for s in (-1, 1):
    m = (((X - (cx + s * (11 + (24 - Z) * 0.55))) / 3.4) ** 2 + ((Y - (cy - 1)) / 3.6) ** 2 <= 1) & (Z >= 15.5) & (Z <= 25)
    put(m, SKIN)
    v.ellipsoid((cx + s * 15.8, cy - 1, 16.5), (3.4, 3.6, 3.0), SKIN)    # 손

put((v.g == SKIN) & (Z >= 22.5), SKIN_SH)                              # 턱 밑 그늘
# ── 머리(폭 36, 깊이 31, 높이 31)
put(np.abs((X - cx) / 18.0) ** 2.2 + np.abs((Y - cy) / 15.5) ** 2.2 + np.abs((Z - 39) / 15.6) ** 2.2 <= 1, SKIN)

def front(x, z, col, depth=2):
    ys = np.nonzero(v.g[z, :, x])[0]
    if len(ys):
        v.g[z, ys[0]:ys[0] + depth, x] = col

# 얼굴: 눈썹, 웃는 눈(^), 볼터치, 코밑 콧수염(끝이 말려 올라감)
for s in (-1, 1):
    for dx, z in ((5, 45), (6, 46), (7, 46), (8, 45)): front(cx + s * dx, z, DARK)        # 눈썹
    for dx, z in ((4, 40), (5, 41), (6, 42), (7, 42), (8, 41), (9, 40)): front(cx + s * dx, z, DARK)  # 눈
    for dx in (11, 12, 13):
        for z in (36, 37): front(cx + s * dx, z, BLUSH)
    for dx, z in ((1, 36), (2, 36), (3, 36), (4, 35), (5, 36), (6, 37)): front(cx + s * dx, z, DARK, 3)  # 콧수염
    for dx, z in ((1, 35), (2, 35), (3, 35)): front(cx + s * dx, z, DARK, 3)
front(cx, 37, NIP); front(cx, 36, DARK, 3)

# 턱수염: 턱에서 가슴 아래까지, 아래로 갈수록 뾰족
for z in range(20, 35):
    half = 3.2 * min(1.0, (z - 19) / 9.0)
    for x in range(W):
        if abs(x - cx) <= half:
            ys = np.nonzero(v.g[z, :, x])[0]
            y0 = ys[0] if len(ys) else cy - 10
            y0 = min(y0, cy - 11)                                   # 가슴 앞으로 늘어짐
            v.g[z, y0 - 2:y0 + 2, x] = BEARD
# 가슴 점, 배꼽, 허리끈 리본
for s in (-1, 1): front(cx + s * 6, 24, NIP, 1)
front(cx, 16, FOOT, 1)
for dx, z in ((0, 13), (-1, 14), (1, 14), (-2, 13), (2, 13), (-1, 12), (1, 12), (-1, 11), (1, 11)): front(cx + dx, z, TIE)

# ── 꽃: 머리 양옆에서 앞-바깥 35도를 향함. 꽃잎 5장 + 노란 꽃술 + 초록 잎 3장
for s in (-1, 1):
    a35 = math.radians(18)
    n = np.array([s * math.cos(a35), -math.sin(a35), 0.0])          # 꽃이 향하는 방향
    e1 = np.array([s * math.sin(a35), math.cos(a35), 0.0]); e2 = np.array([0, 0, 1.0])
    c = np.array([cx, cy, 46.5]) + n * 17.6
    def blob(p, r, col):
        v.ellipsoid(tuple(p), (r, r, r), col)
    for ang in (20, 140, 260):
        blob(c - n * 1.2 + 5.6 * (math.cos(math.radians(ang)) * e1 + math.sin(math.radians(ang)) * e2), 2.9, GREEN)
    for k in range(5):
        ang = math.radians(90 + 72 * k)
        blob(c + 3.3 * (math.cos(ang) * e1 + math.sin(ang) * e2), 2.7, PINK if k % 2 == 0 else PINK_L)
    blob(c + n * 0.6, 2.4, PINK)
    blob(c + n * 2.2, 1.4, YEL)

# ── 커피잔: 받침 + 잔 + 커피 + 손잡이(정면에서 볼 때 오른쪽 = -x)
v.cylinder((cx, cy), 8.2, 54, 54, CUP)
v.cylinder((cx, cy), 5.6, 55, 55, CUP)
for z in range(56, 62):
    v.cylinder((cx, cy), 5.4 + (z - 56) * 0.18, z, z, CUP)
v.cylinder((cx, cy), 5.0, 61, 61, COFFEE)
put((np.abs(Y - cy) <= 1.2) & (((X - (cx - 7.6)) / 2.6) ** 2 + ((Z - 58.5) / 2.6) ** 2 <= 1), CUP)
put((np.abs(Y - cy) <= 1.2) & (((X - (cx - 7.6)) / 1.3) ** 2 + ((Z - 58.5) / 1.3) ** 2 <= 1), 0)


import pickle
pickle.dump((v.g,v.pal),open('prior.pkl','wb'))
print(v.g.shape,len(v.pal))
