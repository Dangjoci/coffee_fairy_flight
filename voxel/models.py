"""향기 순찰 — 게임 요소 복셀 모델. 층 스트립(왼쪽이 맨 아래층)과 메타데이터를 만든다.
좌표: x=오른쪽, y=뒤(작을수록 앞), z=위."""
import math, json, sys
import numpy as np
from stackpix import *

META = {}

def export(name, v, vox, origin, extra=None):
    """vox=복셀 한 칸의 게임 단위 길이, origin=게임에서 원점이 될 복셀 좌표(x,y,z)."""
    s = v.slices()
    save_strip(s, f"{name}_strip.png")
    META[name] = dict(w=v.w, d=v.d, h=v.h, vox=vox, o=list(origin), **(extra or {}))
    return s

# ───────────────────────── 커피요정 (비행 자세) ─────────────────────────
# 치수: 비행 시트·360° 시트에서 잰 값. 머리 폭 36복셀 기준.
W, D, H = 69, 56, 64
CX, CY = 34, 19                    # 머리 중심(x, y)
LEAN = math.radians(35)            # 몸을 뒤로 눕힌 각도(목 기준)
PIV = (CY + 2.0, 26.0)             # 목 회전축 (y, z)

PAL = dict(SKIN=(247, 198, 152), FOOT=(238, 172, 112), SHORT=(96, 60, 44), TIE=(222, 170, 110),
           BEARD=(112, 64, 38), DARK=(66, 40, 30), CUP=(248, 242, 230), COFFEE=(84, 50, 30),
           PINK=(238, 104, 150), PINK_L=(250, 160, 190), YEL=(250, 205, 80), GREEN=(92, 150, 68),
           WING=(246, 243, 238), WING_E=(224, 198, 150), WING_V=(236, 222, 196),
           BLUSH=(246, 150, 142), NIP=(232, 150, 130), SKIN_SH=(232, 176, 130))

def newvox():
    v = Vox(W, D, H)
    c = {k: v.color(rgb) for k, rgb in PAL.items()}
    return v, c

def fairy_body():
    # 1) 서 있는 좌표로 몸통을 만든다(머리 없이). 팔은 앞으로, 다리는 뒤로 접음.
    b, c = newvox()
    X, Y, Z = b._x, b._y, b._z
    cx, cy = CX, CY + 2.0
    def put(m, col): b.g[m] = col
    for s in (-1, 1):                                                   # 뒤로 접은 발(발바닥이 뒤를 봄)
        b.ellipsoid((cx + s * 5.5, cy + 7.5, 5.0), (4.0, 3.2, 4.0), c['FOOT'])
    b.ellipsoid((cx, cy, 10.5), (14.0, 10.5, 7.5), c['SHORT'])
    put((b.g == c['SHORT']) & (Z > 14), 0)
    b.ellipsoid((cx, cy - 0.5, 19), (11.8, 9.6, 9.5), c['SKIN'])
    put((b.g == c['SKIN']) & (Z < 12) & (((X - cx) / 13.6) ** 2 + ((Y - cy) / 10.2) ** 2 <= 1) & (Z >= 4), c['SHORT'])
    for s in (-1, 1):                                                   # 가슴 앞으로 모은 팔
        b.ellipsoid((cx + s * 10.5, cy - 6.5, 20.5), (3.6, 5.4, 3.6), c['SKIN'])
        b.ellipsoid((cx + s * 9.5, cy - 11.0, 21.5), (3.3, 3.3, 3.3), c['SKIN'])
    def front(x, z, col, depth=2):
        ys = np.nonzero(b.g[z, :, x])[0]
        if len(ys): b.g[z, ys[0]:ys[0] + depth, x] = col
    for s in (-1, 1): front(int(cx + s * 6), 24, c['NIP'], 1)
    front(int(cx), 16, c['FOOT'], 1)
    for dx, z in ((0, 13), (-1, 14), (1, 14), (-2, 13), (2, 13), (-1, 12), (1, 12), (-1, 11), (1, 11)):
        front(int(cx) + dx, z, c['TIE'])
    put((b.g == c['SKIN']) & (Z >= 22.5), c['SKIN_SH'])               # 턱 밑 그늘

    # 2) 목을 축으로 35도 눕혀 최종 격자로 옮긴다(가장 가까운 복셀 표본).
    v, c = newvox()
    X, Y, Z = v._x, v._y, v._z
    dy, dz = Y - PIV[0], Z - PIV[1]
    sy = dy * math.cos(LEAN) + dz * math.sin(LEAN) + PIV[0]
    sz = -dy * math.sin(LEAN) + dz * math.cos(LEAN) + PIV[1]
    iy, iz, ix = np.rint(sy).astype(int), np.rint(sz).astype(int), X.astype(int)
    ok = (iy >= 0) & (iy < D) & (iz >= 0) & (iz < H)
    v.g[ok] = b.g[iz[ok], iy[ok], ix[ok]]

    # 3) 머리(눕히지 않음): 아래가 넓적한 둥근 형태
    def put(m, col): v.g[m] = col
    put(np.abs((X - CX) / 18.0) ** 2.6 + np.abs((Y - CY) / 15.5) ** 2.6 + np.abs((Z - 39) / 15.6) ** 2.6 <= 1, c['SKIN'])
    def front(x, z, col, depth=2):
        ys = np.nonzero(v.g[z, :, x])[0]
        if len(ys): v.g[z, ys[0]:ys[0] + depth, x] = col
    for s in (-1, 1):
        for dx, z in ((5, 45), (6, 46), (7, 46), (8, 45)): front(CX + s * dx, z, c['DARK'])
        for dx, z in ((4, 40), (5, 41), (6, 42), (7, 42), (8, 41), (9, 40)): front(CX + s * dx, z, c['DARK'])
        for dx in (11, 12, 13):
            for z in (36, 37): front(CX + s * dx, z, c['BLUSH'])
        for dx, z in ((1, 36), (2, 36), (3, 36), (4, 35), (5, 36), (6, 37)): front(CX + s * dx, z, c['DARK'], 3)
        for dx, z in ((1, 35), (2, 35), (3, 35)): front(CX + s * dx, z, c['DARK'], 3)
    front(CX, 37, c['NIP']); front(CX, 36, c['DARK'], 3)
    for z in range(19, 35):                                             # 턱에서 곧게 늘어진 턱수염
        half = 3.2 * min(1.0, (z - 18) / 9.0)
        for x in range(W):
            if abs(x - CX) <= half:
                v.g[z, CY - 15:CY - 11, x] = c['BEARD']
    for s in (-1, 1):                                                   # 꽃
        a = math.radians(18)
        n = np.array([s * math.cos(a), -math.sin(a), 0.0])
        e1 = np.array([s * math.sin(a), math.cos(a), 0.0]); e2 = np.array([0, 0, 1.0])
        p0 = np.array([CX, CY, 46.5]) + n * 17.6
        def blob(p, r, col): v.ellipsoid(tuple(p), (r, r, r), col)
        for ang in (20, 140, 260):
            blob(p0 - n * 1.2 + 5.6 * (math.cos(math.radians(ang)) * e1 + math.sin(math.radians(ang)) * e2), 2.9, c['GREEN'])
        for k in range(5):
            ang = math.radians(90 + 72 * k)
            blob(p0 + 3.3 * (math.cos(ang) * e1 + math.sin(ang) * e2), 2.7, c['PINK'] if k % 2 == 0 else c['PINK_L'])
        blob(p0 + n * 0.6, 2.4, c['PINK']); blob(p0 + n * 2.2, 1.4, c['YEL'])
    v.cylinder((CX, CY), 8.2, 54, 54, c['CUP'])                         # 커피잔
    v.cylinder((CX, CY), 5.6, 55, 55, c['CUP'])
    for z in range(56, 62): v.cylinder((CX, CY), 5.4 + (z - 56) * 0.18, z, z, c['CUP'])
    v.cylinder((CX, CY), 5.0, 61, 61, c['COFFEE'])
    put((np.abs(Y - CY) <= 1.2) & (((X - (CX - 7.6)) / 2.6) ** 2 + ((Z - 58.5) / 2.6) ** 2 <= 1), c['CUP'])
    put((np.abs(Y - CY) <= 1.2) & (((X - (CX - 7.6)) / 1.3) ** 2 + ((Z - 58.5) / 1.3) ** 2 <= 1), 0)
    return v

SWEEP = 24
ROOT = (CY + 11.0, 26.0)           # 날개 뿌리 (y, z): 눕힌 등 위쪽, 머리 뒤통수 아래
def fairy_wings(up_tilt, lo_tilt, body):
    v, c = newvox()
    X, Y, Z = v._x, v._y, v._z
    def wing(side, a, b, tilt_deg, sweep=SWEEP, t=1.0):
        sw = math.radians(sweep)
        dx, dy = side * math.cos(sw), math.sin(sw)
        rx_, ry_ = X - CX, Y - ROOT[0]
        u = rx_ * dx + ry_ * dy
        n = -rx_ * dy + ry_ * dx
        tl = math.radians(tilt_deg)
        pz = Z - ROOT[1]
        a_ = u * math.cos(tl) + pz * math.sin(tl) - a
        b_ = -u * math.sin(tl) + pz * math.cos(tl)
        e = (a_ / a) ** 2 + (b_ / b) ** 2
        slab = (np.abs(n) <= t) & (u >= 0)
        v.g[slab & (e <= 1)] = c['WING_E']
        v.g[slab & ((a_ / (a - 1.0)) ** 2 + (b_ / (b - 1.0)) ** 2 <= 1)] = c['WING']
        v.g[slab & (e <= 0.7) & (np.abs(b_) < 0.5) & (a_ < a * 0.5)] = c['WING_V']
    for s in (-1, 1):
        wing(s, 15.5, 7.5, up_tilt)
        wing(s, 11.5, 5.2, lo_tilt)
    v.g[body.g > 0] = 0                                                  # 몸에 묻힌 부분은 뺀다
    return v

body = fairy_body()
FV = 0.16                                                               # 요정 복셀 한 칸의 게임 단위 길이(키 약 10)
ORG = (CX, CY + 6, 30)                                                   # 게임에서 요정 위치가 될 점(몸 한가운데)
sb = export("fairy_body", body, FV, ORG)
wings = []
for k, (ut, lt) in enumerate(((30, -12), (14, -24), (-4, -36))):        # 날갯짓: 위 / 가운데 / 아래
    wv = fairy_wings(ut, lt, body)
    export(f"fairy_wing{k}", wv, FV, ORG)
    wings.append(wv)
# 날개 끝·엉덩이 위치(게임의 궤적·원두 발사 지점)
sw = math.radians(SWEEP)
tip = (CX + 31 * math.cos(sw), ROOT[0] + 31 * math.sin(sw), ROOT[1] + 5)
META["fairy_body"]["tip"] = [round(tip[0] - ORG[0], 2), round(tip[1] - ORG[1], 2), round(tip[2] - ORG[2], 2)]
by = PIV[0] + (10) * math.cos(LEAN) - (-18) * math.sin(LEAN)             # 엉덩이(서 있는 좌표 y+10, z 8)
bz = PIV[1] + (10) * math.sin(LEAN) + (-18) * math.cos(LEAN)
META["fairy_body"]["butt"] = [0, round(by - ORG[1], 2), round(bz - ORG[2], 2)]

# 확인용 합성(몸 + 날개 가운데 프레임)
def merged(k):
    m = Vox(W, D, H); m.pal = body.pal
    m.g = np.where(body.g > 0, body.g, wings[k].g)
    return m

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

json.dump(META, open("meta.json", "w"), indent=1)
print(json.dumps(META))

if __name__ == "__main__":
    kw = dict(spread=1, squash=0.36, shade=0.14)
    m = merged(1).slices()
    sh, fs = sheet(m, dirs=8, zoom=3, **kw); fw, fh = fs
    prev = Image.new("RGBA", (fw * 4, fh * 2), (250, 244, 236, 255))
    for k, i in enumerate([0, 7, 6, 2, 4, 3, 5, 1]):      # 비행 시트 순서와 비슷하게: 뒤부터
        prev.alpha_composite(sh.crop((i * fw, 0, (i + 1) * fw, fh)), ((k % 4) * fw, (k // 4) * fh))
    prev.save("fairy_fly_prev.png")
    row = Image.new("RGBA", (900, 300), (250, 244, 236, 255))
    for i, (s_, z, kk) in enumerate(((fl.slices(), 5, dict(spread=1, squash=0.5, shade=0.1)), (sl.slices(), 7, dict(spread=1, squash=0.5, shade=0.2)), (bn.slices(), 14, dict(spread=1, squash=0.6, shade=0.2)))):
        f = render(s_, 205, **kk); f = f.resize((f.width * z, f.height * z), Image.NEAREST)
        row.alpha_composite(f, (10 + i * 300, 10))
    row.save("props_prev.png")
