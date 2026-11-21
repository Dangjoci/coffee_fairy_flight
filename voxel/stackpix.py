"""stackpix — 적층 픽셀(sprite stacking) 도구.
복셀 모델 -> 층(slice) 스트립 PNG -> 회전 스프라이트 시트 / 턴테이블 GIF.
좌표: x=오른쪽, y=뒤쪽(작을수록 앞), z=위. 층 이미지에서 위쪽이 앞(-y)."""
import math
import numpy as np
from PIL import Image


class Vox:
    """색 인덱스 복셀 격자. 0은 빈칸. 나중에 그린 것이 덮어쓴다."""

    def __init__(self, w, d, h):
        self.w, self.d, self.h = w, d, h
        self.g = np.zeros((h, d, w), np.uint8)          # [z, y, x]
        self.pal = [(0, 0, 0, 0)]
        self._z, self._y, self._x = np.mgrid[0:h, 0:d, 0:w].astype(float)

    def color(self, rgb, a=255):
        self.pal.append((*rgb, a))
        return len(self.pal) - 1

    def ellipsoid(self, c, r, col):
        (cx, cy, cz), (rx, ry, rz) = c, r
        m = ((self._x - cx) / rx) ** 2 + ((self._y - cy) / ry) ** 2 + ((self._z - cz) / rz) ** 2 <= 1
        self.g[m] = col

    def box(self, lo, hi, col):
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        self.g[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = col

    def cylinder(self, c, r, z0, z1, col):
        cx, cy = c
        m = ((self._x - cx) ** 2 + (self._y - cy) ** 2 <= r * r) & (self._z >= z0) & (self._z <= z1)
        self.g[m] = col

    def dot(self, x, y, z, col):
        self.g[z, y, x] = col

    def slices(self):
        """아래층부터 RGBA 이미지 목록."""
        pal = np.array(self.pal, np.uint8)
        return [Image.fromarray(pal[self.g[z]], "RGBA") for z in range(self.h)]


def save_strip(slices, path):
    """층들을 가로로 이어 붙인 스트립(왼쪽이 맨 아래층)."""
    w, h = slices[0].size
    strip = Image.new("RGBA", (w * len(slices), h))
    for i, s in enumerate(slices):
        strip.paste(s, (i * w, 0))
    strip.save(path)


def load_strip(path, layers):
    im = Image.open(path).convert("RGBA")
    w = im.width // layers
    return [im.crop((i * w, 0, (i + 1) * w, im.height)) for i in range(layers)]


def render(slices, angle, spread=1, squash=1.0, shade=0.3, outline=(30, 20, 20, 255)):
    """한 방향 렌더. angle=도(시계 방향), spread=층 간격(px),
    squash=층의 세로 눌림(1=바로 위, 0.5=많이 기울인 시점),
    shade=맨 아래층을 어둡게 하는 정도, outline=외곽선 색 또는 None."""
    n = len(slices)
    w, h = slices[0].size
    diag = int(math.ceil(math.hypot(w, h))) + 2
    ch = int(diag * squash) + n * spread + 4
    out = Image.new("RGBA", (diag + 2, ch))
    for i, s in enumerate(slices):
        if shade:
            k = 1 - shade * (1 - i / max(1, n - 1))
            a = np.asarray(s).astype(float)
            a[..., :3] *= k
            s = Image.fromarray(a.astype(np.uint8), "RGBA")
        pad = Image.new("RGBA", (diag, diag))
        pad.paste(s, ((diag - w) // 2, (diag - h) // 2))
        r = pad.rotate(-angle, Image.NEAREST)
        if squash != 1.0:
            r = r.resize((diag, max(1, int(diag * squash))), Image.NEAREST)
        base = ch - r.height - 2
        for k2 in range(spread):                         # 층 사이 틈 메우기
            out.alpha_composite(r, (1, base - i * spread - k2))
    if outline:
        a = np.asarray(out)[..., 3] > 0
        e = np.zeros_like(a)
        e[1:, :] |= a[:-1, :]; e[:-1, :] |= a[1:, :]
        e[:, 1:] |= a[:, :-1]; e[:, :-1] |= a[:, 1:]
        px = np.asarray(out).copy()
        px[e & ~a] = outline
        out = Image.fromarray(px, "RGBA")
    return out


def sheet(slices, dirs=16, zoom=1, **kw):
    """dirs방향 회전 시트(가로 한 줄). 프레임 크기를 함께 돌려준다."""
    frames = [render(slices, 360 * i / dirs, **kw) for i in range(dirs)]
    fw, fh = frames[0].size
    sh = Image.new("RGBA", (fw * dirs, fh))
    for i, f in enumerate(frames):
        sh.paste(f, (i * fw, 0))
    if zoom > 1:
        sh = sh.resize((sh.width * zoom, sh.height * zoom), Image.NEAREST)
    return sh, (fw * zoom, fh * zoom)


def turntable(slices, path, steps=48, zoom=6, bg=(58, 52, 48), ms=70, **kw):
    """확인용 회전 GIF."""
    fr = []
    for i in range(steps):
        f = render(slices, 360 * i / steps, **kw)
        c = Image.new("RGBA", f.size, (*bg, 255))
        c.alpha_composite(f)
        fr.append(c.resize((f.width * zoom, f.height * zoom), Image.NEAREST).convert("P", palette=Image.ADAPTIVE))
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=ms, loop=0)


def exploded(slices, path, gap=5, zoom=5, angle=30, squash=0.5, bg=(40, 36, 34)):
    """층을 벌려 보여 주는 분해도(구조 확인용)."""
    n = len(slices); w, h = slices[0].size
    diag = int(math.ceil(math.hypot(w, h))) + 2
    ch = int(diag * squash) + n * gap + 4
    out = Image.new("RGBA", (diag + 2, ch), (*bg, 255))
    for i, s in enumerate(slices):
        pad = Image.new("RGBA", (diag, diag)); pad.paste(s, ((diag - w) // 2, (diag - h) // 2))
        r = pad.rotate(-angle, Image.NEAREST).resize((diag, max(1, int(diag * squash))), Image.NEAREST)
        out.alpha_composite(r, (1, ch - r.height - 2 - i * gap))
    out.resize((out.width * zoom, out.height * zoom), Image.NEAREST).save(path)
