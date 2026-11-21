"""360° 턴어라운드 시트에서 여덟 뷰를 배경과 분리해 view0.png … view7.png로 잘라 낸다.
사용: python3 extract_views.py <시트 PNG 경로>   (시트 그림은 저장소에 넣지 않았다)"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
a = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int)
cand = np.abs(a - a[a.shape[0] // 2, a.shape[1] // 2]).sum(2) < 26           # 크림색 배경 후보
lab, n = ndi.label(cand); bg = lab == 1 + int(np.argmax(ndi.sum(cand, lab, range(1, n + 1))))
fg = ndi.binary_opening(~bg, iterations=1)
lab2, n2 = ndi.label(fg)
boxes = []
for i, sl in enumerate(ndi.find_objects(lab2)):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if 200 < h < 600 and w > 120: boxes.append((sl[0].start, sl[1].start, h, w, i + 1))
boxes.sort(key=lambda t: (t[0] // 400, t[1]))                                # 윗줄 왼쪽부터: 0°, 45°, … 315°
for k, (y, x, h, w, l) in enumerate(boxes):
    m = ndi.binary_fill_holes(lab2[y:y + h, x:x + w] == l)
    Image.fromarray(np.dstack([a[y:y + h, x:x + w].astype(np.uint8), (m * 255).astype(np.uint8)]), 'RGBA').save(f'view{k}.png')
print(len(boxes), 'views')
