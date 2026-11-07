"""Генератор текстур для Lab 3.

Запуск (из папки Lab3):  python tools/make_textures.py
Нужны пакеты numpy и pillow:  pip install numpy pillow

Все текстуры бесшовные (их можно повторять по поверхности) и лежат в textures/.
Для каждого материала создаются карты:
  *_albedo.png    -- основной цвет (sRGB)
  *_specular.png  -- где поверхность блестит (белое = блестит, чёрное = матовая)
  *_emissive.png  -- где поверхность светится (есть только у lava)
"""
import os
import numpy as np
from PIL import Image

SIZE = 512
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def save(name, array):
    """array: float в [0, 1], форма (H, W) или (H, W, 3)."""
    data = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(data).save(os.path.join(OUT, name))
    print("saved", name)


def perlin(period, size=SIZE):
    """Бесшовный шум Перлина: period x period ячеек, значения примерно в [-1, 1]."""
    angles = rng.uniform(0.0, 2.0 * np.pi, (period, period))
    grad = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    coord = (np.arange(size) + 0.5) / size * period
    gx, gy = np.meshgrid(coord, coord)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0

    def dot_grad(ix, iy, dx, dy):
        g = grad[iy % period, ix % period]  # % period -- делает шум бесшовным
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot_grad(x0, y0, fx, fy)
    n10 = dot_grad(x0 + 1, y0, fx - 1, fy)
    n01 = dot_grad(x0, y0 + 1, fx, fy - 1)
    n11 = dot_grad(x0 + 1, y0 + 1, fx - 1, fy - 1)
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)  # плавная интерполяция
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    top = n00 * (1 - u) + n10 * u
    bottom = n01 * (1 - u) + n11 * u
    return (top * (1 - v) + bottom * v) * 1.4


def fbm(base_period, octaves=5):
    """Сумма шумов разной частоты: нормирована в [0, 1]."""
    total = np.zeros((SIZE, SIZE))
    amplitude = 1.0
    norm = 0.0
    for i in range(octaves):
        total += amplitude * perlin(base_period * 2 ** i)
        norm += amplitude
        amplitude *= 0.5
    return np.clip(0.5 + 0.5 * total / norm * 1.6, 0.0, 1.0)


def mix(a, b, t):
    return a * (1 - t[..., None]) + b * t[..., None]


# ---------------------------------------------------------------- bricks
def bricks():
    rows, brick_w = 8, SIZE // 4
    brick_h = SIZE // rows
    mortar = 8
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    row = y // brick_h
    xs = (x + (row % 2) * (brick_w // 2)) % SIZE
    col = xs // brick_w
    in_x = xs % brick_w
    in_y = y % brick_h
    is_mortar = (in_x < mortar // 2) | (in_x >= brick_w - mortar // 2) | \
                (in_y < mortar // 2) | (in_y >= brick_h - mortar // 2)

    per_brick = rng.uniform(0.75, 1.15, (rows, 4))[row, col]
    noise = fbm(8)
    brick_color = np.array([0.62, 0.22, 0.15])
    mortar_color = np.array([0.62, 0.60, 0.55])
    albedo = brick_color * (per_brick * (0.75 + 0.5 * noise))[..., None]
    albedo[is_mortar] = (mortar_color * (0.7 + 0.5 * noise)[..., None])[is_mortar]
    spec = np.where(is_mortar, 0.0, 0.35 + 0.25 * noise)
    save("bricks_albedo.png", albedo)
    save("bricks_specular.png", spec)


# ---------------------------------------------------------------- checker
def checker():
    cell = SIZE // 8
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    white = ((x // cell + y // cell) % 2) == 0
    noise = fbm(16, 3)
    light = np.array([0.88, 0.86, 0.80]) * (0.9 + 0.1 * noise)[..., None]
    dark = np.array([0.10, 0.12, 0.16]) * (0.9 + 0.2 * noise)[..., None]
    albedo = np.where(white[..., None], light, dark)
    # светлые плитки глянцевые, тёмные почти матовые
    spec = np.where(white, 1.0, 0.25)
    save("checker_albedo.png", albedo)
    save("checker_specular.png", spec)


# ---------------------------------------------------------------- wood
def wood():
    planks = 4
    plank_w = SIZE // planks
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    plank = x // plank_w
    in_x = x % plank_w
    tone = rng.uniform(0.8, 1.15, planks)[plank]
    # вытянутые вдоль Y годовые кольца: шум, сильно растянутый по вертикали
    grain = fbm(4, 4)
    stretch = np.sin((in_x / plank_w * 6.0 + grain * 5.0) * np.pi)
    rings = 0.5 + 0.5 * stretch
    base = np.array([0.55, 0.34, 0.16])
    dark = np.array([0.33, 0.18, 0.08])
    albedo = mix(base, dark, 0.55 * rings + 0.25 * grain) * tone[..., None]
    seam = (in_x < 3) | (in_x >= plank_w - 3)
    albedo[seam] *= 0.35
    spec = np.where(seam, 0.0, 0.45 + 0.2 * rings)
    save("wood_albedo.png", albedo)
    save("wood_specular.png", spec)


# ---------------------------------------------------------------- lava
def lava():
    rock = fbm(6, 5)
    # "жилки": там, где шум близок к 0.5, образуется узкая светящаяся трещина
    cracks_noise = fbm(4, 4)
    ridge = 1.0 - np.abs(cracks_noise - 0.5) * 2.0   # 1 на линии трещины, 0 вдали
    crack = np.clip((ridge - 0.86) / 0.14, 0.0, 1.0) ** 1.2

    rock_color = np.array([0.20, 0.18, 0.17])
    albedo = rock_color * (0.5 + 0.9 * rock)[..., None]
    albedo = albedo * (1.0 - 0.8 * crack[..., None])  # внутри трещин камень темнее

    # emissive: оранжево-жёлтое свечение только в трещинах
    hot = np.stack([np.ones_like(crack), 0.25 + 0.55 * crack, 0.03 + 0.15 * crack ** 3], axis=-1)
    emissive = hot * crack[..., None]

    spec = 0.2 + 0.25 * rock
    save("lava_albedo.png", albedo)
    save("lava_specular.png", spec)
    save("lava_emissive.png", emissive)


if __name__ == "__main__":
    bricks()
    checker()
    wood()
    lava()
