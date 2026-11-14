"""Challenge generators/verifiers for the three tiers."""
import hashlib
import math
import random
import secrets
import string

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CHARSET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L

# What each tier asks of the visitor / enforces on the server.
TIER_CFG = {
    "low":    dict(captcha=None,     length=0, pow_bits=12, min_wait=0.0, max_attempts=3,
                   label="Lightweight check (invisible proof-of-work + honeypot)"),
    "medium": dict(captcha="medium", length=5, pow_bits=0,  min_wait=1.5, max_attempts=3,
                   label="Conventional text CAPTCHA"),
    "high":   dict(captcha="high",   length=7, pow_bits=18, min_wait=3.0, max_attempts=2,
                   label="Harder CAPTCHA + proof-of-work + lockout/rate limiting"),
}


def new_text(length):
    return "".join(random.SystemRandom().choices(CHARSET, k=length))


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_captcha(text, level="medium"):
    hard = level == "high"
    width, height = (280, 80) if hard else (200, 70)
    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    font = _font(38)
    x, step, rot = 15, (34 if hard else 32), (35 if hard else 25)
    for ch in text:
        glyph = Image.new("RGBA", (56, 64), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((5, 5), ch, font=font, fill=(20, 33, 61, 255))
        glyph = glyph.rotate(random.randint(-rot, rot), expand=True)
        img.paste(glyph, (x, random.randint(0, 14)), glyph)
        x += step
    for _ in range(9 if hard else 5):
        draw.line([(random.randint(0, width), random.randint(0, height)),
                   (random.randint(0, width), random.randint(0, height))],
                  fill=(224, 122, 31), width=2)
    for _ in range(300 if hard else 120):
        draw.point((random.randint(0, width), random.randint(0, height)), fill=(90, 90, 90))
    if hard:  # sinusoidal wave distortion
        src, out = img.copy(), Image.new("RGB", img.size, (245, 245, 240))
        amp, per, ph = random.uniform(2.5, 4), random.uniform(25, 45), random.uniform(0, 6.28)
        for yy in range(height):
            off = int(amp * math.sin(2 * math.pi * yy / per + ph))
            out.paste(src.crop((0, yy, width, yy + 1)), (off, yy))
        img = out
    return img.filter(ImageFilter.GaussianBlur(0.6))


# ---- proof of work: find counter so that sha256(f"{nonce}:{counter}") has `bits` leading zero bits
def make_pow(bits):
    return {"nonce": secrets.token_hex(12), "bits": bits}


def leading_zero_bits(digest: bytes) -> int:
    n = 0
    for b in digest:
        if b == 0:
            n += 8
            continue
        n += 8 - b.bit_length()
        break
    return n


def verify_pow(nonce, counter, bits):
    if bits <= 0:
        return True
    try:
        counter = int(counter)
    except (TypeError, ValueError):
        return False
    if counter < 0 or counter > 2 ** 40:
        return False
    return leading_zero_bits(hashlib.sha256(f"{nonce}:{counter}".encode()).digest()) >= bits


def solve_pow(nonce, bits):  # used by tests / bots
    c = 0
    while not verify_pow(nonce, c, bits):
        c += 1
    return c
