
import io
import os
import base64
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List, Union

try:
    from PIL import Image, ImageOps, ImageFilter, ExifTags
except Exception as e:
    raise ImportError("Pillow is required. Install with `pip install pillow`") from e

# Optional extras
try:
    import pytesseract  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

# --- Utils ---

def _img_to_bytes(img: Image.Image, fmt: Optional[str] = None, quality: Optional[int] = None, keep_alpha: bool = True) -> bytes:
    fmt = (fmt or img.format or "PNG").upper()
    buf = io.BytesIO()
    save_params = {}
    if fmt in {"JPEG", "JPG"}:
        # JPEG cannot store alpha; convert if needed
        if keep_alpha and img.mode in ("RGBA", "LA"):
            # flatten on white background
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.getchannel("A"))
            img_to_save = bg
        else:
            img_to_save = img.convert("RGB")
        if quality is not None:
            save_params["quality"] = int(max(1, min(quality, 95)))
        save_params["optimize"] = True
        img_to_save.save(buf, format="JPEG", **save_params)
    else:
        # Propagate quality for formats that support it (e.g., WEBP, AVIF if compiled, JPEG2000)
        if quality is not None and fmt in {"WEBP", "JP2", "JPEG2000"}:
            save_params["quality"] = int(max(1, min(quality, 100)))
        img.save(buf, format=fmt, **save_params)
    return buf.getvalue()

def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGBA")

def _safe_format(fmt: Optional[str]) -> Optional[str]:
    return (fmt.upper() if fmt else None)

def _calc_megapixels(w: int, h: int) -> float:
    return (w * h) / 1_000_000.0

def _ahash(img: Image.Image, hash_size: int = 8) -> str:
    # Perceptual average hash: grayscale -> resize -> compare to mean
    g = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    # pack into hex
    width = (hash_size * hash_size)
    return f"{int(bits, 2):0{width//4}x}"

def _exif_dict(img: Image.Image) -> Dict[str, Any]:
    exif = {}
    try:
        raw = img.getexif()
        if raw is None:
            return {}
        for (k, v) in raw.items():
            tag = ExifTags.TAGS.get(k, str(k))
            exif[tag] = v
    except Exception:
        pass
    return exif

def _auto_orient(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def _strip_metadata(img: Image.Image) -> Image.Image:
    # Create a fresh image and drop all info/exif
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    return clean

def _placeholder_data_url(img: Image.Image, size: int = 16, blur_radius: int = 1) -> str:
    tiny = img.copy()
    tiny.thumbnail((size, size), Image.Resampling.LANCZOS)
    tiny = tiny.filter(ImageFilter.GaussianBlur(blur_radius))
    b = _img_to_bytes(tiny, "PNG")
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")

# --- Public API ---

@dataclass
class ImageInfo:
    width: int
    height: int
    mode: str
    format: Optional[str]
    megapixels: float
    md5: str
    ahash: str
    exif: Dict[str, Any]

def inspect_image(data: Union[bytes, Image.Image]) -> ImageInfo:
    """Return normalized metadata, hashes, EXIF, and safe megapixel count."""
    if isinstance(data, Image.Image):
        img = data
    else:
        img = _bytes_to_image(data)
    img = _auto_orient(img)
    w, h = img.size
    mp = _calc_megapixels(w, h)
    # hashes
    raw_bytes = _img_to_bytes(img, fmt="PNG")  # stable for md5 calculation
    md5 = hashlib.md5(raw_bytes).hexdigest()
    ah = _ahash(img)
    info = ImageInfo(
        width=w,
        height=h,
        mode=img.mode,
        format=_safe_format(img.format),
        megapixels=mp,
        md5=md5,
        ahash=ah,
        exif=_exif_dict(img),
    )
    return info

def validate_image(
    data: Union[bytes, Image.Image],
    allowed_formats: Optional[List[str]] = None,
    max_megapixels: Optional[float] = 64.0,
) -> Dict[str, Any]:
    """Validate format & max resolution; returns {ok, reasons[]}."""
    allowed = set([f.upper() for f in (allowed_formats or ["PNG","JPEG","JPG","WEBP","GIF"])])  # no HEIC by default
    info = inspect_image(data)
    reasons = []
    if info.format and info.format.upper() not in allowed:
        reasons.append(f"format {info.format} not in allowed {sorted(allowed)}")
    if max_megapixels is not None and info.megapixels > float(max_megapixels):
        reasons.append(f"image too large: {info.megapixels:.2f} MP > {max_megapixels} MP")
    return {"ok": len(reasons) == 0, "reasons": reasons, "info": asdict(info)}

def transform_image(
    data: Union[bytes, Image.Image],
    op: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fmt: Optional[str] = None,
    quality: Optional[int] = None,
    keep_alpha: bool = True,
    background: Optional[Tuple[int,int,int]] = None,
) -> Dict[str, Any]:
    """
    Basic transforms:
      - resize_contain: fit within (width,height) keeping aspect (letterbox if background set)
      - resize_cover: fill (width,height) and crop overflow (cover)
      - crop_center: crop to (width,height) from center
      - strip_metadata: remove EXIF and ancillary metadata
      - convert: only change format/quality/alpha handling
    Returns: { "image_b64": str, "data_url": str, "format": str, "width": int, "height": int }
    """
    if isinstance(data, Image.Image):
        img = data
    else:
        img = _bytes_to_image(data)
    img = _auto_orient(img)

    if op == "strip_metadata":
        img = _strip_metadata(img)

    if op in {"resize_contain","resize_cover","crop_center"}:
        if not width or not height:
            raise ValueError("width and height are required for this op")
        if op == "resize_contain":
            # thumbnail keeps aspect; paste on background if provided
            canvas = None
            if background is not None:
                mode = "RGBA" if (img.mode == "RGBA" or keep_alpha) else "RGB"
                canvas = Image.new(mode, (width, height), background + ((0,) if mode=="RGBA" else ()))
            tmp = img.copy()
            tmp.thumbnail((width, height), Image.Resampling.LANCZOS)
            if canvas is not None:
                x = (width - tmp.size[0]) // 2
                y = (height - tmp.size[1]) // 2
                canvas.paste(tmp, (x, y), tmp if tmp.mode == "RGBA" else None)
                img = canvas
            else:
                img = tmp
        elif op == "resize_cover":
            img = ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS, centering=(0.5,0.5))
        elif op == "crop_center":
            w, h = img.size
            if width > w or height > h:
                raise ValueError("crop size larger than image")
            left = (w - width) // 2
            top = (h - height) // 2
            img = img.crop((left, top, left + width, top + height))

    # If op is convert or we specified fmt/quality, encode now
    out_fmt = (fmt or img.format or "PNG").upper()
    out_bytes = _img_to_bytes(img, out_fmt, quality=quality, keep_alpha=keep_alpha)
    out_b64 = base64.b64encode(out_bytes).decode("utf-8")
    data_url = f"data:image/{out_fmt.lower()};base64,{out_b64}"
    return {"image_b64": out_b64, "data_url": data_url, "format": out_fmt, "width": img.size[0], "height": img.size[1]}

def thumbnail_set(
    data: Union[bytes, Image.Image],
    sizes: List[Tuple[int,int]] = [(64,64),(128,128),(256,256),(512,512)],
    fmt: str = "WEBP",
    quality: int = 80,
) -> Dict[str, Any]:
    """Generate a responsive thumbnail set. Returns {items:[{width,height,format,data_url}], placeholder_data_url}"""
    if isinstance(data, Image.Image):
        base = data
    else:
        base = _bytes_to_image(data)
    base = _auto_orient(base)
    items = []
    for (w, h) in sizes:
        tmp = ImageOps.fit(base, (w, h), Image.Resampling.LANCZOS, centering=(0.5,0.5))
        b = _img_to_bytes(tmp, fmt=fmt, quality=quality)
        items.append({
            "width": w,
            "height": h,
            "format": fmt.upper(),
            # JSON-safe: do not return raw bytes; provide a data URL
            "data_url": "data:image/%s;base64,%s" % (fmt.lower(), base64.b64encode(b).decode("utf-8")),
        })
    placeholder = _placeholder_data_url(base, size=16, blur_radius=1)
    return {"items": items, "placeholder_data_url": placeholder}

def ocr_text(
    data: Union[bytes, Image.Image],
    lang: str = "eng",
) -> Dict[str, Any]:
    """Extract text with Tesseract if available. Returns {ok, text, error?}"""
    if not _HAS_TESS:
        return {"ok": False, "text": "", "error": "pytesseract not installed"}
    if isinstance(data, Image.Image):
        img = data
    else:
        img = _bytes_to_image(data)
    img = _auto_orient(img)
    try:
        txt = pytesseract.image_to_string(img, lang=lang)
        return {"ok": True, "text": txt}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e)}

# --- LLM tool wrapper ---

def image_ops(
    operation: str,
    image_b64: Optional[str] = None,
    path: Optional[str] = None,
    # validate
    allowed_formats: Optional[List[str]] = None,
    max_megapixels: Optional[float] = 64.0,
    # transform
    op: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fmt: Optional[str] = None,
    quality: Optional[int] = None,
    keep_alpha: bool = True,
    background: Optional[List[int]] = None,
    # ocr
    lang: str = "eng",
) -> Dict[str, Any]:
    """
    Unified entry point for agent tooling. Provide either image_b64 (data URL or base64) or path.
    """
    if not image_b64 and not path:
        raise ValueError("Provide either image_b64 or path")
    if image_b64:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        data = base64.b64decode(image_b64)
    else:
        with open(path, "rb") as f:
            data = f.read()

    if operation == "inspect":
        info = inspect_image(data)
        payload = asdict(info)
        payload["placeholder_data_url"] = _placeholder_data_url(_bytes_to_image(data))
        return payload
    elif operation == "validate":
        return validate_image(data, allowed_formats=allowed_formats, max_megapixels=max_megapixels)
    elif operation == "transform":
        if not op:
            raise ValueError("transform requires 'op'")
        bg_tuple: Optional[Tuple[int, int, int]] = None
        if background is not None:
            if not isinstance(background, list) or len(background) != 3:
                raise ValueError("background must be a list of three integers [R,G,B]")
            bg_tuple = (int(background[0]), int(background[1]), int(background[2]))
        return transform_image(
            data, op=op, width=width, height=height, fmt=fmt, quality=quality, keep_alpha=keep_alpha, background=bg_tuple
        )
    elif operation == "thumbnails":
        return thumbnail_set(data)
    elif operation == "ocr":
        return ocr_text(data, lang=lang)
    else:
        raise ValueError(f"Unknown operation: {operation}")

tool = {
    "name": "image_ops",
    "description": "Handle user-uploaded images: inspect, validate, transform (resize/crop/convert), generate thumbnails, OCR (optional).",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["inspect","validate","transform","thumbnails","ocr"]},
            "image_b64": {"type": "string", "description": "Base64 data (optionally a data URL)."},
            "path": {"type": "string", "description": "Path to an image file."},
            "allowed_formats": {"type": "array", "items": {"type": "string"}},
            "max_megapixels": {"type": "number"},
            "op": {"type": "string", "enum": ["resize_contain","resize_cover","crop_center","strip_metadata","convert"]},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "fmt": {"type": "string"},
            "quality": {"type": "integer"},
            "keep_alpha": {"type": "boolean"},
            "background": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "RGB tuple for letterboxing background in resize_contain."
            },
            "lang": {"type": "string", "description": "OCR language code, e.g., 'eng'."}
        },
        "required": ["operation"],
    },
    "fn": image_ops,
}
