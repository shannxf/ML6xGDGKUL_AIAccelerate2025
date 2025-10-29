import io
import os
import base64
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List, Union

try:
    from PIL import Image
except Exception:
    Image = None  # thumbnail generation will be unavailable

try:
    import moviepy.editor as mpy  # type: ignore
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False

import urllib.request
import urllib.error

# --- Utils ---

def _fetch_bytes_from_url(url: str, max_bytes: int = 50_000_000, timeout: int = 15) -> bytes:
    """Fetch raw bytes from a URL with a size limit. Accept any content-type.

    This is intentionally permissive because some video hosts use non-standard content-types.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunks = []
            total = 0
            while True:
                chunk = resp.read(16384)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Remote file exceeds maximum allowed size of {max_bytes} bytes")
            return b"".join(chunks)
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to fetch URL: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to fetch URL: {e.reason}") from e


def _write_temp_file(data: bytes, suffix: str = ".mp4") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.write(data)
        f.flush()
        return f.name
    finally:
        f.close()


@dataclass
class VideoInfo:
    width: Optional[int]
    height: Optional[int]
    duration: Optional[float]
    fps: Optional[float]
    format: Optional[str]
    size_bytes: int
    md5: str


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _frame_to_data_url(frame, fmt: str = "PNG") -> str:
    if Image is None:
        raise RuntimeError("Pillow required to create thumbnails")
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b = buf.getvalue()
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b).decode("utf-8")


def inspect_video_source(
    *,
    path: Optional[str] = None,
    data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Inspect a video file given either a local path or raw bytes.

    Returns a dict with metadata and a placeholder thumbnail (data URL) when possible.
    """
    if data is not None:
        size_bytes = len(data)
        md5 = _md5_bytes(data)
        file_path = _write_temp_file(data)
    elif path is not None:
        size_bytes = os.path.getsize(path)
        with open(path, "rb") as f:
            md5 = _md5_bytes(f.read())
        file_path = path
    else:
        raise ValueError("Provide path or data")

    info = {
        "width": None,
        "height": None,
        "duration": None,
        "fps": None,
        "format": None,
        "size_bytes": size_bytes,
        "md5": md5,
        "thumbnail_data_url": None,
    }

    try:
        if _HAS_MOVIEPY:
            clip = mpy.VideoFileClip(file_path)
            info.update({
                "width": int(clip.w) if clip.w else None,
                "height": int(clip.h) if clip.h else None,
                "duration": float(clip.duration) if clip.duration else None,
                "fps": float(clip.fps) if clip.fps else None,
                "format": os.path.splitext(file_path)[1].lstrip('.') or None,
            })
            # generate thumbnail at t=0.5s or at 0 if shorter
            t = 0.5 if (clip.duration and clip.duration > 0.5) else 0
            try:
                frame = clip.get_frame(t)
                info["thumbnail_data_url"] = _frame_to_data_url(frame, fmt="PNG")
            except Exception:
                pass
            clip.reader.close()
            if hasattr(clip, "close"):
                clip.close()
        else:
            # Attempt to use ffprobe (if available) via subprocess
            try:
                cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name",
                    "-of", "json", file_path
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                import json

                parsed = json.loads(out)
                stream = parsed.get("streams", [])[0] if parsed.get("streams") else {}
                width = stream.get("width")
                height = stream.get("height")
                duration = float(stream.get("duration")) if stream.get("duration") else None
                # r_frame_rate is like "30/1"
                fps = None
                if stream.get("r_frame_rate") and "/" in stream.get("r_frame_rate"):
                    num, den = stream.get("r_frame_rate").split("/")
                    try:
                        fps = float(num) / float(den)
                    except Exception:
                        fps = None
                info.update({
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "fps": fps,
                    "format": stream.get("codec_name"),
                })
            except Exception:
                # best-effort only
                pass
    finally:
        if data is not None and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass

    return info


def video_ops(
    operation: str,
    video_b64: Optional[str] = None,
    video_url: Optional[str] = None,
    path: Optional[str] = None,
    max_duration: Optional[float] = None,
    max_megapixels: Optional[float] = None,
) -> Dict[str, Any]:
    """Tool interface for video handling.

    Supported operations: 'inspect', 'validate', 'thumbnails', 'extract_frame'
    """
    if not any([video_b64, video_url, path]):
        raise ValueError("Provide one of: video_b64, video_url, or path")

    data = None
    if video_b64:
        if video_b64.startswith("data:"):
            video_b64 = video_b64.split(",", 1)[1]
        data = base64.b64decode(video_b64)
    elif video_url:
        data = _fetch_bytes_from_url(video_url)

    if operation == "inspect":
        info = inspect_video_source(path=path, data=data)
        return info
    elif operation == "validate":
        info = inspect_video_source(path=path, data=data)
        reasons = []
        if max_duration is not None and info.get("duration") is not None and info.get("duration") > float(max_duration):
            reasons.append(f"duration {info.get('duration')}s > {max_duration}s")
        if max_megapixels is not None and info.get("width") and info.get("height"):
            mp = (info.get("width") * info.get("height")) / 1_000_000.0
            if mp > float(max_megapixels):
                reasons.append(f"video too large: {mp:.2f} MP > {max_megapixels} MP")
        return {"ok": len(reasons) == 0, "reasons": reasons, "info": info}
    elif operation == "thumbnails":
        info = inspect_video_source(path=path, data=data)
        # thumbnail already attempted in inspect
        return {"thumbnail": info.get("thumbnail_data_url"), "info": info}
    elif operation == "extract_frame":
        # extract a specific frame/time if requested via path/url/data; for now return frame at 0.5s
        if data is not None:
            tmp = _write_temp_file(data)
            file_path = tmp
        else:
            file_path = path
        try:
            if _HAS_MOVIEPY:
                clip = mpy.VideoFileClip(file_path)
                t = 0.5 if (clip.duration and clip.duration > 0.5) else 0
                frame = clip.get_frame(t)
                thumb = _frame_to_data_url(frame)
                clip.reader.close()
                if hasattr(clip, "close"):
                    clip.close()
                return {"ok": True, "thumbnail": thumb}
            else:
                return {"ok": False, "error": "moviepy not installed; install moviepy and ffmpeg to enable frame extraction"}
        finally:
            if data is not None and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
    else:
        raise ValueError(f"Unknown operation: {operation}")


tool = {
    "name": "video_ops",
    "description": "Handle user-uploaded videos: inspect metadata, validate (duration/resolution), generate thumbnails, and extract frames. Requires moviepy/ffmpeg for advanced operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["inspect","validate","thumbnails","extract_frame"]},
            "video_b64": {"type": "string", "description": "Base64 data (optionally a data URL)."},
            "video_url": {"type": "string", "description": "HTTP/HTTPS URL to fetch video from."},
            "path": {"type": "string", "description": "Path to a local video file."},
            "max_duration": {"type": "number"},
            "max_megapixels": {"type": "number"}
        },
        "required": ["operation"],
    },
    "fn": video_ops,
}
