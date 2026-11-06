"""Japanese OCR adapters and balloon detection.

Detection is a CPU heuristic (bright regions closed off from the page edge). It is not a
segmentation model. Text recognition uses an installed backend only: Tesseract, RapidOCR, or
EasyOCR. There is no fake recognizer in the default path. Tests pass a fixture backend that
reads text embedded in the PNG and is labeled as not real OCR.
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import shutil
import subprocess

from .safety import OcrUnavailable

REVIEW_FLAGS = {
    "placement-uncertain", "ocr-empty", "low-confidence", "inpaint-flat-plate",
    "ocr-failed", "ocr-backend-error",
}


def needs_review(confidence: float | None, flags: list[str]) -> bool:
    if confidence is not None and confidence < 0.55:
        return True
    return any(f in REVIEW_FLAGS for f in flags)


def direction_of(w: int, h: int) -> str:
    return "vertical" if h >= w * 1.25 else "horizontal"


def order_regions(regions: list[dict], rtl: bool = True) -> list[dict]:
    """Manga reading order: rows from top to bottom, and right-to-left inside a row when rtl."""
    items = [dict(r) for r in regions]
    if not items:
        return []
    items.sort(key=lambda r: (r["y"] + r["h"] / 2.0, r["x"]))
    rows: list[list] = []
    for region in items:
        cy = region["y"] + region["h"] / 2.0
        if rows and abs(cy - rows[-1][0]) <= max(rows[-1][1], region["h"]) * 0.65:
            rows[-1][2].append(region)
        else:
            rows.append([cy, region["h"], [region]])
    ordered = []
    for _cy, _h, group in rows:
        group.sort(key=lambda r: r["x"], reverse=bool(rtl))
        ordered.extend(group)
    for i, region in enumerate(ordered):
        region["idx"] = i
    return ordered


def detect_balloons(image, rtl: bool = True) -> list[dict]:
    """Find outlined bright blobs. `image` is a Pillow image. Boxes are in original pixels."""
    from PIL import Image
    im = image.convert("RGB")
    width, height = im.size
    scale = 1.0
    max_edge = 1000
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        small = im.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
    else:
        small = im
    gray = small.convert("L")
    gw, gh = gray.size
    pix = gray.tobytes()
    total = gw * gh
    white = bytearray(1 if pix[i] >= 242 else 0 for i in range(total))
    seen = bytearray(total)
    min_area = max(80, int(total * 0.002))
    max_area = int(total * 0.45)
    boxes = []
    for start in range(total):
        if not white[start] or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        count = 0
        minx = maxx = start % gw
        miny = maxy = start // gw
        touches = False
        while stack:
            i = stack.pop()
            x, y = i % gw, i // gw
            count += 1
            if x < minx:
                minx = x
            elif x > maxx:
                maxx = x
            if y < miny:
                miny = y
            elif y > maxy:
                maxy = y
            if x == 0 or y == 0 or x == gw - 1 or y == gh - 1:
                touches = True
            for j, ok in ((i - 1, x > 0), (i + 1, x + 1 < gw), (i - gw, y > 0), (i + gw, y + 1 < gh)):
                if ok and white[j] and not seen[j]:
                    seen[j] = 1
                    stack.append(j)
        if touches or count < min_area or count > max_area:
            continue
        bw, bh = maxx - minx + 1, maxy - miny + 1
        if bw < 8 or bh < 8 or bw / bh > 8 or bh / bw > 8:
            continue
        inv = 1.0 / scale
        x0 = max(0, min(width - 1, int(round(minx * inv))))
        y0 = max(0, min(height - 1, int(round(miny * inv))))
        x1 = max(x0 + 1, min(width, int(round((maxx + 1) * inv))))
        y1 = max(y0 + 1, min(height, int(round((maxy + 1) * inv))))
        flags = []
        if x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1:
            flags.append("placement-uncertain")
        if (x1 - x0) * (y1 - y0) > 0.30 * width * height:
            flags.append("placement-uncertain")
        boxes.append({
            "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "direction": direction_of(x1 - x0, y1 - y0),
            "confidence": 0.8, "flags": flags,
        })
    return order_regions(boxes, rtl=rtl)


def gpu_info() -> dict:
    """Report whether a CUDA provider is importable. Never required."""
    info = {"available": False, "providers": [], "selected": "cpu",
            "note": "GPU 공급자가 없어 OCR은 CPU로 실행합니다."}
    forced = os.environ.get("EPUB_OCR_GPU", "auto").strip().lower()
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
        info["providers"].extend(providers)
        if "CUDAExecutionProvider" in providers and forced != "cpu":
            info["available"] = True
            info["selected"] = "cuda"
            info["note"] = "onnxruntime CUDA 공급자를 사용합니다."
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["providers"].append("torch-cuda")
            if forced != "cpu" and info["selected"] != "cuda":
                info["available"] = True
                info["selected"] = "cuda"
                info["note"] = "torch CUDA를 사용할 수 있습니다."
    except Exception:
        pass
    if forced == "cpu":
        info["selected"] = "cpu"
        info["note"] = "EPUB_OCR_GPU=cpu 라서 CPU만 사용합니다."
    return info


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def tesseract_langs() -> set[str]:
    if not shutil.which("tesseract"):
        return set()
    proc = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {line.strip() for line in text.splitlines() if line.strip() and "list" not in line.lower()}


def available_backends() -> list[str]:
    found = []
    langs = tesseract_langs()
    if "jpn" in langs:
        found.append("tesseract")
    if _has_module("rapidocr_onnxruntime"):
        found.append("rapidocr")
    if _has_module("easyocr"):
        found.append("easyocr")
    return found


def normalize_ocr_text(text: str) -> str:
    text = (text or "").replace("\u3000", " ").strip()
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text):
        return re.sub(r"\s+", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


class OcrBackend:
    id = "base"
    real = True
    label = "ocr"

    def recognize(self, image, direction: str) -> tuple[str, float, list[str]]:
        raise NotImplementedError


class FixtureBackend(OcrBackend):
    """Test-only. Reads PNG text metadata. This is not optical character recognition."""
    id = "fixture"
    real = False
    label = "fixture (테스트 전용, 글자 인식이 아님)"

    def recognize(self, image, direction: str):
        raise OcrUnavailable("fixture 백엔드는 영역 단위 인식 함수로 호출하지 않습니다.")


class TesseractBackend(OcrBackend):
    id = "tesseract"
    real = True
    label = "tesseract (CPU)"

    def recognize(self, image, direction: str):
        from PIL import Image
        lang = "jpn_vert" if direction == "vertical" and "jpn_vert" in tesseract_langs() else "jpn"
        psm = "5" if direction == "vertical" else "6"
        im = image.convert("RGB")
        if min(im.size) < 40:
            im = im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST)
        pad = Image.new("RGB", (im.width + 16, im.height + 16), (255, 255, 255))
        pad.paste(im, (8, 8))
        import io
        buf = io.BytesIO()
        pad.save(buf, format="PNG")
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", lang, "--psm", psm, "--oem", "1", "tsv"],
            input=buf.getvalue(), capture_output=True,
        )
        if proc.returncode != 0:
            return "", 0.0, ["ocr-failed"]
        words, confs = [], []
        for line in proc.stdout.decode("utf-8", "replace").splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 12:
                continue
            try:
                conf = float(cols[10])
            except ValueError:
                continue
            token = cols[11].strip()
            if conf < 0 or not token:
                continue
            words.append(token)
            confs.append(conf)
        text = normalize_ocr_text("".join(words))
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        flags = [] if text else ["ocr-empty"]
        return text, conf, flags


class RapidOcrBackend(OcrBackend):
    id = "rapidocr"
    real = True
    _engine = None

    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        self.label = "rapidocr (" + ("cuda" if use_gpu else "cpu") + ")"

    def recognize(self, image, direction: str):
        import tempfile
        from rapidocr_onnxruntime import RapidOCR
        if RapidOcrBackend._engine is None:
            RapidOcrBackend._engine = RapidOCR()
        fd, name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.convert("RGB").save(name)
            result, _elapsed = RapidOcrBackend._engine(name)
        finally:
            os.remove(name)
        if not result:
            return "", 0.0, ["ocr-empty"]
        texts, confs = [], []
        for item in result:
            texts.append(str(item[1]))
            confs.append(float(item[2]))
        return normalize_ocr_text("".join(texts)), (sum(confs) / len(confs) if confs else 0.0), []


class EasyOcrBackend(OcrBackend):
    id = "easyocr"
    real = True
    _reader = None

    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        self.label = "easyocr (" + ("cuda" if use_gpu else "cpu") + ")"

    def recognize(self, image, direction: str):
        import tempfile
        import easyocr
        if EasyOcrBackend._reader is None:
            EasyOcrBackend._reader = easyocr.Reader(["ja"], gpu=self.use_gpu)
        fd, name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.convert("RGB").save(name)
            result = EasyOcrBackend._reader.readtext(name)
        finally:
            os.remove(name)
        if not result:
            return "", 0.0, ["ocr-empty"]
        texts = [str(item[1]) for item in result]
        confs = [float(item[2]) for item in result]
        return normalize_ocr_text("".join(texts)), sum(confs) / len(confs), []


def make_backend(name: str, gpu: dict | None = None) -> OcrBackend:
    gpu = gpu or gpu_info()
    name = (name or "auto").lower()
    if name == "fixture":
        return FixtureBackend()
    if name == "mock":
        raise OcrUnavailable("OCR mock 백엔드는 없습니다. 테스트는 fixture를 쓰고, 실제 인식은 tesseract/rapidocr/easyocr를 설치하세요.")
    avail = available_backends()
    if name == "auto":
        if gpu.get("selected") == "cuda" and "rapidocr" in avail:
            name = "rapidocr"
        elif gpu.get("selected") == "cuda" and "easyocr" in avail:
            name = "easyocr"
        elif "tesseract" in avail:
            name = "tesseract"
        elif "rapidocr" in avail:
            name = "rapidocr"
        elif "easyocr" in avail:
            name = "easyocr"
        else:
            raise OcrUnavailable(
                "일본어 OCR 백엔드가 없습니다. Tesseract(jpn) 또는 선택 패키지 rapidocr-onnxruntime / easyocr 를 설치하세요. "
                "소설 번역만 할 때는 이 패키지가 필요 없습니다."
            )
    if name == "tesseract":
        if "tesseract" not in avail and "jpn" not in tesseract_langs():
            raise OcrUnavailable("tesseract와 jpn 언어 데이터가 없습니다.")
        return TesseractBackend()
    if name == "rapidocr":
        if not _has_module("rapidocr_onnxruntime"):
            raise OcrUnavailable("rapidocr-onnxruntime 이 설치되어 있지 않습니다.")
        return RapidOcrBackend(gpu.get("selected") == "cuda")
    if name == "easyocr":
        if not _has_module("easyocr"):
            raise OcrUnavailable("easyocr 가 설치되어 있지 않습니다. 첫 실행은 모델을 받으며 CPU/GPU는 설치 상태에 따릅니다.")
        return EasyOcrBackend(gpu.get("selected") == "cuda")
    raise OcrUnavailable(f"알 수 없는 OCR 백엔드입니다: {name}")


def read_fixture(path: str) -> list[dict] | None:
    from PIL import Image
    with Image.open(path) as im:
        raw = im.info.get("epub-ocr-fixture")
    if not raw:
        return None
    data = json.loads(raw)
    if isinstance(data, dict):
        return list(data.get("regions") or [])
    if isinstance(data, list):
        return data
    return None


def recognize_page(path: str, backend: OcrBackend, rtl: bool = True) -> tuple[list[dict], dict]:
    """Open one page, detect or load fixture regions, close it before returning."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        detected = detect_balloons(im, rtl=rtl)
        if isinstance(backend, FixtureBackend):
            spec = read_fixture(path) or []
            regions = []
            for item in spec:
                flags = list(item.get("flags") or [])
                conf = float(item.get("confidence", 0.9))
                text = str(item.get("text") or "")
                if conf < 0.55:
                    flags.append("low-confidence")
                if not text.strip():
                    flags.append("ocr-empty")
                regions.append({
                    "x": int(item["x"]), "y": int(item["y"]), "w": int(item["w"]), "h": int(item["h"]),
                    "direction": item.get("direction") or direction_of(int(item["w"]), int(item["h"])),
                    "ocr_text": text, "confidence": conf, "flags": flags,
                    "review": 1 if needs_review(conf, flags) else 0,
                })
            if len(detected) != len(regions):
                for region in regions:
                    if "detection-count-mismatch" not in region["flags"]:
                        region["flags"].append("detection-count-mismatch")
            regions = order_regions(regions, rtl=rtl)
            return regions, {"backend": backend.id, "real": False, "detected_boxes": len(detected), "label": backend.label}
        regions = []
        for box in detected:
            crop = im.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
            text, conf, extra = backend.recognize(crop, box["direction"])
            flags = list(box.get("flags") or []) + list(extra)
            if conf < 0.55 and "low-confidence" not in flags:
                flags.append("low-confidence")
            if not text.strip() and "ocr-empty" not in flags:
                flags.append("ocr-empty")
            regions.append({
                **box, "ocr_text": text, "confidence": conf, "flags": flags,
                "review": 1 if needs_review(conf, flags) else 0,
            })
        return regions, {"backend": backend.id, "real": True, "detected_boxes": len(detected), "label": backend.label}
