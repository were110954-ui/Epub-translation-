"""Japanese OCR adapters.

Text boxes come from text_lines: one horizontal line or one vertical column, not a panel.
Recognition uses an installed backend only: Tesseract, RapidOCR, or EasyOCR. There is no
fake recognizer in the default path. Tests pass a fixture backend that reads text embedded
in the PNG and is labeled as not real OCR.
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
    "ocr-failed", "ocr-backend-error", "typeset-overflow", "outside-balloon",
    "direction-uncertain", "effect-uncertain", "furigana-uncertain",
}


def needs_review(confidence: float | None, flags: list[str]) -> bool:
    if confidence is not None and confidence < 0.55:
        return True
    return any(f in REVIEW_FLAGS for f in flags)


def direction_of(w: int, h: int) -> str:
    return "vertical" if h >= w * 1.25 else "horizontal"


def order_regions(regions: list[dict], rtl: bool = True) -> list[dict]:
    """Rows top to bottom. Inside a row, right-to-left when rtl."""
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


def order_manga_regions(regions: list[dict], rtl: bool = True, page_h: int | None = None) -> list[dict]:
    """Panel bands from top to bottom. Vertical text in a band is ordered as columns.

    Japanese vertical columns run right-to-left, and each column runs top to bottom.
    Horizontal balloons stay in row order. This is an independent reading-order rule,
    not a copy of another application's sorter.
    """
    items = [dict(r) for r in regions]
    if not items:
        return []
    heights = sorted(r["h"] for r in items)
    median_h = heights[len(heights) // 2]
    gap_limit = max(28, int(median_h * 0.9))
    if page_h:
        gap_limit = max(gap_limit, int(page_h * 0.04))
    items.sort(key=lambda r: (r["y"], r["x"]))
    bands: list[list] = [[items[0]]]
    for region in items[1:]:
        prev = bands[-1][-1]
        gap = region["y"] - (prev["y"] + prev["h"])
        if gap > gap_limit:
            bands.append([region])
        else:
            bands[-1].append(region)
    ordered = []
    for band in bands:
        vertical = [r for r in band if r["h"] >= r["w"] * 1.2]
        if vertical and len(vertical) * 2 >= len(band):
            columns: list[list] = []
            for region in sorted(vertical, key=lambda r: r["x"] + r["w"] / 2, reverse=bool(rtl)):
                cx = region["x"] + region["w"] / 2
                placed = False
                for column in columns:
                    ccx = sum(c["x"] + c["w"] / 2 for c in column) / len(column)
                    if abs(cx - ccx) <= max(region["w"], column[0]["w"]) * 0.8:
                        column.append(region)
                        placed = True
                        break
                if not placed:
                    columns.append([region])
            if rtl:
                columns.sort(key=lambda col: -sum(c["x"] for c in col) / len(col))
            else:
                columns.sort(key=lambda col: sum(c["x"] for c in col) / len(col))
            for column in columns:
                column.sort(key=lambda r: r["y"])
                ordered.extend(column)
            horizontal = [r for r in band if r not in vertical]
            ordered.extend(order_regions(horizontal, rtl=rtl))
        else:
            ordered.extend(order_regions(band, rtl=rtl))
    for i, region in enumerate(ordered):
        region["idx"] = i
    return ordered


def _cc_boxes(mask: bytearray, width: int, height: int, min_area: int) -> list[dict]:
    seen = bytearray(width * height)
    boxes = []
    for start in range(width * height):
        if not mask[start] or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        count = 0
        minx = maxx = start % width
        miny = maxy = start // width
        while stack:
            i = stack.pop()
            x, y = i % width, i // width
            count += 1
            if x < minx:
                minx = x
            elif x > maxx:
                maxx = x
            if y < miny:
                miny = y
            elif y > maxy:
                maxy = y
            for j, ok in ((i - 1, x > 0), (i + 1, x + 1 < width), (i - width, y > 0), (i + width, y + 1 < height)):
                if ok and mask[j] and not seen[j]:
                    seen[j] = 1
                    stack.append(j)
        bw, bh = maxx - minx + 1, maxy - miny + 1
        if count < min_area or bw < 2 or bh < 2:
            continue
        if bw * bh > 400 and count / (bw * bh) < 0.12:
            continue
        if bw > width * 0.72 and bh < 7:
            continue
        if bh > height * 0.72 and bw < 7:
            continue
        boxes.append({"x": minx, "y": miny, "w": bw, "h": bh})
    return boxes


def detect_balloons(image, rtl: bool = True) -> list[dict]:
    """Text lines only. The name stays so older callers keep working."""
    from .text_lines import detect_text_lines
    return detect_text_lines(image, rtl=rtl)


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

    def _once(self, image, direction: str):
        from PIL import Image, ImageOps
        langs = tesseract_langs()
        lang = "jpn_vert" if direction == "vertical" and "jpn_vert" in langs else "jpn"
        psm = "5" if direction == "vertical" else "6"
        im = ImageOps.autocontrast(image.convert("L")).convert("RGB")
        short = min(im.size)
        if short < 64:
            factor = max(2, (64 + short - 1) // short)
            im = im.resize((im.width * factor, im.height * factor), Image.Resampling.NEAREST)
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

    def recognize(self, image, direction: str):
        text, conf, flags = self._once(image, direction)
        other = "horizontal" if direction == "vertical" else "vertical"
        if conf < 0.55 or not text:
            alt, alt_conf, alt_flags = self._once(image, other)
            if alt_conf > conf and alt:
                return alt, alt_conf, alt_flags + ["direction-checked"]
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
            if detected and len(detected) != len(regions):
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
