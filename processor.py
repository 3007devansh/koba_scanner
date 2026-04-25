"""
Koba Document Scanner — processor.py
=======================================
Pipeline for each page of a scanned PDF:
  1.  Render page to high-res image
  2.  Build a binary content mask (adaptive threshold)
  3.  Detect skew angle  (Hough lines  OR  Projection profile)
  4.  Deskew — rotate so text runs perfectly horizontal
  5.  Autocrop — find the tightest bounding box around all ink
  6.  Add a uniform white border on all four sides (default 10 mm)
  7.  Export each page as PNG + reassemble into a clean PDF

No OCR is performed.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import img2pdf
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scanner")


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    dpi: int = 300
    """Render resolution.  300 is the safe minimum; 400 for small handwriting."""

    skew_method: str = "projection"
    """'hough' (fast, good for ruled/printed) or 'projection' (slow, best for
    dense handwriting without clear baselines)."""

    max_skew_deg: float = 15.0
    """Angles larger than this are almost certainly wrong detections — ignored."""

    padding_mm: float = 10.0
    """Uniform white border added to every side of the cropped content (mm)."""

    # ── internal tuning ───────────────────────────────────────────────────────
    border_strip_px: int = 15
    """Pixels stripped from each edge before analysis to remove scanner-frame noise."""

    adaptive_block: int = 41
    """Block size for adaptive thresholding.  Must be odd.  Smaller (41 vs 51)
    preserves fine details and handwriting edges better."""

    adaptive_C: int = 10
    """Constant subtracted in adaptive threshold.  Lower (10 vs 15) preserves
    faint ink that would otherwise be lost."""

    morph_close_kernel: Tuple[int, int] = (5, 3)
    """Morphological closing kernel (w, h).  Smaller (5 vs 7) to avoid over-connecting
    strokes while maintaining horizontal text line cohesion."""

    crop_margin_px: int = 20
    """Pixel margin added around detected content box to preserve edge strokes."""

    min_content_fraction: float = 0.01
    """If detected content box < 1% of page (was 3%), assume detection failed
    and fall back to the full page for safety."""

    output_format: str = "pdf"
    """'pdf' | 'images' | 'both'"""

    image_format: str = "jpg"
    """'png' | 'jpg' | 'tiff'"""

    jpeg_quality: int = 85
    """JPEG compression quality (1-100). Lower means smaller file size but more compression artifacts. Only used if image_format is 'jpg'."""

    save_debug: bool = False
    """Write intermediate images (binary mask, bbox overlay) to a debug/ folder."""


# ── per-page result ───────────────────────────────────────────────────────────
@dataclass
class PageResult:
    page_num: int
    success: bool
    skew_angle: float = 0.0
    crop_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h (pre-pad)
    crop_w: int = 0
    crop_h: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    messages: Optional[List[str]] = None


# ── image helpers ─────────────────────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimetres to pixels at the given DPI."""
    return int(round(mm * dpi / 25.4))


# ── binary content mask ───────────────────────────────────────────────────────

def make_binary_mask(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Return a binary image where ink/content pixels are 255 and background is 0.

    Strategy:
    - Light Gaussian blur to suppress grain without destroying strokes.
    - Adaptive (Gaussian-weighted) threshold so uneven scanner lighting,
      yellowed paper, and ink fade are all handled locally.
    - Morphological closing to connect nearby strokes into solid content blobs,
      making the bounding-box detection much more stable.
    - Strip a thin border to eliminate scanner-frame shadow / edge artefacts.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    block = cfg.adaptive_block | 1   # force odd
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, cfg.adaptive_C,
    )

    kw, kh = cfg.morph_close_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Zero out the thin scanner-edge strip
    s = cfg.border_strip_px
    if s > 0:
        closed[:s, :]  = 0
        closed[-s:, :] = 0
        closed[:, :s]  = 0
        closed[:, -s:] = 0

    return closed


# ── skew detection ────────────────────────────────────────────────────────────

def detect_skew_hough(binary: np.ndarray, max_deg: float) -> float:
    """
    Probabilistic Hough line transform on the Canny edge map.

    Works best when:
    - There are clear horizontal text baselines or ruled lines.
    - The document has enough contrast (works for faded ink too after thresholding).

    Returns the median angle of all near-horizontal detected lines (degrees).
    A positive angle means the content is rotated counter-clockwise; we correct
    by rotating clockwise by that amount.
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # minLineLength = 1/6 of image width — long enough to be a real text line,
    # short enough to catch partial lines near the edge.
    min_len = binary.shape[1] // 6
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=30,
    )

    if lines is None:
        return 0.0

    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        if dx == 0:
            continue
        ang = float(np.degrees(np.arctan2(y2 - y1, dx)))
        if abs(ang) <= max_deg:
            angles.append(ang)

    if not angles:
        return 0.0

    skew = float(np.median(angles))
    log.debug(f"    Hough: {skew:+.2f}° from {len(angles)} lines")
    return skew


def detect_skew_projection(binary: np.ndarray, max_deg: float) -> float:
    """
    Horizontal projection profile maximisation.

    Rotate the binary image through a fine angle grid and pick the angle that
    produces the highest variance in the row-sum projection — that means the
    text lines are sharpest (most concentrated per row) and therefore horizontal.

    Slower than Hough but extremely reliable for dense handwriting where no
    individual line is long enough for Hough to latch onto.
    """
    h, w = binary.shape
    cx, cy = w / 2, h / 2
    best_angle, best_var = 0.0, -1.0

    for ang in np.arange(-max_deg, max_deg + 0.25, 0.25):
        M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        row_sums = np.sum(rotated, axis=1, dtype=np.float64)
        var = float(np.var(row_sums))
        if var > best_var:
            best_var = var
            best_angle = ang

    log.debug(f"    Projection: {best_angle:+.2f}°")
    return best_angle


def detect_skew(binary: np.ndarray, cfg: Config) -> float:
    """Dispatch to the chosen method; clamp implausible results to 0."""
    if cfg.skew_method == "projection":
        angle = detect_skew_projection(binary, cfg.max_skew_deg)
    else:
        angle = detect_skew_hough(binary, cfg.max_skew_deg)

    if abs(angle) > cfg.max_skew_deg:
        log.warning(f"    Detected angle {angle:+.1f}° exceeds max — ignoring")
        return 0.0
    return float(angle)


# ── deskew ────────────────────────────────────────────────────────────────────

def deskew(cv_img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by -angle so that text is horizontal.

    The canvas is expanded to fit the rotated content without any clipping.
    New areas (corners) are filled with white (255, 255, 255).
    """
    if abs(angle) < 0.05:
        return cv_img

    h, w = cv_img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    # Compute enlarged canvas that fits the whole rotated rectangle
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    return cv2.warpAffine(
        cv_img, M, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ── autocrop ──────────────────────────────────────────────────────────────────

def make_lab_mask(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # B > 135 effectively targets yellow/brown hues of aging parchment
    mask = cv2.inRange(B, 135, 255)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def make_hsv_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    # Saturation threshold isolates colored objects from gray backgrounds
    blurred = cv2.GaussianBlur(S, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_content_bbox(
    cv_img: np.ndarray, cfg: Config
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    Find bounding box and convex hull using Contour method on an HSV mask (with LAB fallback).
    """
    h_img, w_img = cv_img.shape[:2]
    page_size = h_img * w_img

    mask_hsv = make_hsv_mask(cv_img)
    mask = mask_hsv
    
    # Extract contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback to LAB if HSV contour fails or if area is extremely small
    valid = [c for c in contours if 0.005 < (cv2.contourArea(c) / page_size) < 0.98]
    
    if not valid:
        # Fallback triggered
        mask_lab = make_lab_mask(cv_img)
        mask = mask_lab
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 0.005 < (cv2.contourArea(c) / page_size) < 0.98]

    if not valid:
        return None, None

    cnt = max(valid, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    x_min, y_min, w_cnt, h_cnt = cv2.boundingRect(hull)
    
    # Add safety margin to preserve edge content
    margin = cfg.crop_margin_px
    x = int(max(0, x_min - margin))
    y = int(max(0, y_min - margin))
    w_box = int(min(w_img - x, w_cnt + 2 * margin))
    h_box = int(min(h_img - y, h_cnt + 2 * margin))

    return (x, y, w_box, h_box), hull

    
def crop_to_content(cv_img: np.ndarray, bbox: Optional[Tuple], hull: Optional[np.ndarray] = None) -> np.ndarray:
    if bbox is None:
        return cv_img
    x, y, w, h = bbox
    cropped = cv_img[y : y + h, x : x + w].copy()
    
    if hull is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        local_hull = hull.copy()
        local_hull[:, 0, 0] -= x
        local_hull[:, 0, 1] -= y
        cv2.drawContours(mask, [local_hull], -1, 255, thickness=cv2.FILLED)
        cropped[mask == 0] = [255, 255, 255]
        
    return cropped


# ── uniform border ────────────────────────────────────────────────────────────

def add_uniform_border(cv_img: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Add a uniform white border of exactly cfg.padding_mm millimetres
    on all four sides, computed from cfg.dpi so the physical size is exact.
    """
    pad = mm_to_px(cfg.padding_mm, cfg.dpi)
    return cv2.copyMakeBorder(
        cv_img, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )

def finalize_page(
    page_num: int,
    temp_path: Path,
    out_dir: Path,
    max_w: int,
    max_h: int,
    cfg: Config
) -> str:
    """Read temp cropped image, pad to max_w/max_h, add uniform border, save final."""
    cropped = cv2.imread(str(temp_path))
    if cropped is None:
        raise ValueError(f"Could not read temp file: {temp_path}")
    h, w = cropped.shape[:2]
    
    # Pad to max_w, max_h (centered)
    pad_top = max(0, (max_h - h) // 2)
    pad_bottom = max(0, max_h - h - pad_top)
    pad_left = max(0, (max_w - w) // 2)
    pad_right = max(0, max_w - w - pad_left)
    
    uniform_cropped = cv2.copyMakeBorder(
        cropped, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    
    # Add the configured uniform padding
    padded = add_uniform_border(uniform_cropped, cfg)
    
    # Save final
    out_pil = cv_to_pil(padded)
    img_path = out_dir / f"page_{page_num:03d}.{cfg.image_format}"
    if cfg.image_format.lower() in ("jpg", "jpeg"):
        out_pil.save(str(img_path), dpi=(cfg.dpi, cfg.dpi), quality=cfg.jpeg_quality)
    else:
        out_pil.save(str(img_path), dpi=(cfg.dpi, cfg.dpi))
        
    # Delete temp file
    temp_path.unlink(missing_ok=True)
    
    return str(img_path)

# ── per-page pipeline ─────────────────────────────────────────────────────────

def process_page(
    page_num: int,
    pil_img: Image.Image,
    out_dir: Path,
    cfg: Config,
    debug_dir: Optional[Path] = None,
) -> PageResult:

    result = PageResult(page_num=page_num, success=False, messages=[])
    
    def _locallog(m):
        result.messages.append(m)
        log.info(m)

    _locallog(f"  Page {page_num:>3}: {pil_img.width}×{pil_img.height} px")

    try:
        cv_img = pil_to_cv(pil_img)
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 1. Binary mask for analysis
        mask = make_binary_mask(gray, cfg)

        if cfg.save_debug and debug_dir:
            cv2.imwrite(
                str(debug_dir / f"p{page_num:03d}_1_mask.png"),
                mask,
            )

        # 2. Skew detection
        angle = detect_skew(mask, cfg)
        result.skew_angle = angle
        _locallog(f"           skew = {angle:+.2f}°  [{cfg.skew_method}]")

        # 3. Deskew
        deskewed = deskew(cv_img, angle)

        # 4. Autocrop — find content bounding box and hull on the DESKEWED image
        bbox, hull = find_content_bbox(deskewed, cfg)
        result.crop_box = bbox

        if bbox:
            x, y, w, h = bbox
            _locallog(f"           crop = ({x}, {y}, {w}×{h})")
        else:
            _locallog(f"           crop = full page (no reliable box found)")

        if cfg.save_debug and debug_dir and bbox:
            overlay = deskewed.copy()
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 0), 4)
            if hull is not None:
                cv2.drawContours(overlay, [hull], -1, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"p{page_num:03d}_2_bbox.png"), overlay)

        cropped = crop_to_content(deskewed, bbox, hull)

        result.crop_w = cropped.shape[1]
        result.crop_h = cropped.shape[0]

        # Save intermediate lossless crop
        temp_path = out_dir / f"p{page_num:03d}_crop.png"
        cv2.imwrite(str(temp_path), cropped)
        result.output_path = str(temp_path)

        result.success = True

    except Exception as exc:
        result.error = str(exc)
        err_msg = f"  Page {page_num} FAILED: {exc}"
        result.messages.append(err_msg)
        log.error(err_msg, exc_info=True)

    return result


# ── PDF assembler ─────────────────────────────────────────────────────────────

def assemble_pdf(image_paths: List[str], out_pdf: Path, jlog=log) -> None:
    valid = [p for p in image_paths if p and Path(p).exists()]
    if not valid:
        jlog.error("No valid page images to assemble into PDF.")
        return
    with open(out_pdf, "wb") as fh:
        fh.write(img2pdf.convert(valid))
    jlog.info(f"  PDF assembled → {out_pdf}")


# ── multiprocessing worker ────────────────────────────────────────────────────

def _process_page_worker(
    page_num: int,
    pil_img: Image.Image,
    out_dir_str: str,
    cfg_dict: dict,
    debug_dir_str: Optional[str],
) -> PageResult:
    """
    Worker function for parallel page processing.  Reconstructs Config from dict
    (required for pickling across process boundaries) and processes one page.
    """
    # Reconstruct Config and Path objects from pickled state
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in [
        'dpi', 'skew_method', 'max_skew_deg', 'padding_mm', 'output_format',
        'image_format', 'jpeg_quality', 'save_debug', 'border_strip_px', 'adaptive_block',
        'adaptive_C', 'morph_close_kernel', 'crop_margin_px', 'min_content_fraction'
    ]})
    
    out_dir = Path(out_dir_str)
    debug_dir = Path(debug_dir_str) if debug_dir_str else None
    
    return process_page(page_num, pil_img, out_dir, cfg, debug_dir)


# ── main orchestrator ─────────────────────────────────────────────────────────

def process_pdf(
    input_pdf: str,
    output_dir: str,
    cfg: Config,
    progress_cb=None,          # optional callable(page_num, total) for GUI
    job_control=None,          # optional dict for state pause/cancel
    log_cb=None,               # optional callback for isolated logs
    output_stem: Optional[str] = None,  # override output filename stem (uses input stem if None)
) -> List[PageResult]:
    """
    Full pipeline: PDF → cleaned pages → output PDF.

    progress_cb is called after each page with (page_num, total_pages).
    """
    job_control = job_control or {}
    
    class _JobLogger:
        def debug(self, msg): 
            log.debug(msg)
            if log_cb: log_cb(msg)
        def info(self, msg): 
            log.info(msg)
            if log_cb: log_cb(msg)
        def error(self, msg, exc_info=None): 
            log.error(msg, exc_info=exc_info)
            if log_cb: log_cb(msg)
        def warning(self, msg): 
            log.warning(msg)
            if log_cb: log_cb(msg)
            
    jlog = _JobLogger()
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    debug_dir: Optional[Path] = None
    if cfg.save_debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    jlog.info("=" * 55)
    jlog.info(f"Input  : {input_path.name}")
    jlog.info(f"Output : {out_dir}")
    jlog.info(f"DPI    : {cfg.dpi}")
    jlog.info(f"Method : {cfg.skew_method}")
    pad_px = mm_to_px(cfg.padding_mm, cfg.dpi)
    jlog.info(f"Padding: {cfg.padding_mm} mm  ({pad_px} px at {cfg.dpi} dpi)")
    jlog.info("=" * 55)

    # Render PDF pages
    jlog.info("Rendering PDF pages via pypdfium2…")
    t0 = time.time()
    
    pil_pages = []
    pdf_doc = pdfium.PdfDocument(str(input_path))
    scale = cfg.dpi / 72.0
    try:
        for page in pdf_doc:
            state = job_control.get("state", "running")
            if state == "cancelled":
                jlog.warning("Job cancelled during PDF rendering.")
                return []
            while job_control.get("state") == "paused":
                time.sleep(0.5)
                if job_control.get("state") == "cancelled":
                    jlog.warning("Job cancelled during PDF rendering.")
                    return []

            bitmap = page.render(scale=scale)
            pil_pages.append(bitmap.to_pil())
    finally:
        pdf_doc.close()   # release file handle so hot-folder can move the file on Windows

    jlog.info(f"  {len(pil_pages)} page(s) rendered in {time.time() - t0:.1f}s")

    results: List[PageResult] = []
    total = len(pil_pages)

    # Prepare Config for pickling to worker processes
    cfg_dict = asdict(cfg)

    from concurrent.futures import FIRST_COMPLETED, wait

    # Process pages in parallel — use all available cores minus 2 to keep
    # the system responsive while still maximising throughput.
    total_cores = os.cpu_count() or 4
    num_workers = max(1, total_cores - 3)
    jlog.info(f"Using {num_workers} worker process(es) out of {total_cores} available")

    completed_pages = 0
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            active_futures = {}
            pages_iter = enumerate(pil_pages, start=1)

            def submit_more():
                while len(active_futures) < num_workers:
                    try:
                        i, pil_img = next(pages_iter)
                        fut = executor.submit(
                            _process_page_worker,
                            i,
                            pil_img,
                            str(pages_dir),
                            cfg_dict,
                            str(debug_dir) if debug_dir else None,
                        )
                        active_futures[fut] = i
                    except StopIteration:
                        break
            
            submit_more()
            
            # Collect results as they complete, parsing job state dynamically
            while active_futures:
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED, timeout=0.5)
                
                state = job_control.get("state", "running")
                if state == "cancelled":
                    jlog.warning("Job cancelled. Stopping workers...")
                    for f in active_futures.keys():
                        f.cancel()
                    break

                for fut in done:
                    page_idx = active_futures.pop(fut)
                    try:
                        r = fut.result()
                        results.append(r)
                        
                        # Emit the child pages' logs linearly so UI receives them cleanly
                        if r.messages:
                            for msg in r.messages:
                                jlog.info(msg)
                                
                        completed_pages += 1
                        if progress_cb:
                            progress_cb(completed_pages, total)
                    except Exception as e:
                        jlog.error(f"Page {page_idx} processing failed: {e}", exc_info=True)
                        error_result = PageResult(page_num=page_idx, success=False, error=str(e))
                        results.append(error_result)
                        completed_pages += 1
                        if progress_cb:
                            progress_cb(completed_pages, total)
                
                # Check paused state before pushing more work to workers
                if job_control.get("state", "running") == "running":
                    submit_more()
        
        # Sort results by page number to maintain order
        results.sort(key=lambda r: r.page_num)

    except Exception as e:
        jlog.error(f"Executor error: {e}", exc_info=True)
        raise

    # ── Phase 2: Uniform Resizing ──
    successful_results = [r for r in results if r.success]
    if successful_results:
        max_w = max(r.crop_w for r in successful_results)
        max_h = max(r.crop_h for r in successful_results)
        jlog.info(f"Normalising all pages to largest crop size: {max_w}x{max_h} px")
        
        for r in successful_results:
            state = job_control.get("state", "running")
            if state == "cancelled":
                break
                
            if r.output_path and Path(r.output_path).exists():
                try:
                    final_path = finalize_page(
                        r.page_num,
                        Path(r.output_path),
                        pages_dir,
                        max_w, max_h, cfg
                    )
                    r.output_path = final_path
                except Exception as e:
                    jlog.error(f"Failed to finalize page {r.page_num}: {e}")
                    r.success = False
                    r.error = str(e)

    # Output
    ok_paths = [r.output_path for r in results if r.success]

    if cfg.output_format in ("pdf", "both") and ok_paths:
        pdf_stem = output_stem if output_stem else input_path.stem
        out_pdf = out_dir / f"{pdf_stem}_cleaned.pdf"
        assemble_pdf(ok_paths, out_pdf, jlog)

    # Clean up intermediate page images — they are only needed for PDF assembly.
    # The final PDF (and any debug images) are kept; raw page files are deleted.
    if cfg.output_format in ("pdf", "both") and pages_dir.exists():
        try:
            import shutil as _shutil
            _shutil.rmtree(str(pages_dir))
            jlog.info(f"  Temp page images removed ({pages_dir.name}/)")
        except Exception as _e:
            jlog.warning(f"  Could not remove temp pages dir: {_e}")

    # Summary
    ok_count = sum(1 for r in results if r.success)
    jlog.info("-" * 55)
    jlog.info(f"Done — {ok_count}/{total} pages OK")
    for r in results:
        mark = "✓" if r.success else "✗"
        detail = f"skew {r.skew_angle:+.2f}°" if r.success else (r.error or "")
        jlog.info(f"  {mark} Page {r.page_num:>3}  {detail}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Koba Document Scanner — autocrop & deskew scanned PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_pdf",   help="Path to the scanned input PDF")
    p.add_argument("output_dir",  help="Directory to write results into")

    p.add_argument("--dpi", type=int, default=300,
                   help="Render DPI (300 recommended; 400 for fine handwriting)")
    p.add_argument("--skew-method", choices=["hough", "projection"], default="hough",
                   help="Skew detection algorithm")
    p.add_argument("--max-skew", type=float, default=15.0,
                   help="Maximum plausible skew angle in degrees")
    p.add_argument("--padding-mm", type=float, default=10.0,
                   help="Uniform white border added to each side (millimetres)")
    p.add_argument("--output-format", choices=["pdf", "images", "both"], default="pdf",
                   help="What to produce")
    p.add_argument("--image-format", choices=["png", "jpg", "tiff"], default="jpg",
                   help="Format for individual page images")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG quality 1-100 (used when image-format is jpg)")
    p.add_argument("--debug", action="store_true",
                   help="Save intermediate binary mask and bounding-box images")
    return p


def main():
    args = _build_parser().parse_args()
    cfg = Config(
        dpi           = args.dpi,
        skew_method   = args.skew_method,
        max_skew_deg  = args.max_skew,
        padding_mm    = args.padding_mm,
        output_format = args.output_format,
        image_format  = args.image_format,
        jpeg_quality  = args.jpeg_quality,
        save_debug    = args.debug,
    )
    process_pdf(args.input_pdf, args.output_dir, cfg)


if __name__ == "__main__":
    main()
