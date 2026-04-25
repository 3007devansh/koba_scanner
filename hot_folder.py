"""
Koba Document Scanner — hot_folder.py
======================================
Watches a folder for incoming PDFs and processes them automatically
using the same default settings as the web dashboard.

Folder layout (created automatically):

    hot/
      input/      ← drop PDFs here
      output/     ← cleaned results appear here (one sub-folder per file)
      processed/  ← originals are moved here after successful processing

Usage:
    python hot_folder.py                   # uses ./hot
    python hot_folder.py --folder D:/scan  # custom root folder
    python hot_folder.py --interval 10     # poll every 10 seconds

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

from processor import Config, process_pdf

# ── logging ───────────────────────────────────────────────────────────────────
from logging.handlers import RotatingFileHandler

# Detailed log format
LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(message)s"
DATE_FMT = "%H:%M:%S"

# Configure the main logger
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
log = logging.getLogger("scanner.hotfolder")

# Add a rotating file handler (5MB max per file, keeps 2 backups)
file_handler = RotatingFileHandler("hot_folder.log", maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
logging.getLogger("").addHandler(file_handler)
logging.getLogger("scanner").setLevel(logging.INFO)   # Capture processor logs too


# ── defaults (must mirror app.py / Config) ────────────────────────────────────
DEFAULT_CONFIG = Config(
    dpi           = 300,
    skew_method   = "projection",
    max_skew_deg  = 15.0,
    padding_mm    = 10.0,
    output_format = "pdf",
    image_format  = "jpg",
    jpeg_quality  = 85,
    save_debug    = False,
)


# ── hot folder watcher ────────────────────────────────────────────────────────

class HotFolder:
    """Manages the three-folder watch loop."""

    def __init__(self, root: Path, cfg: Config, interval: int) -> None:
        self.root      = root
        self.cfg       = cfg
        self.interval  = interval

        self.input_dir     = root / "input"
        self.output_dir    = root / "output"
        self.processed_dir = root / "processed"

        # Create folders if they don't exist
        for folder in (self.input_dir, self.output_dir, self.processed_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self._running = True
        self._in_flight: set[Path] = set()   # files currently being processed

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("=" * 55)
        log.info("  Koba Hot Folder Watcher")
        log.info(f"  Root     : {self.root}")
        log.info(f"  Input    : {self.input_dir}")
        log.info(f"  Output   : {self.output_dir}")
        log.info(f"  Processed: {self.processed_dir}")
        log.info(f"  Poll     : every {self.interval}s")
        log.info(f"  DPI      : {self.cfg.dpi}")
        log.info(f"  Method   : {self.cfg.skew_method}")
        log.info(f"  Padding  : {self.cfg.padding_mm} mm")
        log.info("  Watching for PDFs…  (Ctrl+C to stop)")
        log.info("=" * 55)

        while self._running:
            try:
                self._scan_and_process()
            except Exception as exc:
                log.error(f"Unexpected error in scan loop: {exc}", exc_info=True)
            time.sleep(self.interval)

    def _scan_and_process(self) -> None:
        pdfs = sorted(self.input_dir.glob("*.pdf"))
        pdfs_to_process = [p for p in pdfs if p not in self._in_flight]

        if not pdfs_to_process:
            return

        log.info(f"Found {len(pdfs_to_process)} PDF(s) to process")

        for pdf_path in pdfs_to_process:
            if not self._running:
                break
            self._process_one(pdf_path)

    def _process_one(self, pdf_path: Path) -> None:
        """Process a single PDF — move to processed/ on success, leave on failure."""
        self._in_flight.add(pdf_path)

        stem      = pdf_path.stem
        out_dir   = self.output_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("-" * 55)
        log.info(f"Processing: {pdf_path.name}")

        try:
            results = process_pdf(
                input_pdf   = str(pdf_path),
                output_dir  = str(out_dir),
                cfg         = self.cfg,
                progress_cb = self._progress_cb_factory(pdf_path.name),
                log_cb      = lambda msg: log.info(f"  [{stem}] {msg}"),
            )

            ok_count    = sum(1 for r in results if r.success)
            total_count = len(results)

            if ok_count == 0 and total_count > 0:
                raise RuntimeError(f"All {total_count} pages failed — see log above")

            # Move the final PDF to the root output folder for cleaner access
            cleaned_pdf_name = f"{stem}_cleaned.pdf"
            generated_pdf = out_dir / cleaned_pdf_name
            
            if generated_pdf.exists():
                final_dest = self.output_dir / cleaned_pdf_name
                # If target exists, overwrite it (user dropped file twice)
                if final_dest.exists(): final_dest.unlink()
                shutil.move(str(generated_pdf), str(final_dest))
                log.info(f"✓ Done: {cleaned_pdf_name}  →  output/")
            else:
                log.info(f"✓ Done: {pdf_path.name}  →  {out_dir.name}/")

            # Move original to processed/
            dest = self.processed_dir / pdf_path.name
            # Handle name collision in processed/
            if dest.exists():
                ts   = time.strftime("%Y%m%d_%H%M%S")
                dest = self.processed_dir / f"{stem}_{ts}{pdf_path.suffix}"

            shutil.move(str(pdf_path), str(dest))
            log.info(f"  Original moved  →  processed/{dest.name}")

            # Clean up the temp subfolder if it's now empty (normal for PDF-only mode)
            try:
                if not any(out_dir.iterdir()):
                    out_dir.rmdir()
            except Exception:
                pass

        except Exception as exc:
            log.error(f"✗ Failed: {pdf_path.name}  —  {exc}")
            log.error("  File left in input/ for retry.")

        finally:
            self._in_flight.discard(pdf_path)

    @staticmethod
    def _progress_cb_factory(filename: str):
        def _cb(done: int, total: int) -> None:
            pct = int(done / max(total, 1) * 100)
            log.info(f"  [{filename}]  page {done}/{total}  ({pct}%)")
        return _cb

    def stop(self) -> None:
        self._running = False


# ── entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Koba Hot Folder — auto-processes PDFs dropped into input/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--folder", default="./hot",
        help="Root folder containing input/, output/, processed/ sub-folders",
    )
    p.add_argument("--dpi",        type=int,   default=300,    help="Render DPI")
    p.add_argument("--method",     default="projection",
                   choices=["hough", "projection"],            help="Skew detection method")
    p.add_argument("--padding-mm", type=float, default=10.0,   help="Border padding (mm)")
    p.add_argument("--interval",   type=int,   default=5,      help="Poll interval (seconds)")
    p.add_argument("--output-format", choices=["pdf", "images", "both"], default="pdf",
                   help="What to produce per file")
    p.add_argument("--debug", action="store_true", help="Save debug mask/bbox images")
    return p


def main() -> None:
    args   = _build_parser().parse_args()
    root   = Path(args.folder).resolve()

    cfg = Config(
        dpi           = args.dpi,
        skew_method   = args.method,
        max_skew_deg  = 15.0,
        padding_mm    = args.padding_mm,
        output_format = args.output_format,
        image_format  = "jpg",
        jpeg_quality  = 85,
        save_debug    = args.debug,
    )

    watcher = HotFolder(root=root, cfg=cfg, interval=args.interval)

    # Graceful shutdown on Ctrl+C or SIGTERM
    def _handle_signal(sig, frame):
        log.info("Shutdown signal received. Finishing current file then exiting…")
        watcher.stop()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    watcher.run()
    log.info("Hot folder watcher stopped.")


if __name__ == "__main__":
    main()
