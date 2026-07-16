#!/usr/bin/env python3
"""
Extract unique static slides from training videos.

The video is treated as a sequence of still slides (not continuous motion).
Slide changes are detected by comparing sampled frames; each new stable slide
is saved as 01.png, 02.png, ... into a directory named after the video file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Извлекает уникальные слайды из обучающего видео. "
            "Принимает путь к видеофайлу или к директории с видео."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Путь к видеофайлу или директории с видеофайлами",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Родительская директория для папок со слайдами "
            "(по умолчанию — рядом с каждым видео)"
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Сколько кадров в секунду анализировать (по умолчанию: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help=(
            "Доля пикселей, которые должны отличаться, "
            "чтобы считать кадр новым слайдом (по умолчанию: 0.03 = 3%%)"
        ),
    )
    parser.add_argument(
        "--pixel-diff",
        type=int,
        default=25,
        help="Порог разницы яркости пикселя 0–255 (по умолчанию: 25)",
    )
    parser.add_argument(
        "--stability",
        type=int,
        default=2,
        help=(
            "Сколько подряд похожих выборок нужно, "
            "чтобы зафиксировать новый слайд (по умолчанию: 2)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg", "jpeg", "webp"),
        default="png",
        help="Формат сохраняемых картинок (по умолчанию: png)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Меньше вывода в консоль",
    )
    return parser.parse_args()


def collect_videos(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Путь не найден: {path}")

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Не похоже на видеофайл (расширение {path.suffix}): {path}"
            )
        return [path]

    videos = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"В директории нет видеофайлов: {path}")
    return videos


def frames_similar(
    a: np.ndarray,
    b: np.ndarray,
    threshold: float,
    pixel_diff: int,
) -> bool:
    """True if frames look like the same slide."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # Downscale for speed and to ignore tiny encoding noise.
    h, w = gray_a.shape
    scale = min(1.0, 480 / max(h, w))
    if scale < 1.0:
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        gray_a = cv2.resize(gray_a, size, interpolation=cv2.INTER_AREA)
        gray_b = cv2.resize(gray_b, size, interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(gray_a, gray_b)
    changed_ratio = float(np.mean(diff > pixel_diff))
    return changed_ratio < threshold


def slide_output_dir(video: Path, parent: Path | None) -> Path:
    base = parent if parent is not None else video.parent
    return (base / video.stem).resolve()


def extract_slides(
    video_path: Path,
    output_dir: Path,
    *,
    sample_fps: float,
    threshold: float,
    pixel_diff: int,
    stability: int,
    image_format: str,
    quiet: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(source_fps / sample_fps)))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous extraction for this video so numbering stays clean.
    ext = f".{image_format.lstrip('.')}"
    for old in output_dir.glob(f"*{ext}"):
        if old.stem.isdigit():
            old.unlink()

    last_saved: np.ndarray | None = None
    candidate: np.ndarray | None = None
    candidate_stable = 0
    saved_count = 0
    frame_idx = 0

    def save_slide(frame: np.ndarray) -> None:
        nonlocal saved_count, last_saved
        saved_count += 1
        name = f"{saved_count:02d}{ext}"
        out_path = output_dir / name
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise RuntimeError(f"Не удалось сохранить: {out_path}")
        last_saved = frame.copy()
        if not quiet:
            print(f"  + {name}")

    if not quiet:
        duration = total_frames / source_fps if source_fps and total_frames else 0
        print(
            f"→ {video_path.name}: "
            f"{source_fps:.1f} fps, ~{duration:.0f}s, sample every {step} frames"
        )
        print(f"  output: {output_dir}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            if last_saved is None:
                save_slide(frame)
                candidate = None
                candidate_stable = 0
                continue

            if frames_similar(frame, last_saved, threshold, pixel_diff):
                candidate = None
                candidate_stable = 0
                continue

            # New content relative to last saved slide.
            if candidate is None or not frames_similar(
                frame, candidate, threshold, pixel_diff
            ):
                candidate = frame.copy()
                candidate_stable = 1
            else:
                candidate_stable += 1

            if candidate_stable >= stability:
                save_slide(candidate)
                candidate = None
                candidate_stable = 0
    finally:
        cap.release()

    # If the video ended during a transition that never fully stabilized,
    # still keep the last distinct candidate.
    if candidate is not None and (
        last_saved is None
        or not frames_similar(candidate, last_saved, threshold, pixel_diff)
    ):
        save_slide(candidate)

    if not quiet:
        print(f"  готово: {saved_count} слайд(ов)")

    return saved_count


def main() -> int:
    args = parse_args()

    try:
        videos = collect_videos(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)

    total_slides = 0
    failed = 0

    for video in videos:
        out_dir = slide_output_dir(video, args.output_dir)
        try:
            total_slides += extract_slides(
                video,
                out_dir,
                sample_fps=args.sample_fps,
                threshold=args.threshold,
                pixel_diff=args.pixel_diff,
                stability=args.stability,
                image_format=args.format,
                quiet=args.quiet,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue batch
            failed += 1
            print(f"Ошибка при обработке {video}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(
            f"\nИтого: {len(videos) - failed}/{len(videos)} видео, "
            f"{total_slides} слайдов"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
