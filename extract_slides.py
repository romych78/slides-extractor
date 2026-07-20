#!/usr/bin/env python3
"""
Extract unique static slides from training videos, save timing metadata,
and optionally rebuild a video from updated slides with identical timings.

Extract mode (default):
  video.mp4 → video/01.png … + video/metadata.json

Update mode (--update):
  expects video/ (original slides + metadata.json)
  and video_upd/01_upd.png … (replacements; missing slides keep originals)
  writes video_upd.mp4 next to the source video
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
METADATA_NAME = "metadata.json"
UPD_SUFFIX = "_upd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Извлекает уникальные слайды из обучающего видео с таймингами, "
            "либо (--update) пересобирает видео из обновлённых слайдов "
            "с теми же интервалами показа."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Путь к видеофайлу или директории с видеофайлами",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Режим обновления: для каждого видео взять слайды из "
            "<stem>_upd (имена NN_upd.ext) и metadata.json, "
            "собрать <stem>_upd.mp4 с исходными таймингами"
        ),
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
        "--crf",
        type=int,
        default=18,
        help="CRF для libx264 при --update (по умолчанию: 18, меньше = лучше)",
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
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.endswith(UPD_SUFFIX)
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


def slide_upd_dir(video: Path, parent: Path | None) -> Path:
    base = parent if parent is not None else video.parent
    return (base / f"{video.stem}{UPD_SUFFIX}").resolve()


def frame_to_sec(frame: int, fps: float) -> float:
    return frame / fps if fps else 0.0


def format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет {METADATA_NAME}: сначала извлеките слайды "
            f"(python extract_slides.py …). Ожидался файл: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


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

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if source_fps <= 1e-3:
        source_fps = 25.0
    prop_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(source_fps / sample_fps)))

    output_dir.mkdir(parents=True, exist_ok=True)

    ext = f".{image_format.lstrip('.')}"
    for old in output_dir.glob(f"*{ext}"):
        if old.stem.isdigit():
            old.unlink()
    meta_path = output_dir / METADATA_NAME
    if meta_path.exists():
        meta_path.unlink()

    last_saved: np.ndarray | None = None
    candidate: np.ndarray | None = None
    candidate_frame = 0
    candidate_stable = 0
    saved_count = 0
    frame_idx = 0
    slides: list[dict[str, Any]] = []

    def close_previous(end_frame: int) -> None:
        if not slides:
            return
        prev = slides[-1]
        prev["end_frame"] = end_frame
        prev["end_sec"] = frame_to_sec(end_frame, source_fps)
        prev["duration_sec"] = prev["end_sec"] - prev["start_sec"]
        prev["end_ts"] = format_ts(prev["end_sec"])

    def save_slide(frame: np.ndarray, start_frame: int) -> None:
        nonlocal saved_count, last_saved
        close_previous(start_frame)
        saved_count += 1
        name = f"{saved_count:02d}{ext}"
        out_path = output_dir / name
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise RuntimeError(f"Не удалось сохранить: {out_path}")
        last_saved = frame.copy()
        start_sec = frame_to_sec(start_frame, source_fps)
        slides.append(
            {
                "index": saved_count,
                "file": name,
                "start_frame": start_frame,
                "end_frame": start_frame,  # filled when next slide starts / EOF
                "start_sec": start_sec,
                "end_sec": start_sec,
                "duration_sec": 0.0,
                "start_ts": format_ts(start_sec),
                "end_ts": format_ts(start_sec),
            }
        )
        if not quiet:
            print(f"  + {name}  @ {format_ts(start_sec)} (frame {start_frame})")

    if not quiet:
        duration = prop_frames / source_fps if source_fps and prop_frames else 0
        print(
            f"→ {video_path.name}: "
            f"{source_fps:.3f} fps, {width}x{height}, "
            f"~{duration:.0f}s, sample every {step} frames"
        )
        print(f"  output: {output_dir}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                if last_saved is None:
                    # Первый слайд показывается с начала ролика.
                    save_slide(frame, start_frame=0)
                    candidate = None
                    candidate_stable = 0
                elif frames_similar(frame, last_saved, threshold, pixel_diff):
                    candidate = None
                    candidate_stable = 0
                else:
                    # Новый контент относительно последнего сохранённого слайда.
                    if candidate is None or not frames_similar(
                        frame, candidate, threshold, pixel_diff
                    ):
                        candidate = frame.copy()
                        candidate_frame = frame_idx
                        candidate_stable = 1
                    else:
                        candidate_stable += 1

                    if candidate_stable >= stability:
                        save_slide(candidate, start_frame=candidate_frame)
                        candidate = None
                        candidate_stable = 0

            frame_idx += 1
    finally:
        cap.release()

    total_frames = frame_idx
    if total_frames == 0:
        raise RuntimeError(f"В видео нет кадров: {video_path}")

    # Нестабильный candidate в конце всё равно сохраняем.
    if candidate is not None and (
        last_saved is None
        or not frames_similar(candidate, last_saved, threshold, pixel_diff)
    ):
        save_slide(candidate, start_frame=candidate_frame)

    close_previous(total_frames)

    metadata = {
        "version": 1,
        "video_file": video_path.name,
        "video_path": str(video_path.resolve()),
        "fps": source_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": frame_to_sec(total_frames, source_fps),
        "sample_fps": sample_fps,
        "threshold": threshold,
        "pixel_diff": pixel_diff,
        "stability": stability,
        "image_format": image_format.lstrip("."),
        "slides": slides,
    }
    write_metadata(meta_path, metadata)

    if not quiet:
        print(f"  metadata: {meta_path.name} ({saved_count} слайд(ов))")
        for s in slides:
            print(
                f"    {s['file']}: {s['start_ts']} → {s['end_ts']} "
                f"({s['duration_sec']:.2f}s)"
            )
        print(f"  готово: {saved_count} слайд(ов)")

    return saved_count


def find_slide_image(directory: Path, index: int, *, updated: bool) -> Path | None:
    """Find NN.png or NN_upd.png (any supported extension)."""
    stem = f"{index:02d}{UPD_SUFFIX}" if updated else f"{index:02d}"
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # Also accept non-padded names: 1_upd.png
    stem_alt = f"{index}{UPD_SUFFIX}" if updated else f"{index}"
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem_alt}{ext}"
        if candidate.is_file():
            return candidate
    return None


def resolve_slide_images(
    slides: list[dict[str, Any]],
    original_dir: Path,
    upd_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Return ordered image paths and labels (upd/original) per slide."""
    paths: list[Path] = []
    sources: list[str] = []
    missing: list[int] = []

    for slide in slides:
        index = int(slide["index"])
        upd = find_slide_image(upd_dir, index, updated=True)
        if upd is not None:
            paths.append(upd)
            sources.append("upd")
            continue
        original_name = slide.get("file")
        original: Path | None = None
        if original_name:
            candidate = original_dir / original_name
            if candidate.is_file():
                original = candidate
        if original is None:
            original = find_slide_image(original_dir, index, updated=False)
        if original is None:
            missing.append(index)
            continue
        paths.append(original)
        sources.append("original")

    if missing:
        raise FileNotFoundError(
            "Нет изображений для слайдов: "
            + ", ".join(f"{i:02d}" for i in missing)
            + f" (искал в {upd_dir} и {original_dir})"
        )
    return paths, sources


def ffprobe_has_audio(video_path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())


def run_ffmpeg(cmd: list[str], *, quiet: bool) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите ffmpeg для режима --update."
        )
    full = cmd if quiet else cmd[:1] + ["-hide_banner"] + cmd[1:]
    result = subprocess.run(
        full,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg завершился с кодом {result.returncode}:\n{detail[-2000:]}"
        )


def write_ffconcat(path: Path, images: list[Path], durations: list[float]) -> None:
    lines = ["ffconcat version 1.0"]
    for image, duration in zip(images, durations, strict=True):
        # Absolute paths; escape single quotes for the concat demuxer.
        resolved = image.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{resolved}'")
        lines.append(f"duration {duration:.6f}")
    # ffmpeg concat demuxer requires the last file to be listed once more.
    last = images[-1].resolve().as_posix().replace("'", r"'\''")
    lines.append(f"file '{last}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_video(
    video_path: Path,
    original_dir: Path,
    upd_dir: Path,
    output_path: Path,
    *,
    crf: int,
    quiet: bool,
) -> Path:
    meta = load_metadata(original_dir / METADATA_NAME)
    slides = meta.get("slides") or []
    if not slides:
        raise RuntimeError(f"В metadata нет слайдов: {original_dir / METADATA_NAME}")

    fps = float(meta.get("fps") or 25.0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if width <= 0 or height <= 0:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap.release()

    if not upd_dir.is_dir():
        raise FileNotFoundError(
            f"Нет папки с обновлёнными слайдами: {upd_dir}\n"
            f"Создайте её и положите файлы вида 01{UPD_SUFFIX}.png, 02{UPD_SUFFIX}.png, …"
        )

    images, sources = resolve_slide_images(slides, original_dir, upd_dir)
    durations: list[float] = []
    for slide in slides:
        start_f = int(slide["start_frame"])
        end_f = int(slide["end_frame"])
        if end_f < start_f:
            raise RuntimeError(
                f"Некорректный интервал слайда {slide.get('file')}: "
                f"{start_f}..{end_f}"
            )
        # Prefer frame-based duration for exact rebuild.
        durations.append((end_f - start_f) / fps)

    if abs(sum(durations) - float(meta.get("duration_sec") or 0)) > 1.0:
        # Soft warning only — frame math is authoritative.
        if not quiet:
            print(
                f"  предупреждение: сумма длительностей слайдов "
                f"{sum(durations):.3f}s ≠ duration_sec "
                f"{meta.get('duration_sec')}"
            )

    if not quiet:
        print(f"→ update {video_path.name}")
        print(f"  metadata: {original_dir / METADATA_NAME}")
        print(f"  upd dir:  {upd_dir}")
        n_upd = sources.count("upd")
        n_orig = sources.count("original")
        print(f"  слайды:   {len(images)} (upd={n_upd}, original={n_orig})")
        print(f"  output:   {output_path}")

    with tempfile.TemporaryDirectory(prefix="slides_upd_") as tmp:
        tmp_path = Path(tmp)
        concat_path = tmp_path / "slides.concat"
        silent_video = tmp_path / "silent.mp4"
        write_ffconcat(concat_path, images, durations)

        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={fps}"
        )
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-r",
                str(fps),
                str(silent_video),
            ],
            quiet=quiet,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_audio = ffprobe_has_audio(video_path)
        if has_audio:
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(silent_video),
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(output_path),
                ],
                quiet=quiet,
            )
        else:
            shutil.copy2(silent_video, output_path)

    if not quiet:
        print(f"  готово: {output_path.name}")
    return output_path


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

    total_ok = 0
    failed = 0

    if args.update:
        for video in videos:
            original_dir = slide_output_dir(video, args.output_dir)
            upd_dir = slide_upd_dir(video, args.output_dir)
            out_video = video.with_name(f"{video.stem}{UPD_SUFFIX}{video.suffix}")
            try:
                rebuild_video(
                    video,
                    original_dir,
                    upd_dir,
                    out_video,
                    crf=args.crf,
                    quiet=args.quiet,
                )
                total_ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"Ошибка при update {video}: {exc}", file=sys.stderr)
        if not args.quiet:
            print(f"\nИтого update: {total_ok}/{len(videos)} видео")
        return 1 if failed else 0

    total_slides = 0
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
            total_ok += 1
        except Exception as exc:  # noqa: BLE001 — report and continue batch
            failed += 1
            print(f"Ошибка при обработке {video}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(
            f"\nИтого: {total_ok}/{len(videos)} видео, "
            f"{total_slides} слайдов"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
