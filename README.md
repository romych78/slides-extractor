# slides-extractor

CLI-утилита для работы с обучающими видео, где «видео» — это **смена статических слайдов** (screencast / презентация), а не обычный видеопоток.

Два режима:

1. **Извлечение** — найти смены слайдов, сохранить картинки и **точные тайминги** показа.
2. **`--update`** — подставить обновлённые слайды (`*_upd`) и пересобрать видео с **теми же** интервалами начала/конца.

## Требования

- Python 3.10+
- зависимости из `requirements.txt` (OpenCV, NumPy)
- **ffmpeg** (+ желательно **ffprobe**) в `PATH` — нужен для режима `--update`

```bash
# macOS
brew install ffmpeg
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Извлечение слайдов

```bash
# одно видео → папка с именем файла + metadata.json
python extract_slides.py path/to/lecture.mp4
# → path/to/lecture/01.png, 02.png, …
# → path/to/lecture/metadata.json

# все видео в директории
python extract_slides.py path/to/videos/

# положить папки со слайдами в другое место
python extract_slides.py path/to/videos/ -o slides_out/
```

Поддерживаемые расширения: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.m4v`, `.wmv`.

### Структура после извлечения

```text
videos/
  видео 1.mp4
  видео 1/
    01.png
    02.png
    …
    metadata.json    # тайминги и параметры исходника
```

### `metadata.json` (фрагмент)

```json
{
  "version": 1,
  "video_file": "видео 1.mp4",
  "fps": 24.0,
  "width": 1280,
  "height": 720,
  "total_frames": 7760,
  "duration_sec": 323.333,
  "slides": [
    {
      "index": 1,
      "file": "01.png",
      "start_frame": 0,
      "end_frame": 240,
      "start_sec": 0.0,
      "end_sec": 10.0,
      "duration_sec": 10.0,
      "start_ts": "00:00.000",
      "end_ts": "00:10.000"
    }
  ]
}
```

`end_frame` — **исключающая** граница (как в half-open интервале `[start, end)`): слайд занимает кадры `start_frame … end_frame-1`.

Повторный запуск по тому же видео **перезаписывает** нумерованные картинки и `metadata.json`.

## 2. Обновление слайдов и пересборка видео (`--update`)

1. Извлеките слайды как обычно (нужны папка `<stem>/` и `metadata.json`).
2. Создайте папку **`<stem>_upd`** рядом (или в `-o`) и положите замены:
   - было: `видео 1/01.png`
   - стало: `видео 1_upd/01_upd.png`
3. Запустите:

```bash
python extract_slides.py videos/ --update
# или одно видео:
python extract_slides.py "videos/видео 1.mp4" --update
```

Результат: **`видео 1_upd.mp4`** рядом с исходником.

- Тайминги берутся из `metadata.json` (кадры → длительность).
- Если для номера `NN` нет `NN_upd.*`, подставляется оригинал `NN.*`.
- Аудиодорожка из исходного видео **копируется** (если есть).
- Разрешение выходного видео = `width` × `height` из metadata (картинки scale+pad).

### Пример раскладки для update

```text
videos/
  видео 1.mp4
  видео 1/
    01.png … 12.png
    metadata.json
  видео 1_upd/
    01_upd.png      # заменили только 1-й и 3-й
    03_upd.png
  видео 1_upd.mp4   # ← результат --update
```

## Параметры

| Флаг | Смысл | По умолчанию |
|------|--------|--------------|
| `path` | Видеофайл или директория с видео | — |
| `--update` | Пересобрать видео из `*_upd` слайдов | off |
| `-o`, `--output-dir` | Родительская папка для `stem/` и `stem_upd/` | рядом с видео |
| `--sample-fps` | Частота выборки кадров (только extract) | `2` |
| `--threshold` | Доля отличающихся пикселей → «новый слайд» | `0.03` (3%) |
| `--pixel-diff` | Порог разницы яркости пикселя (0–255) | `25` |
| `--stability` | Подряд похожих выборок для фиксации слайда | `2` |
| `--format` | `png` / `jpg` / `jpeg` / `webp` (extract) | `png` |
| `--crf` | Качество libx264 при `--update` (меньше = лучше) | `18` |
| `-q`, `--quiet` | Меньше логов | off |

### Тонкая настройка extract

- **Слишком много дублей / шум / анимация** — увеличьте `--threshold` (например `0.05`).
- **Пропускает смену слайда** — уменьшите `--threshold` (например `0.015`) или поднимите `--sample-fps`.
- **Ловит кадры посередине перехода** — увеличьте `--stability` (например `3`–`4`).

## Как это устроено

### Extract

1. OpenCV читает видео, сэмплирует ~`sample_fps` кадров/с.
2. Сравнивает кадр с последним сохранённым слайдом (доля заметно отличающихся пикселей).
3. Новый слайд фиксируется после `stability` стабильных выборок.
4. Пишет `01.png`, … и `metadata.json` с `start_frame` / `end_frame` / секундами.

### Update

1. Читает `metadata.json` из `<stem>/`.
2. Для каждого слайда берёт `<stem>_upd/NN_upd.ext` или оригинал.
3. Через **ffmpeg** concat demuxer собирает видеопоток с точными `duration` по кадрам.
4. Накладывает аудио из исходного файла (если есть).

## Локальные данные

Папка `videos/` (исходники, слайды, `*_upd`) **не коммитится** — см. `.gitignore`.

```bash
python extract_slides.py videos/
python extract_slides.py videos/ --update
```

## Лицензия

Private / personal use, unless stated otherwise.
