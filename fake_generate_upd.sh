#!/usr/bin/env bash
# Создаёт фейковые *_upd.* файлы — простые копии оригиналов слайдов,
# без реального вызова Gemini/Imagen. Повторяет файловый эффект
# slide_processor.py (NN.ext -> NN_upd.ext рядом с оригиналом), но
# вместо AI-генерации просто копирует исходный файл.
#
# После этого скрипта, как обычно, прогоните move_upd_files.sh, чтобы
# перенести *_upd.* в сестринские папки <stem>_upd/ для extract_slides.py --update.
#
# Использование:
#   ./fake_generate_upd.sh              # директория по умолчанию: videos
#   ./fake_generate_upd.sh /path/to      # произвольная директория
#   ./fake_generate_upd.sh videos --force  # пересоздать, даже если NN_upd.* уже есть

set -euo pipefail

BASE_DIR="videos"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE=1
      ;;
    *)
      BASE_DIR="$arg"
      ;;
  esac
done

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Ошибка: директория '$BASE_DIR' не существует." >&2
  exit 1
fi

echo "=== Рабочая директория: $BASE_DIR ==="
[[ "$FORCE" -eq 1 ]] && echo "(режим --force: пересоздаём существующие *_upd.*)"
echo

created=0
skipped=0

# Те же расширения и правило исключения, что в slide_processor.py:
# IMAGE_EXTENSIONS = {".jpg", ".png"}, файлы с "_upd" в имени не трогаем.
while IFS= read -r -d '' file; do
  dir_name="$(dirname "$file")"
  base_name="$(basename "$file")"
  stem="${base_name%.*}"
  ext="${base_name##*.}"

  # Пропускаем файлы, у которых "_upd" уже есть в имени (сами *_upd.* файлы)
  if [[ "$stem" == *_upd* ]]; then
    continue
  fi

  out_file="${dir_name}/${stem}_upd.${ext}"

  if [[ -f "$out_file" && "$FORCE" -eq 0 ]]; then
    echo "  [SKIP] $out_file (уже существует)"
    ((skipped++)) || true
    continue
  fi

  cp -p "$file" "$out_file"
  echo "  [FAKE] $base_name  ->  $(basename "$out_file")"
  ((created++)) || true
done < <(find "$BASE_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' \) -print0)

echo
echo "=== Готово: создано $created, пропущено $skipped ==="
