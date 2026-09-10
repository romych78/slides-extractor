#!/usr/bin/env bash
# Переносит файлы с суффиксом _upd из исходных директорий
# в одноимённые директории с суффиксом _upd.
#
# Использование:
#   ./move_upd_files.sh           # директория по умолчанию: videos
#   ./move_upd_files.sh /path/to  # произвольная директория

set -euo pipefail

BASE_DIR="${1:-videos}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Ошибка: директория '$BASE_DIR' не существует." >&2
  exit 1
fi

echo "=== Рабочая директория: $BASE_DIR ==="
echo

# Находим только директории верхнего уровня без суффикса _upd
while IFS= read -r -d '' src_dir; do
  dir_name="$(basename "$src_dir")"

  # Пропускаем уже созданные *_upd директории
  if [[ "$dir_name" == *_upd ]]; then
    continue
  fi

  dest_dir="${src_dir}_upd"

  # Создаём целевую директорию, если её ещё нет
  if [[ -d "$dest_dir" ]]; then
    echo "[SKIP] Директория уже существует: $dest_dir"
  else
    echo "[CREATE] Создаём: $dest_dir"
    mkdir "$dest_dir"
  fi

  # Ищем файлы с "_upd" в имени внутри исходной директории
  moved=0
  while IFS= read -r -d '' file; do
    file_name="$(basename "$file")"
    echo "  [MOVE] $file_name  ->  $(basename "$dest_dir")/"
    mv "$file" "$dest_dir/"
    ((moved++)) || true
  done < <(find "$src_dir" -maxdepth 1 -type f -name '*_upd*' -print0)

  if (( moved == 0 )); then
    echo "  [INFO] Файлов *_upd* не найдено в: $dir_name"
  else
    echo "  [DONE] Перенесено файлов: $moved"
  fi
  echo

done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

echo "=== Готово ==="
