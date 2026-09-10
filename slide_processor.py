#!/usr/bin/env python3
"""
Batch-regenerate training slides: analyze each slide with Gemini Vision,
generate a clean high-res watercolor background with Imagen 3, and lay the
original text back on top with Pillow.

  video/01.png  →  video/01_upd.jpg   (next to the source image)

Run `move_upd_files.sh` afterwards to move the *_upd.jpg files into the
sibling <stem>_upd/ directories expected by `extract_slides.py --update`.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types

IMAGE_EXTENSIONS = {".jpg", ".png"}
UPD_MARK = "_upd"
CANVAS_SIZE = (1920, 1080)
MAX_RECONCILE_PASSES = 3

# font_size in schema = cap height as fraction of slide height (e.g. 0.08 → ~86px at 1080p).
MIN_FONT_RATIO = 0.016
MAX_FONT_RATIO = 0.28
DEFAULT_MAIN_TITLE_SIZE = 0.065
DEFAULT_BLOCK_TITLE_SIZE = 0.042
DEFAULT_DESC_SIZE = 0.028

LINE_SPACING = 1.2
TEXT_COLLISION_GAP_PX = 8

TITLE_COLOR = (26, 26, 26)
HALO_COLOR = (253, 246, 227)

QUALITY_SUFFIX = (
    "highly detailed watercolor biological illustration, vintage warm paper "
    "background, professional clean composition, artistic splashes."
)
NEGATIVE_PROMPT_TEXT = (
    "text, letters, watermark, labels, user interface, ui elements, buttons, "
    "rectangular cards, boxes, arrows, notebooklm, birdhouse, bird house, "
    "bird nest box, nesting box, bird feeder, treehouse, mailbox on tree"
)
# Gemini image models (generate_content) have no dedicated negative-prompt
# field, so the exclusion list is appended as plain instruction text instead.
NEGATIVE_SUFFIX = f"Negative prompt: {NEGATIVE_PROMPT_TEXT}."

ANALYSIS_INSTRUCTIONS = """
Ты анализируешь слайд обучающей презентации (акварельная иллюстрация + текстовые
надписи). Верни структурированный JSON строго по заданной схеме.

Правила:
1. Полностью игнорируй и не переноси в blocks графические примитивы:
   прямоугольные плашки, карточки, кружки, маркеры списков, стрелки — это не
   текстовые блоки.
2. Категорически игнорируй водяные знаки, логотипы и системные надписи
   (NotebookLM, Gemini, элементы плеера) — не включай их ни в main_title, ни
   в blocks, ни в imagen_prompt.
2а. Различай **рисованную графику** и **отдельный текстовый слой**:
    - Большая стилизованная цифра номера слайда (1, 2, 3…) — часть иллюстрации:
      опиши в key_visual_subjects/imagen_prompt; добавь цифру в illustration_only_texts;
      **НЕ** включай в elements (не накладывать текстом — она уже в рисунке).
    - Надпись нарисована на объекте (например «Сахар» на мешке) — то же:
      illustration_only_texts + key_visual_subjects, **не** elements.
    - Показания приборов, температуры (+10°C), даты календаря как отдельный текст:
      **elements**; в imagen рисуй прибор без напечатанных цифр на нём.
3. Если слайд перегружен сплошным мелким текстом (is_text_heavy_only=true):
   оставь только главную суть в main_title, а мелкий текст не переноси
   дословно — преврати его смысл в метафорический визуальный сюжет для
   imagen_prompt.
4. Не придумывай новый текст и не искажай факты. Разрешено только исправлять
   явные опечатки и грамматические ошибки в исходном тексте.
5. В imagen_prompt на английском языке детально опиши художественный стиль,
   объекты и композицию так, чтобы области под координатами (x, y) каждого
   элемента из elements оставались свободным контрастным фоном (negative space) —
   без текста, плашек, стрелок и водяных знаков.
6. Все текстовые надписи перечисли в elements — каждая отдельная надпись или
   логический текстовый блок = один элемент. Не сливай надписи разного размера
   в один элемент.
7. Координаты x, y — левый верхний угол **самого текста** на оригинале (не плашки,
   не карточки). max_width — ширина текстовой области на оригинале (0.0–1.0).
8. font_size — визуально оцени размер шрифта на оригинале как долю высоты слайда
   (высота заглавных букв / высота изображения). Ориентиры:
   - очень крупный акцент / номер слайда: 0.12–0.18
   - главный заголовок: 0.07–0.11
   - подзаголовок / тезис блока: 0.04–0.06
   - обычный текст / описание: 0.025–0.035
   - мелкий текст: 0.018–0.024
   Разные надписи на слайде должны иметь **разный** font_size, соответствующий
   оригиналу — не унифицируй кегль.
9. bold=true для заголовков и акцентов, bold=false для описаний и мелкого текста.
10. Одна логическая надпись или заголовок (даже на несколько строк) — один элемент
    elements с полным текстом. Не дроби «Скрытая угроза клеща» на отдельные слова.
11. Координата y следующего элемента должна быть ниже нижней границы предыдущего
    текстового блока на оригинале — не ставь элементы на одну вертикаль без зазора.
12. Смысл иллюстрации (imagen_prompt, key_visual_subjects, composition_summary)
    должен **точно сохранять** учебный смысл оригинала. Не подменяй объекты
    похожими, но разными: улей ≠ скворечник/домик на дереве; термометр в улье ≠
    почтовый ящик; пчелиная семья ≠ просто «пчёлы у дерева».
13. key_visual_subjects — 3–8 обязательных визуальных элементов (на английском),
    без которых смысл слайда теряется. Пример: "stacked wooden Langstroth beehive
    box", "horizontal landing board at hive entrance", "dense bee cluster inside
    hive cavity", "round dial thermometer mounted inside hive wall".
14. composition_summary — точная композиция на английском: ракурс (close-up,
    cutaway, side view), что в фоне vs что в центре, как объекты расположены
    **реально** на оригинале. Дерево/листья рядом с ульем — фон сбоку, а не
    «улей вмонтирован в дерево». Не выдумывай сцену, которой нет на слайде.
15. В imagen_prompt опиши объекты с учебной точностью: для улья — ящик пчелиного
    улья с летком и подставкой, пчёлы на летке и внутри; НЕ birdhouse, НЕ nesting
    box на ветке.
16. illustration_only_texts — все строки, которые на оригинале являются частью
    иллюстрации (рисованный номер слайда, надпись на мешке/банке и т.п.) и
    **не должны** дублироваться текстовым слоем Pillow. Для слайда с большой
    рисованной «2» слева — illustration_only_texts должен содержать «2».
""".strip()

_SHORT_NUMBER_RE = re.compile(r"^\d{1,2}$")
_NUMBER_IN_SUBJECT_RE = re.compile(
    r"(?:stylized|decorative|large|outlined|prominent|graphic)\s+number\s*['\"]?(?P<num>\d{1,2})['\"]?",
    re.IGNORECASE,
)
_LABEL_ON_OBJECT_RE = re.compile(
    r"(?:labeled|label(?:ed)?|marked|written on|text on)\s+['\"](?P<label>[^'\"]+)['\"]",
    re.IGNORECASE,
)


class TextElement(BaseModel):
    text: str = Field(description="Текст надписи как на оригинале слайда")
    x: float = Field(description="Относительная X левого верхнего угла текста (0.0 - 1.0)")
    y: float = Field(description="Относительная Y левого верхнего угла текста (0.0 - 1.0)")
    max_width: float = Field(default=0.35, description="Максимальная ширина текстовой области (0.0 - 1.0)")
    font_size: float = Field(
        default=DEFAULT_BLOCK_TITLE_SIZE,
        description=(
            "Размер шрифта как доля высоты слайда (cap height). "
            "Оцени визуально с оригинала; разные надписи — разный font_size."
        ),
    )
    bold: bool = Field(default=True, description="True для заголовков/акцентов, False для описаний")


class TextBlock(BaseModel):
    title: str = Field(description="Заголовок смыслового блока или ключевой тезис")
    description: str = Field(default="", description="Поясняющий текст / описание блока")
    x: float = Field(description="Относительная координата X левого верхнего угла блока (0.0 - 1.0)")
    y: float = Field(description="Относительная координата Y левого верхнего угла блока (0.0 - 1.0)")
    max_width: float = Field(default=0.35, description="Максимальная ширина текстового блока (0.0 - 1.0)")


class SlideLayoutSchema(BaseModel):
    is_text_heavy_only: bool = Field(description="True, если слайд содержит только массив сплошного мелкого текста")
    elements: list[TextElement] = Field(
        default_factory=list,
        description=(
            "Только отдельный текстовый слой (заголовки, тезисы, показания). "
            "НЕ включать рисованные номера слайдов и надписи на объектах — "
            "они в illustration_only_texts и рисуются в иллюстрации."
        ),
    )
    # Legacy fields — used only if elements is empty (older API responses).
    main_title: str = Field(default="", description="Главный заголовок слайда")
    blocks: list[TextBlock] = Field(default_factory=list, description="Список блоков для верстки поверх фона")
    imagen_prompt: str = Field(
        description=(
            "Детальный англоязычный промпт для генерации фона: художественный стиль, "
            "объекты, акварельная текстура, зоны negative space под текст. "
            "Точно сохраняй смысл оригинала; СТРОГО без текста, плашек, стрелок, "
            "водяных знаков."
        )
    )
    key_visual_subjects: list[str] = Field(
        default_factory=list,
        description=(
            "Обязательные визуальные элементы (англ.), которые должны быть явно "
            "узнаваемы на иллюстрации — без подмены похожих объектов"
        ),
    )
    composition_summary: str = Field(
        default="",
        description=(
            "Композиция оригинала (англ.): ракурс, расположение объектов, "
            "что в фоне и что в фокусе, без искажения смысла"
        ),
    )
    illustration_only_texts: list[str] = Field(
        default_factory=list,
        description=(
            "Текст/символы, которые являются частью иллюстрации (рисованный номер "
            "слайда, надпись на объекте) — рисуются в фоне, НЕ накладываются Pillow"
        ),
    )


@dataclass
class Result:
    path: Path
    status: str  # "ok" / "skipped" / "error"
    detail: str = ""


@dataclass
class Config:
    input_dir: Path
    force: bool
    limit: int | None
    font_bold: Path
    font_regular: Path
    vision_model: str
    imagen_model: str
    use_vertex: bool
    api_key: str | None
    vertex_project: str | None
    vertex_location: str | None
    vertex_api_key: str | None


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("1", "true", "yes")
    api_key = os.environ.get("GEMINI_API_KEY")
    vertex_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    vertex_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    vertex_api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")

    if use_vertex:
        # Vertex AI ("Gemini Enterprise Agent Platform"): either an Express Mode
        # API key, or ADC/service account tied to a GOOGLE_CLOUD_PROJECT.
        if not vertex_api_key and not vertex_project:
            raise RuntimeError(
                "GOOGLE_GENAI_USE_VERTEXAI=true, но не задан ни GOOGLE_CLOUD_API_KEY "
                "(Express Mode), ни GOOGLE_CLOUD_PROJECT (ADC)."
            )
    elif not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY не найден. Добавьте его в .env (см. .env.example), "
            "либо включите GOOGLE_GENAI_USE_VERTEXAI=true для Vertex AI."
        )

    font_bold = Path(args.font_bold or os.environ.get("FONT_TITLE_PATH", "./assets/fonts/Inter-Bold.ttf"))
    font_regular = Path(args.font_regular or os.environ.get("FONT_BODY_PATH", "./assets/fonts/Inter-Regular.ttf"))
    for label, font_path in (("FONT_TITLE_PATH", font_bold), ("FONT_BODY_PATH", font_regular)):
        if not font_path.is_file():
            raise RuntimeError(
                f"Шрифт {label} не найден: {font_path}\n"
                "Положите файл .ttf с поддержкой кириллицы по этому пути "
                "(например, Inter-Bold.ttf / Inter-Regular.ttf в assets/fonts/)."
            )

    default_imagen_model = "imagen-3.0-generate-002" if use_vertex else "gemini-2.5-flash-image"

    return Config(
        input_dir=args.input_dir,
        force=args.force,
        limit=args.limit,
        font_bold=font_bold,
        font_regular=font_regular,
        vision_model=os.environ.get("VISION_MODEL", "gemini-2.5-flash"),
        imagen_model=os.environ.get("IMAGEN_MODEL", default_imagen_model),
        use_vertex=use_vertex,
        api_key=api_key,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_api_key=vertex_api_key,
    )


def build_client(config: Config) -> genai.Client:
    if config.use_vertex:
        if config.vertex_api_key:
            # Vertex AI Express Mode: API key, no ADC/service account needed.
            return genai.Client(vertexai=True, api_key=config.vertex_api_key)
        return genai.Client(
            vertexai=True,
            project=config.vertex_project,
            location=config.vertex_location or "us-central1",
        )
    return genai.Client(api_key=config.api_key)


def collect_images(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {root}")
    found = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and UPD_MARK not in p.stem
    ]
    return sorted(found)


def output_path_for(source: Path) -> Path:
    return source.with_name(f"{source.stem}{UPD_MARK}.jpg")


@api_retry
def analyze_slide(client: genai.Client, model: str, image_path: Path) -> SlideLayoutSchema:
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_bytes = image_path.read_bytes()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=ANALYSIS_INSTRUCTIONS),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SlideLayoutSchema,
            temperature=0.4,
            # Structured JSON without the SDK AFC loop (avoids noisy warning + extra latency).
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    if not isinstance(response.parsed, SlideLayoutSchema):
        raise RuntimeError(f"Не удалось разобрать ответ Gemini Vision: {response.text!r}")
    return response.parsed


def build_generation_prompt(layout: SlideLayoutSchema) -> str:
    """Assemble image prompt with mandatory subjects and composition fidelity."""
    parts = [
        "Educational training slide watercolor illustration. Faithfully preserve the "
        "exact subject matter and meaning of the original slide — never substitute "
        "similar-looking objects (beehive must stay a beehive, not a birdhouse or "
        "tree-mounted nesting box).",
    ]
    if layout.key_visual_subjects:
        parts.append(
            "MANDATORY clearly recognizable elements: "
            + "; ".join(layout.key_visual_subjects)
        )
    if layout.composition_summary.strip():
        parts.append(f"Exact composition and spatial layout: {layout.composition_summary.strip()}")
    parts.append(layout.imagen_prompt.strip())
    return " ".join(parts)


@api_retry
def generate_background(client: genai.Client, model: str, layout: SlideLayoutSchema) -> Image.Image:
    full_prompt = f"{build_generation_prompt(layout)} {QUALITY_SUFFIX}"

    if "imagen" in model.lower():
        # Real Imagen 3, only reachable via Vertex AI ("Gemini Enterprise Agent
        # Platform") credentials — supports a proper negative_prompt field.
        result = client.models.generate_images(
            model=model,
            prompt=full_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                negative_prompt=NEGATIVE_PROMPT_TEXT,
            ),
        )
        image_bytes = result.generated_images[0].image.image_bytes
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Gemini Developer API path: native image models (e.g. gemini-2.5-flash-image)
    # have no negative-prompt field, so exclusions are appended as plain text.
    response = client.models.generate_content(
        model=model,
        contents=f"{full_prompt} {NEGATIVE_SUFFIX}",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        raise RuntimeError("Модель не вернула ответ (возможно, сработал safety-фильтр)")
    for part in candidates[0].content.parts or []:
        if part.inline_data is not None:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    raise RuntimeError("Модель не вернула изображение (возможно, сработал safety-фильтр)")


def clamp_font_ratio(ratio: float, default: float) -> float:
    if ratio <= 0:
        return default
    return max(MIN_FONT_RATIO, min(MAX_FONT_RATIO, ratio))


def font_size_px(ratio: float, canvas_height: int, default_ratio: float) -> int:
    return max(12, round(clamp_font_ratio(ratio, default_ratio) * canvas_height))


def layout_elements(layout: SlideLayoutSchema) -> list[TextElement]:
    """Prefer flat elements; fall back to legacy main_title + blocks."""
    if layout.elements:
        return layout.elements

    legacy: list[TextElement] = []
    if layout.main_title:
        legacy.append(
            TextElement(
                text=layout.main_title,
                x=0.05,
                y=0.04,
                max_width=0.9,
                font_size=DEFAULT_MAIN_TITLE_SIZE,
                bold=True,
            )
        )
    for block in layout.blocks:
        if block.title:
            legacy.append(
                TextElement(
                    text=block.title,
                    x=block.x,
                    y=block.y,
                    max_width=block.max_width,
                    font_size=DEFAULT_BLOCK_TITLE_SIZE,
                    bold=True,
                )
            )
        if block.description:
            legacy.append(
                TextElement(
                    text=block.description,
                    x=block.x,
                    y=block.y + 0.04,
                    max_width=block.max_width,
                    font_size=DEFAULT_DESC_SIZE,
                    bold=False,
                )
            )
    return legacy


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def intersects(self, other: BoundingBox, gap: float = 0.0) -> bool:
        return (
            self.left < other.right + gap
            and other.left < self.right + gap
            and self.top < other.bottom + gap
            and other.top < self.bottom + gap
        )


@dataclass
class PreparedText:
    x: float
    y: float
    lines: list[str]
    font: ImageFont.FreeTypeFont
    stroke_width: int
    bbox: BoundingBox


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def is_short_slide_number(text: str) -> bool:
    return bool(_SHORT_NUMBER_RE.match(text.strip()))


def subject_references_slide_number(number: str, subjects: list[str], prompt: str) -> bool:
    combined = " ".join(subjects) + " " + prompt
    num = number.strip()
    for match in _NUMBER_IN_SUBJECT_RE.finditer(combined):
        if match.group("num") == num:
            return True
    if re.search(rf"\bnumber\s*['\"]?{re.escape(num)}['\"]?", combined, re.IGNORECASE):
        return True
    if re.search(rf"\bdigit\s*['\"]?{re.escape(num)}['\"]?", combined, re.IGNORECASE):
        return True
    return False


def text_label_on_illustrated_object(text: str, subjects: list[str]) -> bool:
    norm = normalize_text(text)
    for subject in subjects:
        for match in _LABEL_ON_OBJECT_RE.finditer(subject):
            if normalize_text(match.group("label")) == norm:
                return True
        subject_norm = normalize_text(subject)
        if norm in subject_norm and any(
            kw in subject_norm for kw in ("labeled", "label", "marked", "written on", "on bag", "on the bag")
        ):
            return True
    return False


def should_skip_text_overlay(elem: TextElement, layout: SlideLayoutSchema) -> bool:
    """Skip Pillow text when the same content is drawn in the generated illustration."""
    text = elem.text.strip()
    if not text:
        return True

    blocked = {normalize_text(t) for t in layout.illustration_only_texts if t.strip()}
    norm = normalize_text(text)
    if norm in blocked:
        return True

    subjects = layout.key_visual_subjects
    prompt = layout.imagen_prompt

    if is_short_slide_number(text):
        if subject_references_slide_number(text, subjects, prompt):
            return True
        # Large decorative slide numbers are illustration, not overlay text.
        if elem.font_size >= 0.10:
            return True

    if text_label_on_illustrated_object(text, subjects):
        return True

    return False


def overlay_elements(layout: SlideLayoutSchema) -> list[TextElement]:
    """Text elements to render with Pillow — excludes illustration-integrated labels."""
    return [elem for elem in layout_elements(layout) if not should_skip_text_overlay(elem, layout)]


def dedupe_elements(elements: list[TextElement]) -> list[TextElement]:
    """Drop duplicate labels returned at nearly the same position."""
    kept: list[TextElement] = []
    for elem in elements:
        text = elem.text.strip()
        if not text:
            continue
        norm = normalize_text(text)
        is_duplicate = False
        for prev in kept:
            if norm != normalize_text(prev.text):
                continue
            if abs(elem.x - prev.x) < 0.08 and abs(elem.y - prev.y) < 0.08:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(elem)
    return kept


def stroke_for_font(size_px: int) -> int:
    return max(2, round(size_px * 0.08))


def measure_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    x: float,
    y: float,
    stroke_width: int,
) -> BoundingBox:
    if not lines:
        return BoundingBox(x, y, x, y)

    left = right = x
    top = bottom = y
    y_cursor = y
    for line in lines:
        line_box = draw.textbbox((x, y_cursor), line, font=font, stroke_width=stroke_width)
        left = min(left, line_box[0])
        top = min(top, line_box[1])
        right = max(right, line_box[2])
        bottom = max(bottom, line_box[3])
        y_cursor += font.size * LINE_SPACING
    return BoundingBox(left, top, right, bottom)


def resolve_vertical_position(
    bbox: BoundingBox,
    y: float,
    placed: list[BoundingBox],
    canvas_height: float,
    gap: float,
) -> float:
    resolved_y = y
    max_y = max(0.0, canvas_height - bbox.height)
    for _ in range(400):
        shifted = BoundingBox(bbox.left, resolved_y, bbox.right, resolved_y + bbox.height)
        colliders = [box for box in placed if shifted.intersects(box, gap)]
        if not colliders:
            return min(resolved_y, max_y)
        resolved_y = max(box.bottom + gap for box in colliders)
    return min(resolved_y, max_y)


def prepare_text_placements(
    draw: ImageDraw.ImageDraw,
    layout: SlideLayoutSchema,
    font_bold_path: Path,
    font_regular_path: Path,
    canvas_width: int,
    canvas_height: int,
) -> list[PreparedText]:
    elements = dedupe_elements(overlay_elements(layout))
    sorted_elements = sorted(elements, key=lambda e: (e.y, e.x, -e.font_size))

    placed_boxes: list[BoundingBox] = []
    prepared: list[PreparedText] = []

    for elem in sorted_elements:
        text = elem.text.strip()
        if not text:
            continue

        size_px = font_size_px(elem.font_size, canvas_height, DEFAULT_BLOCK_TITLE_SIZE)
        font_path = font_bold_path if elem.bold else font_regular_path
        font = ImageFont.truetype(str(font_path), size_px)
        x_px = max(0.0, min(0.98, elem.x)) * canvas_width
        y_px = max(0.0, min(0.98, elem.y)) * canvas_height
        max_w_px = max(elem.max_width, 0.05) * canvas_width
        stroke = stroke_for_font(size_px)
        lines = wrap_text(draw, text, font, max_w_px)
        if not lines:
            continue

        bbox = measure_text_block(draw, lines, font, x_px, y_px, stroke)
        y_px = resolve_vertical_position(
            bbox,
            y_px,
            placed_boxes,
            canvas_height,
            TEXT_COLLISION_GAP_PX,
        )
        bbox = measure_text_block(draw, lines, font, x_px, y_px, stroke)
        placed_boxes.append(bbox)
        prepared.append(PreparedText(x_px, y_px, lines, font, stroke, bbox))

    return prepared


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width_px: float) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = [words[0]]
    for word in words[1:]:
        trial = f"{lines[-1]} {word}"
        if draw.textlength(trial, font=font) <= max_width_px:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float = LINE_SPACING,
    stroke_width: int | None = None,
) -> float:
    x, y = xy
    stroke = stroke_width if stroke_width is not None else stroke_for_font(font.size)
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=TITLE_COLOR,
            stroke_width=stroke,
            stroke_fill=HALO_COLOR,
        )
        y += font.size * line_spacing
    return y


def render_slide(
    background: Image.Image,
    layout: SlideLayoutSchema,
    font_bold_path: Path,
    font_regular_path: Path,
) -> Image.Image:
    canvas = background.resize(CANVAS_SIZE, Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    w, h = CANVAS_SIZE

    placements = prepare_text_placements(draw, layout, font_bold_path, font_regular_path, w, h)
    for item in placements:
        draw_text_block(draw, (item.x, item.y), item.lines, item.font, stroke_width=item.stroke_width)

    return canvas


def process_one(
    client: genai.Client,
    config: Config,
    path: Path,
    console: Console | None = None,
) -> Result:
    out_path = output_path_for(path)
    if out_path.exists() and not config.force:
        return Result(path, "skipped", str(out_path))

    def log_step(message: str) -> None:
        if console is not None:
            console.print(f"  [dim]{path.name}:[/dim] {message}")

    try:
        t0 = time.monotonic()
        log_step(f"анализ ({config.vision_model})…")
        layout = analyze_slide(client, config.vision_model, path)
        log_step(f"анализ готов за {time.monotonic() - t0:.0f}s, генерация фона ({config.imagen_model})…")
        t1 = time.monotonic()
        background = generate_background(client, config.imagen_model, layout)
        log_step(f"фон готов за {time.monotonic() - t1:.0f}s, верстка текста…")
        final = render_slide(background, layout, config.font_bold, config.font_regular)
        final.save(out_path, "JPEG", quality=95)
        log_step(f"сохранено → {out_path.name} (всего {time.monotonic() - t0:.0f}s)")
        return Result(path, "ok", str(out_path))
    except Exception as exc:  # noqa: BLE001 — report and keep batch going
        log_step(f"[red]ошибка[/red]: {exc}")
        return Result(path, "error", str(exc))


def run_pass(client: genai.Client, config: Config, images: list[Path], console: Console, label: str) -> list[Result]:
    results: list[Result] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=len(images))
        for image_path in images:
            results.append(process_one(client, config, image_path, console))
            progress.advance(task)
    return results


def print_summary(console: Console, results: dict[Path, Result]) -> None:
    ok = sum(1 for r in results.values() if r.status == "ok")
    skipped = sum(1 for r in results.values() if r.status == "skipped")
    errors = [r for r in results.values() if r.status == "error"]

    table = Table(title="Итог обработки слайдов")
    table.add_column("Метрика")
    table.add_column("Значение", justify="right")
    table.add_row("Всего найдено файлов", str(len(results)))
    table.add_row("Успешно обработано", str(ok))
    table.add_row("Пропущено (уже существовали)", str(skipped))
    table.add_row("Ошибки", str(len(errors)))
    console.print(table)

    if errors:
        console.print("\n[bold red]Файлы с ошибками:[/bold red]")
        for r in errors:
            console.print(f"  {r.path}: {r.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Пакетно переверстывает обучающие слайды: Gemini Vision анализирует "
            "исходный слайд, Imagen 3 генерирует чистый акварельный фон, Pillow "
            "накладывает исходный текст. Результат: <имя>_upd.jpg рядом с оригиналом."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Корневая директория для рекурсивного поиска слайдов (*.jpg, *.png)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перегенерировать файлы, даже если <имя>_upd.jpg уже существует",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Обработать только первые N слайдов (для тестов)",
    )
    parser.add_argument(
        "--font-bold",
        type=Path,
        default=None,
        help="Путь к жирному TTF-шрифту (по умолчанию: FONT_TITLE_PATH из .env)",
    )
    parser.add_argument(
        "--font-regular",
        type=Path,
        default=None,
        help="Путь к обычному TTF-шрифту (по умолчанию: FONT_BODY_PATH из .env)",
    )
    return parser.parse_args()


def main() -> int:
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    console = Console()
    args = parse_args()

    try:
        config = load_config(args)
        images = collect_images(config.input_dir)
        if config.limit is not None:
            images = images[:config.limit]
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[bold red]Ошибка:[/bold red] {exc}")
        return 1

    if not images:
        console.print(f"В {config.input_dir} не найдено слайдов для обработки.")
        return 0

    auth_mode = "Vertex Express" if config.use_vertex and config.vertex_api_key else (
        "Vertex ADC" if config.use_vertex else "Gemini API"
    )
    est_min = max(1, round(len(images) * 0.5))
    console.print(
        f"Слайдов: {len(images)} | vision: {config.vision_model} | "
        f"image: {config.imagen_model} | auth: {auth_mode} | "
        f"ориентир ~{est_min}–{est_min * 2} мин"
    )

    client = build_client(config)

    all_results: dict[Path, Result] = {}
    pending = images
    for attempt in range(1, MAX_RECONCILE_PASSES + 1):
        label = f"Обработка слайдов (попытка {attempt}/{MAX_RECONCILE_PASSES})"
        results = run_pass(client, config, pending, console, label)
        for result in results:
            all_results[result.path] = result
        pending = [r.path for r in results if r.status == "error"]
        if not pending:
            break
        console.print(f"[yellow]Повторный проход: {len(pending)} файл(ов) с ошибками[/yellow]")

    print_summary(console, all_results)
    return 1 if any(r.status == "error" for r in all_results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
