# Spectra — JWST Molecular Detection Pipeline

![tests](https://github.com/kikobin/Spectra_Analisys/actions/workflows/ci.yml/badge.svg)

Автоматизированный пайплайн для поиска молекулярных сигнатур (H₂O, CH₄, CO, CO₂, O₂, O₃)
в атмосферах субзвёздных объектов по спектрам JWST/NIRSpec и JWST/MIRI.

---

## Portfolio (что показывает проект)

- **End-to-end аналитический пайплайн**: ingest FITS → (опционально) merge NRS1+NRS2 → continuum fit → измерение полос → confidence → отчёт.
- **Устойчивость к реальным данным**: авто-определение колонок/единиц, fail-safe ML слой, структурированные выходы.
- **Воспроизводимость**: демо-режим без скачивания данных + тесты.

## Быстрый старт

### 0. Демо без данных (рекомендовано)

Создаёт синтетический FITS в `data/inputs/demo/` и прогоняет полный пайплайн:

```bash
python run_pipeline.py --demo --outdir outputs_demo
```

Результаты: `outputs_demo/DEMO/<RUN_ID>/` (plot + `results.json` + `summary.txt`).

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
# или установить пакет в режиме разработки:
pip install -e .
```

### 2. Скачать данные с MAST

```bash
python scripts/collect_data.py --ra 24.2354 --dec 9.5631 --download
```

Данные сохранятся в `data/raw/`.

### 3. Анализ

**Авто-режим** (папка — сам найдёт NRS1+NRS2, склеит и проанализирует):
```bash
python run_pipeline.py data/raw/
```

**Один файл:**
```bash
python run_pipeline.py my_spectrum_x1d.fits
```

Результаты сохраняются в `outputs/<TARGET>/<RUN_ID>/`.

---

## Пример результатов

![demo spectrum](docs/assets/spectrum_demo.png)

- Пример JSON: `docs/sample_results.json`
- Пример текста отчёта: `docs/sample_summary.txt`

## Структура проекта

```
Spectra_Analisys/
│
├── config.py              # Единый источник истины: диапазоны полос,
│                          # пороги SNR, пороги confidence
│
├── run_pipeline.py        # Главный CLI: авто-поиск, склейка, анализ, отчёт
├── io_fits.py             # Чтение/запись FITS с авто-определением единиц
├── continuum.py           # Подбор BlackBody-континуума (NMAD, sigma-clipping)
├── features.py            # Измерение полос: depth, EQW, SNR, FAP
├── merge.py               # Оценка и склейка NRS1+NRS2 с разрешением overlap
├── plotting.py            # Двухпанельный график (flux + residual)
├── organize_io.py         # Структура выходных директорий
│
├── ml/                    # ML-слой (работает в fail-safe режиме)
│   ├── __init__.py        # Публичное API: assess_quality, ConfidenceAssessor, ReportWriter
│   ├── quality.py         # Оценка качества данных: SNR, coverage, gaps, smoothness
│   ├── detection_confidence.py  # Confidence score с физическими априорями
│   └── report_writer.py   # Генерация научного текста (Abstract / Results / Limitations)
│
├── scripts/               # Вспомогательные скрипты (не часть пайплайна)
│   └── collect_data.py    # Загрузка спектров с MAST
│
├── tests/                 # Тесты
│   ├── conftest.py        # Pytest-фикстуры (синтетические спектры)
│   ├── fixtures/
│   │   └── make_dummy_fits.py   # Генерация тестового FITS-файла
│   └── test_report_writer.py
│
├── data/
│   ├── inputs/            # Входные спектры по объектам
│   ├── raw/               # Скачанные данные JWST
│   └── working/           # Промежуточные файлы (склеенные спектры)
│
├── outputs/               # Результаты: outputs/<TARGET>/<RUN_ID>/
│   └── <TARGET>/
│       └── <RUN_ID>/
│           ├── input/     # Копия использованного спектра
│           ├── plots/     # spectrum.png
│           └── reports/   # summary.txt, results.json
│
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Опции CLI (`run_pipeline.py`)

| Опция | По умолчанию | Описание |
|---|---|---|
| `INPUT` | `.` | Файл `.fits` или директория |
| `--demo` | off | Синтетический FITS (без скачивания данных) |
| `--demo-seed INT` | `42` | Seed для `--demo` |
| `--target-name STR` | из заголовка | Имя объекта |
| `--no-plot` | off | Отключить графики |
| `--no-json` | off | Отключить JSON-отчёт |
| `--force-single` | off | Не искать пары NRS1+NRS2 |
| `--verbose` | off | Подробный лог |

---

## Модули

### `config.py`
Все пороги и определения полос. Менять здесь — применяется везде.

### `features.py`
Для каждой полосы вычисляет:
- **depth** — глубина поглощения с коррекцией на спектральное разрешение R
- **depth_err** — формальная погрешность (σ/√n)
- **EQW** — эквивалентная ширина (мкм)
- **SNR** — max(peak SNR, matched-filter SNR)
- **FAP** — вероятность ложной тревоги (t-распределение для n < 30)
- **contamination** — список молекул, чьи полосы пересекаются с континуум-окнами

### `ml/`
Работает поверх физического анализа, ошибки не останавливают пайплайн.

| Модуль | Функция |
|---|---|
| `quality.py` | SNR, coverage, gap-penalty, smoothness → quality_score ∈ [0,1] |
| `detection_confidence.py` | STRONG / LIKELY / MARGINAL / WEAK / NOT DETECTED + confidence ∈ [0,1] |
| `report_writer.py` | Научный текст: Abstract, Results, Limitations |

---

## Запуск тестов

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
