# Planemo Biosignature Search Pipeline (JWST)

Автоматизированный комплекс для поиска биосигнатур (H₂O, O₃, O₂) в атмосферах планет-сирот по спектрам JWST.

---

## 🚀 Быстрый старт (Smart CLI)

Проект использует "Smart CLI" — скрипт сам определяет режим работы в зависимости от того, что вы ему подаёте на вход (папку или файл).

### Шаг 1: Скачать данные
```bash
python collect_data.py --ra 24.2354 --dec 9.5631 --download
```
*Данные сохранятся в `data/inputs/` (или `data/raw`).*

### Шаг 2: Анализ

**Вариант А: Авто-режим (Рекомендуемый)**
Просто укажите папку. Пайплайн сам найдет лучшие спектры, склеит NRS1+NRS2 (если есть) и создаст отчет.
```bash
python run_pipeline.py data/raw/
```
*Самостоятельно включит графики, JSON и сохранит в `outputs/<TARGET>/<RUN_ID>/`.*

**Вариант Б: Один файл**
Укажите конкретный FITS-файл. Название объекта подтянется из заголовка.
```bash
python run_pipeline.py my_spectrum_x1d.fits
```

---

## ⚙️ Поведение по умолчанию

Для того чтобы "просто работало", по умолчанию включено всё необходимое:

| Опция | Default | Как изменить |
| :--- | :--- | :--- |
| **Plotting** | **ON** | `--no-plot` |
| **Reports** | **ON (JSON+TXT)** | `--no-json` |
| **Merging** | **AUTO** | `--force-single` (игнорировать пары) |
| **Target Name**| **AUTO** (Header/Dir) | `--target-name "Name"` |
| **ML Analysis**  | **AUTO (if deps found)** | Automatic fallback if missing |

---

## 🧠 ML-Features (New!)

Пайплайн включает в себя "Sidecar" ML-модули, которые дополняют физический анализ:

1.  **AI Quality Assessment**: Оценивает качество спектра (gaps, stability, SNR) перед анализом.
2.  **Detection Confidence**: ML интерпретирует physical detections (H2O, O2, O3) и выдает уровень уверенности (LIKELY, UNCERTAIN, UNLIKELY).
3.  **Auto-Report**: Генерирует научное текстовое описание результатов ("Abstract", "Results", "Limitations") для вставки в статьи.

*ML работает в fail-safe режиме: ошибки ML не останавливают основной физический пайплайн.*

---

## 📂 Структура данных

Пайплайн наводит строгий порядок в файлах для воспроизводимости (Research Grade).

```text
/
├── data/
│   ├── inputs/               # Исходные данные
│   └── working/              # Промежуточные (склеенные спектры)
│
└── outputs/
    └── <TARGET_NAME>/        # Папка объекта
        └── <RUN_ID>/         # Отдельный запуск (Timestamp + Mode)
            ├── input/        # Копия спектра, использованного в анализе
            ├── plots/        # Графики (PNG)
            └── reports/      # Отчеты (summary.txt, results.json)
```

---

## 📖 Справочник CLI

### `run_pipeline.py`

```bash
python run_pipeline.py [INPUT] [OPTIONS]
```

**Аргументы:**
- `INPUT`: Путь к **файлу** (.fits) или **директории**.

**Опции:**
- `--target-name STR`: Принудительно задать имя объекта.
- `--no-plot`: Не генерировать графики.
- `--no-json`: Не сохранять JSON.
- `--force-single`: В режиме папки НЕ искать пары для склейки, а брать лучший одиночный файл.
- `--verbose`: Подробный лог.

### `collect_data.py`

Утилита для скачивания данных из MAST.

- `--target "NAME"` или `--ra`/`--dec`: Поиск.
- `--download`: Скачать найденное.
- `--radius`: Радиус поиска (deg).

---

## 🛠 Установка

Требуется Python 3.9+ и библиотеки:
```bash
pip install numpy scipy astropy matplotlib pandas astroquery
```

*Note: Для воспроизводимости результатов, `run_pipeline.py` автоматически архивирует используемый спектр в папку запуска.*
