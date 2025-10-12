### Новый файл README, чтобы было норм понятно что вообще происходит.

## Структура проекта

``` Structure
lstm-rusvectores-classified/
├── data/ # Сырые и промежуточные данные
├── pretrained/  внешние модели и веса, в частности модели rusvectores (каждая в отдельной директории)
├── notebooks/ # Jupyter-ноутбуки для экспериментов и анализа
├── src/ # Исходный код
│ ├── preprocessing/ # Скрипты токенизации, POS-теггинга и очистки текста
│ ├── embeddings/ # Построение матриц эмбеддингов и векторов слов
│ ├── utils/ # Вспомогательные функции и утилиты
│ └── pipeline.py # Главный сценарий пайплайна, объединяющий все шаги
├── models/ # Сохранённые обученные модели
├── outputs/ # Результаты работы пайплайна (npz, словари, отчёты)
├── requirements.txt # Список зависимостей Python
└── README.md # Документация проекта
```

## Важные замечания

### Запуск на windows и проблемы с кодировкой 'utf-8'
Pipeline работает только при запуске с кодировкой консоли 'utf-8'
На Windows Python по умолчанию использует cp1251, поэтому русские символы в выходных файлах могут отображаться как � при postagging (UDPipe / rusvectores).
Git Bash:

export PYTHONIOENCODING=utf-8
python src/pipeline.py --csv './data/texts/dataset_as_of_2025-10-05.csv' --column 'text'

cmd.exe:

chcp 65001
python src\pipeline.py --csv ".\data\texts\dataset_as_of_2025-10-05.csv" --column "text"

На Linux/WSL настройка не нужна. Если символы всё ещё ломаются, используйте WSL/Linux.

###  Использование TensorFlow с GPU на Windows

**Поддерживаемая конфигурация:**
- TensorFlow 2.10.0 (последняя версия с официальной поддержкой GPU на Windows)
- CUDA 11.2.0 
- cuDNN 8.1.0

**Проверено на:**
- NVIDIA GeForce RTX 3060
- Драйвер NVIDIA 576.52
- Windows 10/11

