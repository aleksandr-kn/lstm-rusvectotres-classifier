# 🔤 Векторизация текстов и классификация с использованием RusVectores и LSTM

## 📌 Описание проекта

Этот проект демонстрирует, как преобразовать тексты на русском языке в числовое представление с помощью предобученных эмбеддингов RusVectores и обучить рекуррентную модель (LSTM) для задачи классификации или аналогичных задач обработки текста.

---

## 🧱 Архитектура модели
Текст → Индексы слов → Embedding (с матрицей RusVectores) → LSTM → Dense (классификация)

---

## 📝 Подробные шаги

### 1. Построение словаря `word2index`

Создаётся отображение от слова к уникальному индексу:

```python
word2index = {
    'кот': 0,
    'собака': 1,
    'дом': 2, 
    'идти': 3,
    # ...
}
```

### 2. Формирование матрицы эмбеддингов W
``` python
W = [
    [0.12, -0.03, 0.55],    # вектор слова 'кот'
    [-0.44, 0.22, 0.31],    # вектор слова 'собака'
    # ...
]
```

Размерность: (vocab_size, embedding_dim)
Эта матрица загружается в слой Embedding модели как фиксированные веса.

### 3. Векторизация текстов
Текст преобразуется в последовательность индексов:
"собака идти дом" → [1, 3, 2]

Слова вне словаря заменяются на <UNK> или индекс по умолчанию.

### 4. Архитектура модели и обучение
```python
model = Sequential([
    Embedding(input_dim=vocab_size, 
             output_dim=embedding_dim, 
             weights=[W], 
             trainable=False),
    LSTM(64),
    Dense(num_classes, activation='softmax')
])
```
- Эмбеддинги фиксированы (не обучаются)
- LSTM извлекает последовательные зависимости
- Выходной слой — классификатор по классам

## 💡 Важные замечания
- Слова, отсутствующие в RusVectores, заменяются на <UNK>
- Используется паддинг для выравнивания длин последовательностей
- Оценка модели проводится с train/val разбиением
- Метрики рассчитываются с помощью sklearn
- Требуется датасет в формате CSV с POS-разметкой: собака -> собака_NOUN
- Разметить можно с помощью предоставляемых rusvectores скриптов, например: rus_preprocessing_mystem.py
- Использовались модели: ruscorpora_upos_cbow_300_20_2019 и ruwikiruscorpora_upos_cbow_300_10_2021
Отсутствуют в git репозитории из-за их размера. Можно поместить их при необходимости в /data/models/rusvectores/
В основном ruscorpora_upos_cbow_300_20_2019 (в ней больше слов). Возможно подойдут и другие, требуемый алгоритм Continuous Bag-of-Words 

## 🚀 Запуск проекта

#### Установите зависимости:
``` bash 
pip install -r requirements.txt
```
#### Подготовьте данные:
- Словарь и матрицу эмбеддингов (.npz)
- Подготовьте словарь и матрицу эмбеддингов в формате .npz. Можно использовать скрипты build_embedding_matrix.py, csv_to_word2index.py
- Поместите подготовленный датасет (каждое слово должно быть с частью речи, например собака_NOUN) в формате CSV с колонками text и reasoning_label.
- Запустите скрипт model.py для обучения модели.

#### Запустите обучение:
``` python
python model.py
```

## 📂 Структура проекта

```text
/data
    /texts
        rusvectores_word2index.npz       # Словарь слов и индексов
        rusvectores_embedding_matrix.npz # Матрица весов
        clean-progress-07-08_3_rusvectores.csv       # Исходный датасет
        clean-progress-07-08_3_rusvectores_POS.csv   # Размеченный датасет
model.py              # Основной скрипт
language_model.py     # Вспомогательные функции
README.md
requirements.txt
```