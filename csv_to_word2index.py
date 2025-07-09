"""
csv_to_word2index_npz.py

Скрипт для построения словаря word2index из текстовой колонки CSV-файла
и сохранения его вместе с индексами в формате .npz.

Пример использования:
    python csv_to_word2index_npz.py --input data.csv --output word2index.npz --column text

Аргументы:
    --input     Путь к входному CSV-файлу.
    --output    Путь к выходному .npz-файлу, в который будет сохранён словарь.
    --column    Название колонки с текстами в CSV-файле.

Результат:
    .npz файл с массивами 'words' (слова) и 'indices' (индексы).
"""

import pandas as pd
import argparse
import numpy as np
from gensim.utils import simple_preprocess

def normalize_word(word):
    """
    Нормализует POS-тег в слове, преобразуя его к верхнему регистру.

    Например, если слово имеет вид 'кот_noun', функция вернёт 'кот_NOUN'.
    """
    if '_' in word:
        base, pos = word.rsplit('_', 1)
        return f"{base}_{pos.upper()}"
    return word

def build_word2index(texts):
    """
    Строит словарь word2index из списка текстов.

    :param texts: list[str]
        Список строк, содержащих текстовые данные (например, из колонки CSV).
    :return: dict[str, int]
        Словарь, в котором каждому уникальному слову присвоен уникальный индекс.
    """

    vocab = set()
    for text in texts:
        if isinstance(text, str):
            tokens = simple_preprocess(text)
            normalized_tokens = [normalize_word(t) for t in tokens]
            vocab.update(normalized_tokens)
    word2index = {word: idx for idx, word in enumerate(sorted(vocab))}
    return word2index

def main():
    parser = argparse.ArgumentParser(description="Build word2index dict from CSV text column and save to npz")
    parser.add_argument('--input', required=True, help='Path to CSV file')
    parser.add_argument('--output', required=True, help='Path to output .npz file')
    parser.add_argument('--column', default='text', help='Text column name (default: text)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in CSV. Available columns: {df.columns.tolist()}")

    texts = df[args.column].astype(str).tolist()
    word2index = build_word2index(texts)

    # Сохраняем в .npz файл: два массива - слова и индексы
    words = np.array(list(word2index.keys()))
    indices = np.array(list(word2index.values()))

    np.savez_compressed(args.output, words=words, indices=indices)

    print(f"[+] Saved word2index with {len(word2index)} words to {args.output}")

if __name__ == "__main__":
    main()
