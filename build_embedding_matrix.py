"""
build_embedding_matrix.py

Скрипт для построения матрицы эмбеддингов W на основе заранее созданного словаря word2index
и модели русскоязычных векторов (например, RusVectōrēs).

Пример использования:
    python build_embedding_matrix.py \
        --input word2index.npz \
        --model rusvectores.model.bin \
        --output embedding_matrix.npz

Аргументы:
    --input     Путь к .npz файлу, содержащему словарь word2index (ключи: 'words' и 'indices').
    --model     Путь к бинарной модели в формате word2vec (RusVectores, .bin).
    --output    Путь для сохранения матрицы эмбеддингов в формате .npz.

Результат:
    Файл .npz с матрицей эмбеддингов W, где W[i] — это вектор для слова с индексом i в словаре.
"""

import numpy as np
import sys
import argparse
from gensim.models import KeyedVectors


def load_word2index(npz_path):
    """
    Загружает словарь word2index из .npz файла.

    :param npz_path: str
        Путь к .npz файлу, содержащему 'words' и 'indices'.
    :return: dict[str, int]
        Словарь word2index.
    """
    data = np.load(npz_path)
    words = data['words']
    indices = data['indices']
    return {word: int(idx) for word, idx in zip(words, indices)}


def build_embedding_matrix(word2index, model_path):
    """
    Строит матрицу эмбеддингов из модели и словаря.

    :param word2index: dict[str, int]
        Словарь word2index.
    :param model_path: str
        Путь к модели rusvectores в формате word2vec (.bin).
    :return: np.ndarray
        Матрица эмбеддингов W.
    """
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    embedding_dim = model.vector_size
    vocab_size = len(word2index)

    W = np.zeros((vocab_size, embedding_dim))

    not_found = 0
    for word, idx in word2index.items():
        if word in model:
            W[idx] = model[word]
        else:
            not_found += 1
            # Можно заменить на случайные значения, если нужно
            # W[idx] = np.random.uniform(-0.01, 0.01, embedding_dim)
    print(f"[+] Эмбеддинги построены. Не найдено слов: {not_found} из {vocab_size}")
    return W


def main():
    parser = argparse.ArgumentParser(description="Build embedding matrix W from word2index and RusVectores model")
    parser.add_argument('--input', required=True, help='Path to word2index .npz file')
    parser.add_argument('--model', required=True, help='Path to word2vec binary model (.bin)')
    parser.add_argument('--output', required=True, help='Path to output .npz file with embedding matrix')
    args = parser.parse_args()

    word2index = load_word2index(args.input)
    W = build_embedding_matrix(word2index, args.model)

    np.savez_compressed(args.output, W=W)
    print(f"[✓] Матрица эмбеддингов сохранена в {args.output}")


if __name__ == '__main__':
    main()
