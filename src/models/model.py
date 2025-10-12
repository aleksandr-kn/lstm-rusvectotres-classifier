import sys
import os

# На Windows поддержка GPU есть только на TF 2.10
# И необходимо напрямую указывать путь до CUDA библиотек
# (Должны быть установлены в системе).
# Рабочие версии: CUDA 11.2, CuDnn 8.1.0 (остальные не заводятся)
if sys.platform == "win32":
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

    import tensorflow as tf

    # Еще один костыль для Windows, связанный с тем что выкидавает
    # Ошибку нехватки памяти
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print("Could not set memory growth:", e)

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class LanguageModel:
    """
    Класс для загрузки данных, построения и обучения модели на основе LSTM с предобученными эмбеддингами.

    Атрибуты:
        model_file (str): Путь к файлу модели (пока не используется).
        epochs (int): Количество эпох обучения (текущий счётчик).
        word2index_vocab (dict): Словарь сопоставления слова к его индексу в словаре.
        embedding_matrix (np.ndarray): Матрица предобученных эмбеддингов. Представляет из себя словарь всех
        dataset (dict): Словарь с ключами 'X', 'y', 'raw_texts' для хранения датасета.
    """

    def __init__(self, model_file):
        self.model_file = model_file
        self.epochs = 0

        self.word2index_vocab = {} # Словарь слово -> index
        self.embedding_matrix = {} # Матрица весов

        self.dataset = {}

    def load_word2index_vocab_from_npz(self, path):
        """
        Загружает словарь word2index из .npz, где он сохранён как два массива: words и indices.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found: {path}")

        data = np.load(path)
        words = data['words']
        indices = data['indices']

        self.word2index_vocab = dict(zip(words, indices))

    def load_embedding_matrix_from_npz(self, path):
        """
        Загружает матрицу эмбеддингов из .npz файла, где она сохранена под ключом 'W'.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        data = np.load(path)
        self.embedding_matrix = data['W']

    def load_vectorized_dataset_from_csv(self, dataset_path, text_column='text', label_column='reasoning_label'):
        """
        Загружает и векторизует датасет из CSV.
        Если словарь и матрица эмбеддингов не загружены — загружает их из файлов.
        """

        # Загружаем словарь, если не загружен
        if self.word2index_vocab is None:
            raise ValueError("word2index_vocab не загружен")

        # Загружаем матрицу эмбеддингов, если не загружена
        if self.embedding_matrix is None:
            raise ValueError("embedding_matrix не загружен")

        dataset = {
            'X': [],
            'y': [],
            'raw_texts': []
        }

        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row[text_column].strip()
                label = int(row[label_column].strip())

                tokens = text.split()

                #todo делать в словаре word2index индекс для неизвестных слов, UNNWN например
                vector = [self.word2index_vocab.get(token, self.word2index_vocab.get('<UNK>', 0)) for token in tokens]

                dataset['X'].append(vector)
                dataset['y'].append(label)
                dataset['raw_texts'].append(text)

        self.dataset = dataset

    def check_distribution(self, texts):
        """
        Проверяет распределение текстов по количеству слов
        """
        # Считаем слова вместо символов
        text_lengths = [len(text.split()) for text in texts]

        plt.figure(figsize=(10, 6))
        plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue')
        plt.title('Распределение текстов по количеству слов')
        plt.xlabel('Количество слов в тексте')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"Mean: {np.mean(text_lengths):.1f}")
        print(f"Median: {np.median(text_lengths):.1f}")
        print(f"85% percentile: {np.percentile(text_lengths, 85):.1f}")
        print(f"95% percentile: {np.percentile(text_lengths, 95):.1f}")
        print(f"Max: {np.max(text_lengths)}")

        # Дополнительная полезная статистика
        print(f"Min: {np.min(text_lengths)}")
        print(f"Std: {np.std(text_lengths):.1f}")
        print(f"Текстов > 500 слов: {sum(1 for x in text_lengths if x > 500)}")

    def build_and_train_model(self, X, y, W, embedding_dim, max_len, num_classes, epochs=25, batch_size=32, bidirectional=True):
        # Паддинг последовательностей до max_len
        X_padded = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

        # Преобразуем метки в one-hot encoding
        y_cat = to_categorical(y, num_classes=num_classes)

        vocab_size = W.shape[0]

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[W], trainable=False,
                            input_length=max_len))

        # Добавляем LSTM с регуляризацией
        if bidirectional:
            model.add(Bidirectional(LSTM(
                64,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(1e-3),
                recurrent_regularizer=regularizers.l2(1e-3),
                bias_regularizer=regularizers.l2(1e-3)
            )))
        else:
            model.add(LSTM(
                64,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(1e-3),
                recurrent_regularizer=regularizers.l2(1e-3),
                bias_regularizer=regularizers.l2(1e-3)
            ))

        model.add(Dropout(0.4))

        model.add(
            Dense(
                num_classes,
                activation='softmax',
                kernel_regularizer=regularizers.l2(1e-3)
            )
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Разбиваем на train и val для оценки
        X_train, X_val, y_train, y_val = train_test_split(X_padded, y_cat, test_size=0.2, random_state=42)

        # Early stopping: останавливаем обучение, если val_loss не улучшается 4 эпох
        early_stop = EarlyStopping(
            monitor='val_loss',  # метрика для отслеживания
            patience=5,  # сколько эпох ждать улучшения
            restore_best_weights=True  # вернуть веса лучшей эпохи
        )

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

        # Оценка на валидационной выборке
        y_val_pred = model.predict(X_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        print(classification_report(y_val_true, y_val_pred_classes))

        return model

def main(data_dir: str):
    language_model = LanguageModel(model_file=None)

    print(f"[+] Подгружаем словарь и матрицу эмбеддингов из {data_dir}")

    # Пути до файлов
    word2index_path = os.path.join(data_dir, "word2index.npz")
    embedding_matrix_path = os.path.join(data_dir, "embedding_matrix.npz")
    dataset_csv_path = os.path.join(data_dir, "postagged_dataset.csv")

    # Загружаем словарь слово -> index
    language_model.load_word2index_vocab_from_npz(word2index_path)

    # Загружаем матрицу весов Embedding matrix
    language_model.load_embedding_matrix_from_npz(embedding_matrix_path)

    print(f"[+] Загружаем датасет из {data_dir}")

    # Загружаем датасет в формат "Предложение из датасета" -> [0, 1, 2]
    language_model.load_vectorized_dataset_from_csv(dataset_csv_path)

    embedding_dim = language_model.embedding_matrix.shape[1]  # размерность векторов

    print("[+] Проверяем размерность датасета:")
    # language_model.check_distribution(language_model.dataset['raw_texts'])

    print("[+] Строим и обучаем модель...")
    language_model.build_and_train_model(
        language_model.dataset['X'],
        language_model.dataset['y'],
        language_model.embedding_matrix,
        embedding_dim,
        150, # Выявлено по результатам распределения текстов, 85 процентиль кол-ва слов
        2
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model on RusVectores embeddings")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing word2index.npz and embedding_matrix.npz"
    )

    args = parser.parse_args()

    main(args.data_dir)
