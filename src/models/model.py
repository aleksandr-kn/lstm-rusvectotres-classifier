import os
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
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

    def build_and_train_model(self, X, y, W, embedding_dim, max_len, num_classes, epochs=10, batch_size=32):
        # Паддинг последовательностей до max_len
        X_padded = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

        # Преобразуем метки в one-hot encoding
        y_cat = to_categorical(y, num_classes=num_classes)

        vocab_size = W.shape[0]

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[W], trainable=False,
                            input_length=max_len))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Разбиваем на train и val для оценки
        X_train, X_val, y_train, y_val = train_test_split(X_padded, y_cat, test_size=0.2, random_state=42)

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        # Оценка на валидационной выборке
        y_val_pred = model.predict(X_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true = np.argmax(y_val, axis=1)

        print(classification_report(y_val_true, y_val_pred_classes))

        return model

def main():
    language_model = LanguageModel(model_file=None)
    print("Подготовка датасета...")

    # Загружаем словарь слово -> index
    language_model.load_word2index_vocab_from_npz('data/texts/rusvectores_word2index.npz')

    # Загружаем матрицу весов Embedding matrix
    language_model.load_embedding_matrix_from_npz('data/texts/rusvectores_embedding_matrix.npz')

    # Загружаем датасет в формат "Предложение из датасета" -> [0, 1, 2]
    language_model.load_vectorized_dataset_from_csv('data/texts/clean-progress-07-08_3_rusvectores.csv')

    embedding_dim = language_model.embedding_matrix.shape[1]  # размерность векторов

    language_model.build_and_train_model(language_model.dataset['X'], language_model.dataset['y'], language_model.embedding_matrix, embedding_dim, 50, 2)

if __name__ == "__main__":
    main()
