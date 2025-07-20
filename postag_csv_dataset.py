#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
postag_csv_dataset.py

cкрипт производит pre-preocessing текстов для подачи в
модели rusvectores
нужно разобраться почему, но процент таких текстов (<10%)
"""

import sys
import re
import pandas as pd
import nltk
from tqdm import tqdm
import argparse
import os
import requests
from pymystem3 import Mystem

# Параметр для progress-bar
tqdm.pandas(desc="Lemmatizing and PosTagging")

# Загрузка nltk stopwords (один раз)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Глобальная переменная для русского набора стоп-слов
russian_stopwords = stopwords.words("russian")

def tag_mystem(
    text, mystem_instance, mapping=None, postags=True
):
    # если частеречные тэги не нужны (например, их нет в модели), выставьте postags=False
    # в этом случае на выход будут поданы только леммы

    processed = mystem_instance.analyze(text)
    tagged = []
    for w in processed:
        # Пропускаем, если анализатор не распознал токен
        if not w.get("analysis"):
            continue

        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(",")[0]
            pos = pos.split("=")[0].strip()
            if mapping:
                if pos in mapping:
                    pos = mapping[pos]  # здесь мы конвертируем тэги
                else:
                    pos = "X"  # на случай, если попадется тэг, которого нет в маппинге
            tagged.append(lemma.lower() + "_" + pos)
        except KeyError:
            continue  # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
    if not postags:
        tagged = [t.split("_")[0] for t in tagged]
    return tagged

def text_cleaning(text: str) -> str:
    # TODO возможно это не нужно, весь метод всмысле, пока не знаю

    """
    Очистка текста:
    - Приводит к нижнему регистру,
    - Удаляет все символы, кроме русских букв и пробелов,
    - Удаляет стоп-слова.
    Лемматизация не производится.
    """
    text = text.lower()
    # Оставляем только русские буквы и пробелы
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = text.split()
    # Удаляем стоп-слова
    tokens = [t for t in tokens if t not in russian_stopwords]

    return " ".join(tokens)

def custom_lemmatize_and_postag_with_mystem(text: str, mystem_instance, mapping: dict, postags: bool = True) -> list[str]:
    """
    Лемматизирует и размечает части речи в русском тексте с помощью Yandex Mystem.

    Вывод в формате, совместимом с RusVectōrēs: 'мама_NOUN мыть_VERB рама_NOUN'.
    При отключении postags=True вернёт только леммы.

    :param text: Русский текст (строка).
    :param mapping: (опц.) словарь для конвертации POS-тегов (например, в UPoS).
    :param postags: Включить части речи в вывод (по умолчанию True).
    :return: Список лемм или лемм с POS-тегами.
    """

    clean = text.strip()
    return tag_mystem(clean, mystem_instance, mapping=mapping)

def get_mystem2upos_mapping():
    """
    Возвращает маппинг mystem2upos
    """
    # Таблица преобразования частеречных тэгов Mystem в тэги UPoS:
    mapping_url = "https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map"

    mystem2upos = {}
    r = requests.get(mapping_url, stream=True)
    for pair in r.text.split("\n"):
        pair = pair.split()
        if len(pair) > 1:
            mystem2upos[pair[0]] = pair[1]

    return mystem2upos

# =============== MAIN ===============

def main():
    """
    Основная функция:
    """

    parser = argparse.ArgumentParser(description="Preprocess text dataset for rusvectores")
    parser.add_argument("input_csv", help="Path to input CSV file with 'text' and 'reasoning_label'")
    parser.add_argument("output_csv", help="Path to output CSV file")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Файл не найден: {args.input_csv}")
        sys.exit(1)

    # Загружаем данные
    df = pd.read_csv(args.input_csv, encoding="utf-8")
    df = df[df["reasoning_label"].isin([0, 1])].copy()
    df["reasoning_label"] = df["reasoning_label"].astype(int)

    if len(df) == 0:
        print("Нет валидных данных для обучения!")
        return

    # препроцессинг текста
    mystem2upos_mapping = get_mystem2upos_mapping()
    mystem_instance = Mystem()
    df["text"] = (
        df["text"]
        .progress_apply(
            lambda text: custom_lemmatize_and_postag_with_mystem(
                text,
                mystem_instance,
                mapping=mystem2upos_mapping,
                postags=True
            )
        )
    )

    # Убираем пустые строки и строки, содержащие только пробелы
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    # сохраняем в csv
    df.to_csv(args.output_csv, index=False)

    print(f"Всего примеров после препроцессинга: {len(df)}")
    sys.exit(0)

if __name__ == "__main__":
    main()