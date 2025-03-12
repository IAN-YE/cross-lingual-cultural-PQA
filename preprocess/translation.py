from langdetect import detect
import re
import json
import tqdm
from google.cloud import translate_v2 as translate


def detect_language(text):
    language = detect(text)
    return language

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in data.items():
        English_reviews = value['Western_reviews']
        for review in English_reviews:
            language = detect_language(review['review'])
            review['language'] = language
    return data

def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """


    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]

def translate_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in tqdm.tqdm(data.items()):
        info = value['info']
        translate_entry = {}
        for item in info:
            for key, value in item.items():
                new_key = key + '_en'
                try:
                    translate_entry[new_key] = translate_text(target='en', text=value)
                except:
                    translate_entry[new_key] = None
            item.clear()
            item.update(translate_entry)

    return data

def translate_reviews(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in tqdm.tqdm(data.items()):
        Chinese_reviews = value['Chinese reviews']
        Western_reviews = value['Western_reviews']
        for review in Chinese_reviews:
            review['review_en'] = translate_text(target='en', text=review['review'])
        for review in Western_reviews:
            review['review_en'] = translate_text(target='en', text=review['review'])

import requests

API_KEY = "AIzaSyCDpBW8kH_rjkqvB6ktYoodGzPhog3F498"  # 这里替换成你的 API Key
URL = "https://translation.googleapis.com/language/translate/v2"


def translate_text(text, target_language="zh"):
    params = {
        "q": text,
        "target": target_language,
        "format": "text",
        "key": API_KEY  # 直接使用 API Key
    }

    response = requests.get(URL, params=params)
    result = response.json()

    if "data" in result and "translations" in result["data"]:
        return result["data"]["translations"][0]["translatedText"]
    else:
        return f"Error: {result}"


# main
if __name__ == '__main__':
    text_to_translate = "Hello, how are you?"
    translated_text = translate_text(text_to_translate, "zh")
    print(f"翻译结果: {translated_text}")

    # path = '../data/merge_data.json'
    # data = read_file(file_path=path)
    # with open('../data/merge_data_language.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    # path_language = '../data/merge_data_language.json'
    # with open(path_language, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    #
    # language = {}
    # for key, value in data.items():
    #     English_reviews = value['Western_reviews']
    #     for review in English_reviews:
    #         if review['language'] not in language:
    #             language[review['language']] = 1
    #         else:
    #             language[review['language']] += 1
    #
    # print(language)





