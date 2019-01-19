from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import urllib.parse, urllib.request
import hashlib
import urllib
import random
import json
import time


tqdm.pandas()
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated
# Lets load our data

from translate import Translator



appid = '20151113000005349'#自己去百度api找到id和secret
secretKey = 'osubCEzlGjzvw8qdQc41'

url_baidu = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
def translateBaidu(text, f='en', t='zh'):
    salt = random.randint(32768, 65536)
    sign = appid + text + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    url = url_baidu + '?appid=' + appid + '&q=' + urllib.parse.quote(text) + '&from=' + f + '&to=' + t + '&salt=' + str(salt) + '&sign=' + sign
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    data = json.loads(content)
    result = str(data['trans_result'][0]['dst'])
    return result
# comment = "Textblob is amazingly simple to use. What a great fun!"
# print(translateBaidu(comment))

def mytranslate1(comment,language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")
        print('ok')

    translation1 = translateBaidu(comment,f='en', t=language)
    translation2 = translateBaidu(translation1,f=language, t='en')
    if translation2!=comment:
        return str(translation2)
    else:
        pass
# print(mytranslate1(comment,'zh'))

NAN_WORD = "_NAN_"

def mytranslate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")
        print('ok')

    translator1 = Translator(from_lang="english", to_lang=language)
    translation1 = translator1.translate(comment)
    translator2 = Translator(from_lang=language, to_lang="english")
    translation2 = translator2.translate(translation1)
    if translation2!=comment:
        return str(translation2)
    else:
        pass


train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
## fill up the missing values
train_X = train_df["question_text"].fillna("_##_").values
test_X = test_df["question_text"].fillna("_##_").values


y = train_df.target.values
pos_samples = train_X[y == 1][::3]
# trans_data = pd.DataFrame(columns = ["question_text", "p_sku", "sale", "sku"]) #创建一个空的dataframe

translated_data = []
for comment in pos_samples[:50]:
    # translated_data.append(mytranslate1(comment, "de"))
    translated_data.append(comment)
train_X1 = list(train_X[0:3])
train_X1.extend(translated_data)
y_over = np.array(list(y) + [1] * len(pos_samples))

pass