print ( "------------Import library at first------------" )

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, convolutional, MaxPooling1D, Dense, Flatten
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from parsivar import Normalizer, spell_checker
import pandas as pd
import numpy as np
import re

# *********** خواندن فایل csv برای گرفتن دیتا و انتقال آن به متغییر ***********
df = pd.read_csv( r"C:\Users\yousefi-pc\PycharmProjects\data.csv", encoding='utf-8',
                   index_col=False)

