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
from parsivar import Normalizer, spell_checker, SpellCheck
from xlsxwriter import Workbook
import pandas as pd
import numpy as np
import re

# *********** خواندن فایل csv برای گرفتن دیتا و انتقال آن به متغییر ***********
df = pd.read_csv(r"C:\Users\yousefi-pc\PycharmProjects\data.csv", encoding='utf-8',
                   index_col = False)

# ********* جداسازی دیتا فریم ها (همان گروه بندی ) با استفاده از یک ستون خاص *********
data_f1 = df[df['Suggestion'] == 1]
data_f2 = df[df['Suggestion'] == 2]
data_f3 = df[df['Suggestion'] == 3]

# ********** نوشتن در فایل اکسل و ساخت فایل مورد نظر *********
writer = pd.ExcelWriter(r"C:\Users\yousefi-pc\PycharmProjects\MarksData.xlsx", engine='xlsxwriter')

# ********** پر کردن شیت ها با استفاده از مقادیر متناطر در فایل اکسل و sheet های متناظر آن **********
data_f1.to_excel(writer, sheet_name='Suggestion__1')
data_f2.to_excel(writer, sheet_name='Suggestion__2')
data_f3.to_excel(writer, sheet_name='Suggestion__3')

# ********** مشخص کردن تعدا سطرهای جداشده از دیتا فریم ها **********
data_f11 = data_f1[:400]
data_f12 = data_f2[:400]
data_f13 = data_f3[:400]

# ********** الحاق کردن تمام دیتا فریم ها در sheet مورد نظر **********
df_append = [data_f11, data_f12, data_f13]
df_append = pd.concat(df_append)

# ********** شافل كردن ديتا فريم، منظور بهم ريختگي ديتا فريم بر اساس سطر هست **********
df_append = df_append.sample(frac=1).reset_index (drop=True)
df_append.to_excel( writer, sheet_name='Append_Suggestion')

# *********** ذخیره کردن تغییران اعمال شده بر روی فایل اکسل
# writer.save()

MarksData = pd.DataFrame(df_append)
mylist = MarksData['Text'].tolist()

print('******************** Tahlil *************************')

my_normalizer = Normalizer()  # from parsivar import Normalizer
myspell_checker = SpellCheck()  # from parsivar import SpellCheck

#  ********** حذف اعداد- كلمات بي اثر - علائم نگارشی - نرمالایز - فضاهای خالی -**************
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
mylist2 = []
for String in mylist:
    string1 = myspell_checker.spell_corrector(
                             my_normalizer.space_correction(
                                 (my_normalizer.normalize(String)).strip()))

    text_tokens = word_tokenize(string1)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('persion')]
    string1 = (" ").join(tokens_without_sw)
    no_punct = ""
    for char in string1:
        if char not in punctuations:
            no_punct = no_punct + char
    mylist2.append(no_punct)
#    print(no_punct)

# ********** اضافه کردن ستون clean به دیتا فریم جاری **********
MarksData['clean'] = mylist2
# ********** ثبت دیتا فریم تکمیلی در فایل اکسل و شیت مورد نظر **********
MarksData.to_excel(writer, sheet_name='Append_Suggestion')

# ********** ذخیره کردن تغییران اعمال شده بر روی فایل اکسل ***********
writer.save()

print('************ پایان کار پیش پردازش **************')

# ********************* تقسیم داده ها به دو بخش داده Test و Train برای بردار سازی ********************
Data_Train, Data_Test = train_test_split(MarksData, test_size=0.1)

# OR
print('***************** Vectorize **********************')

#Data_Train = MarksData.head(1000) # 1000 سطر اول از ديتا فريم
#Data_Test = MarksData.tail(200)   # 200 سطر آخر از ديتا فريم

#clean = pd.get_dummies(MarksData['Suggestion'])
# print(clean)
###################################################   ##############################################
tokenize = Tokenizer()
tokenize.fit_on_texts(Data_Train['clean'])
#b=tokenize.fit_on_texts(Data_Test['clean'])

encoded_xtrain = tokenize.texts_to_sequences(Data_Train['clean'])
encoded_test = tokenize.texts_to_sequences(Data_Test['clean'])
vocab_size = len(tokenize.word_index) + 1

# ********** تبدیل به لیست می کنیم که خروجی ها همه به صورت لیست باشند **********
encoded_xtrain1 = (pd.get_dummies(Data_Train['Suggestion'])).values.tolist()
encoded_test1 = (pd.get_dummies(Data_Test['Suggestion'])).values.tolist()

arr = np.array(encoded_xtrain1)
arr1 = np.array(encoded_test1)

# ********** همسان سازی طول بردارها به عبارتی یکسان‌سازی ابعاد بردار ها **********
Data_Train_p = pad_sequences(encoded_xtrain, maxlen=100, padding='post')
Data_Test_p = pad_sequences(encoded_test, maxlen=100, padding='post')
