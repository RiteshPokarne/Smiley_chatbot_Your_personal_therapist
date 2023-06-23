import numpy as np
import joblib
import re
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import string
df2=pd.read_csv("predict/static/models/response.csv")
lemmatizer = WordNetLemmatizer()
def get_text(str_text):
    # print(str_text)
    input_text  = [str_text]
    df_input = pd.DataFrame(input_text,columns=['questions'])
    df_input
    return df_input

from tensorflow.keras.models import load_model
model1 = load_model('predict/static/models/model2.h5')
tokenizer_t = joblib.load('predict/static/models/tokenizer_t.pkl')
vocab = joblib.load('predict/static/models/vocab.pkl')

def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

def remove_stop_words_for_input(tokenizer,df,feature):
    doc_without_stopwords = []
    entry = df[feature][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df

def encode_input_text(tokenizer_t,df,feature):
    t = tokenizer_t
    entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=10, padding='post')
    return padded

def get_pred(model,encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred

def bot_precausion(df_input,pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred

def get_response(df2,pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]

def predict_bot(df_input):
    # tokenizer_t = joblib.load('tokenizer_t.pkl')
    # vocab = joblib.load('vocab.pkl')

    # df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
    df_input=get_text(df_input)
    encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

    pred = get_pred(model1,encoded_input)
    pred = bot_precausion(df_input,pred)

    response = get_response(df2,pred)
    return response

